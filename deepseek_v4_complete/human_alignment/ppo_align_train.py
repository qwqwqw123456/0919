"""
PPO (Proximal Policy Optimization) 对齐训练模块

PPO 是一种策略优化算法，用于 RLHF (Reinforcement Learning from Human Feedback) 流程中。
PPO 通过限制策略更新的幅度来保证训练的稳定性。

核心组件:
1. Actor (策略模型): 生成响应
2. Critic (价值模型): 估计状态价值
3. Reward Model: 评估响应的奖励分数
4. Reference Model: 提供 KL 散度约束

PPO 优势:
1. 训练稳定，梯度更新有保障
2. 可以处理复杂的奖励信号
3. 支持在线学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import json
import os
import logging
from pathlib import Path
from tqdm import tqdm
import math
import copy
from collections import deque
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """PPO 训练配置参数"""
    learning_rate: float = 1e-5
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    ppo_epochs: int = 4
    mini_batch_size: int = 1
    gamma: float = 1.0
    lam: float = 0.95
    max_steps: int = 10000
    max_response_length: int = 512
    max_prompt_length: int = 512
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    output_dir: str = "./ppo_output"
    use_amp: bool = True
    kl_penalty: float = 0.1
    reward_scale: float = 1.0
    warmup_ratio: float = 0.1
    use_value_network: bool = True
    use_reference: bool = True
    reference_kl_weight: float = 0.1


@dataclass
class RolloutData:
    """Rollout 数据结构"""
    prompts: List[str]
    prompt_ids: torch.Tensor
    prompt_mask: torch.Tensor
    generated_ids: torch.Tensor
    generated_mask: torch.Tensor
    ref_log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    advantages: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None


class ValueNetwork(nn.Module):
    """
    价值网络
    
    估计状态的价值函数 V(s)，用于计算 advantage。
    """
    
    def __init__(self, base_model: nn.Module, hidden_size: Optional[int] = None):
        super().__init__()
        self.base_model = base_model
        
        if hidden_size is None:
            hidden_size = base_model.config.d_model if hasattr(base_model, 'config') else 7168
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1, bias=False)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化价值头权重"""
        for module in self.value_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """估计价值"""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_hidden=True
        )
        
        if isinstance(outputs, tuple):
            hidden_states = outputs[-1]
        else:
            hidden_states = outputs
        
        values = self.value_head(hidden_states).squeeze(-1)
        
        return values[:, -1]


class ActorCriticModel(nn.Module):
    """
    Actor-Critic 模型
    
    同时输出策略概率和价值估计。
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: Optional[int] = None
    ):
        super().__init__()
        self.base_model = base_model
        
        if hidden_size is None:
            hidden_size = base_model.config.d_model if hasattr(base_model, 'config') else 7168
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1, bias=False)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.value_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            logits: 语言模型 logits
            values: 价值估计
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_hidden=True
        )
        
        if isinstance(outputs, tuple):
            hidden_states = outputs[-1]
        else:
            hidden_states = outputs
        
        values = self.value_head(hidden_states).squeeze(-1)
        
        return outputs[0] if isinstance(outputs, tuple) else outputs, values[:, -1]


class PPODataset(Dataset):
    """
    PPO 训练数据集
    
    用于存储 prompts 和相关元数据。
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_prompt_length: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.prompts = self._load_prompts(data_path)
    
    def _load_prompts(self, data_path: str) -> List[str]:
        """加载 prompts"""
        prompts = []
        
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    prompt = data.get('prompt', data.get('text', ''))
                    prompts.append(prompt)
        elif isinstance(data_path, list):
            prompts = data_path
        else:
            raise ValueError(f"Invalid data_path: {data_path}")
        
        logger.info(f"Loaded {len(prompts)} prompts from {data_path}")
        return prompts
    
    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {'prompt': self.prompts[idx]}


class PPOLoss(nn.Module):
    """
    PPO 损失函数
    
    包含策略损失、价值损失和熵正则项。
    
    PPO 裁剪目标:
        L^CLIP(θ) = -E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]
    
    其中 r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) 是概率比。
    """
    
    def __init__(
        self,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01
    ):
        super().__init__()
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
    
    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算 PPO 损失
        
        Args:
            log_probs: 新策略的对数概率
            old_log_probs: 旧策略的对数概率
            values: 价值估计
            old_values: 旧的价值估计
            returns: 回报
            advantages: 优势估计
            mask: 有效 token 的掩码
            
        Returns:
            total_loss: PPO 总损失
            metrics: 训练指标字典
        """
        ratio = torch.exp(log_probs - old_log_probs)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio,
            1 - self.clip_epsilon,
            1 + self.clip_epsilon
        ) * advantages
        
        policy_loss = -torch.min(surr1, surr2)
        
        if mask is not None:
            policy_loss = (policy_loss * mask).sum() / mask.sum()
        else:
            policy_loss = policy_loss.mean()
        
        if mask is not None:
            value_pred_clipped = old_values + torch.clamp(
                values - old_values,
                -self.clip_epsilon,
                self.clip_epsilon
            )
            vf_losses = (values - returns) ** 2
            vf_losses_clipped = (value_pred_clipped - returns) ** 2
            value_loss = torch.max(vf_losses, vf_losses_clipped)
            value_loss = (value_loss * mask).sum() / mask.sum()
        else:
            value_pred_clipped = old_values + torch.clamp(
                values - old_values,
                -self.clip_epsilon,
                self.clip_epsilon
            )
            value_loss = torch.max(
                (values - returns) ** 2,
                (value_pred_clipped - returns) ** 2
            ).mean()
        
        metrics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'ratio_mean': ratio.mean().item(),
            'ratio_max': ratio.max().item(),
            'advantages_mean': advantages.mean().item(),
            'values_mean': values.mean().item()
        }
        
        total_loss = policy_loss + self.value_loss_coef * value_loss
        
        return total_loss, metrics


class PPOTrainer:
    """
    PPO 训练器
    
    支持:
    - Actor-Critic 架构
    - Rollout 生成
    - 奖励计算
    - PPO 损失优化
    - KL 散度约束
    """
    
    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: Optional[nn.Module],
        value_model: Optional[nn.Module],
        tokenizer,
        reward_model: Optional[nn.Module],
        config: PPOConfig,
        train_dataset: Optional[Dataset] = None,
        device: Optional[torch.device] = None
    ):
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.value_model = value_model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.config = config
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.policy_model.to(self.device)
        if self.reference_model is not None:
            self.reference_model.to(self.device)
            self.reference_model.eval()
        if self.value_model is not None:
            self.value_model.to(self.device)
        if self.reward_model is not None:
            self.reward_model.to(self.device)
            self.reward_model.eval()
        
        self.ppo_loss = PPOLoss(
            clip_epsilon=config.clip_epsilon,
            value_loss_coef=config.value_loss_coef,
            entropy_coef=config.entropy_coef
        )
        
        self.optimizer = AdamW(
            self.policy_model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        warmup_steps = int(config.max_steps * config.warmup_ratio)
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=warmup_steps,
            T_mult=2,
            eta_min=config.learning_rate * 0.1
        )
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)
        
        self.global_step = 0
        self.best_reward = 0.0
        
        self.train_dataset = train_dataset
        self.train_loader = None
        
        os.makedirs(config.output_dir, exist_ok=True)
    
    def _create_dataloader(self):
        """创建数据加载器"""
        if self.train_dataset is not None:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=1,
                shuffle=True,
                num_workers=0,
                pin_memory=True
            )
    
    @torch.no_grad()
    def _generate_rollout(
        self,
        prompts: List[str],
        max_length: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        生成 Rollout
        
        Args:
            prompts: prompt 列表
            max_length: 最大生成长度
            temperature: 采样温度
            top_p: top-p 采样参数
            
        Returns:
            generated_ids: 生成的 token IDs
            generated_texts: 生成的文本
        """
        self.policy_model.eval()
        
        all_generated_ids = []
        all_generated_texts = []
        
        for prompt in prompts:
            prompt_tokens = self.tokenizer.encode(
                prompt,
                add_special_tokens=True,
                max_length=self.config.max_prompt_length,
                truncation=True
            )
            
            input_ids = torch.tensor([prompt_tokens], dtype=torch.long).to(self.device)
            
            generated_ids = input_ids.clone()
            
            for _ in range(max_length):
                outputs = self.policy_model(generated_ids)
                
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                logits = logits[:, -1, :] / temperature
                
                probs = F.softmax(logits, dim=-1)
                
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    sorted_indices_to_remove = cumsum_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        2, sorted_indices, sorted_indices_to_remove
                    )
                    probs[indices_to_remove] = 0
                    
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
            
            all_generated_ids.append(generated_ids[0])
            generated_text = self.tokenizer.decode(
                generated_ids[0][len(prompt_tokens):],
                skip_special_tokens=True
            )
            all_generated_texts.append(generated_text)
        
        max_len = max(ids.size(0) for ids in all_generated_ids)
        
        padded_ids = []
        for ids in all_generated_ids:
            padding = torch.zeros(
                max_len - ids.size(0),
                dtype=ids.dtype,
                device=ids.device
            )
            padded_ids.append(torch.cat([ids, padding]))
        
        generated_ids_batch = torch.stack(padded_ids)
        
        return generated_ids_batch, all_generated_texts
    
    @torch.no_grad()
    def _compute_rewards(
        self,
        prompts: List[str],
        responses: List[str]
    ) -> torch.Tensor:
        """
        计算奖励分数
        
        Args:
            prompts: prompt 列表
            responses: 响应列表
            
        Returns:
            rewards: 奖励分数
        """
        if self.reward_model is None:
            return torch.zeros(len(prompts), device=self.device)
        
        self.reward_model.eval()
        
        rewards = []
        
        for prompt, response in zip(prompts, responses):
            sep_id = getattr(self.tokenizer, 'sep_token_id', None) or 151643
            bos_id = getattr(self.tokenizer, 'bos_token_id', None) or 151643
            eos_id = getattr(self.tokenizer, 'eos_token_id', None) or 151643
            
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
            
            full_tokens = [bos_id] + prompt_tokens + [sep_id] + response_tokens + [eos_id]
            
            input_ids = torch.tensor([full_tokens], dtype=torch.long).to(self.device)
            attention_mask = torch.ones_like(input_ids)
            
            outputs = self.reward_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            if isinstance(outputs, torch.Tensor):
                reward = outputs[0, 0].item()
            else:
                reward = outputs[0]
            
            rewards.append(reward * self.config.reward_scale)
        
        return torch.tensor(rewards, device=self.device)
    
    @torch.no_grad()
    def _compute_reference_log_probs(
        self,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """计算参考模型的 log 概率"""
        if self.reference_model is None:
            return torch.zeros(input_ids.size(0), device=self.device)
        
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        outputs = self.reference_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        
        log_probs = F.log_softmax(logits, dim=-1)
        
        ref_log_probs = []
        for i in range(input_ids.size(0)):
            seq_len = attention_mask[i].sum().item()
            seq_log_probs = log_probs[i, :seq_len-1]
            target_ids = input_ids[i, 1:seq_len]
            
            token_log_probs = seq_log_probs.gather(
                dim=-1,
                index=target_ids.unsqueeze(-1)
            ).squeeze(-1)
            
            ref_log_probs.append(token_log_probs.mean())
        
        return torch.stack(ref_log_probs)
    
    @torch.no_grad()
    def _estimate_values(
        self,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """估计价值"""
        if self.value_model is None:
            return torch.zeros(input_ids.size(0), device=self.device)
        
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        outputs = self.value_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        return outputs[:, -1]
    
    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        gamma: float = 1.0,
        lam: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算 GAE (Generalized Advantage Estimation)
        
        Args:
            rewards: 奖励
            values: 价值估计
            gamma: 折扣因子
            lam: GAE lambda
            
        Returns:
            advantages: 优势估计
            returns: 回报估计
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            next_value = values[t]
        
        return advantages, returns
    
    def _compute_kl_penalty(
        self,
        policy_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor
    ) -> torch.Tensor:
        """计算 KL 散度惩罚"""
        kl = policy_log_probs - ref_log_probs
        return kl.mean()
    
    def _ppo_train_step(
        self,
        input_ids: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """执行单个 PPO 训练步骤"""
        self.policy_model.train()
        
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
            outputs = self.policy_model(input_ids, attention_mask)
            
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            log_probs = F.log_softmax(logits, dim=-1)
            
            seq_log_probs = []
            for i in range(input_ids.size(0)):
                seq_len = attention_mask[i].sum().item()
                seq_lp = log_probs[i, :seq_len-1]
                target_ids = input_ids[i, 1:seq_len]
                
                token_log_probs = seq_lp.gather(
                    dim=-1,
                    index=target_ids.unsqueeze(-1)
                ).squeeze(-1)
                
                if mask is not None:
                    seq_log_probs.append((token_log_probs * mask[i, :len(token_log_probs)]).sum() / mask[i, :len(token_log_probs)].sum())
                else:
                    seq_log_probs.append(token_log_probs.mean())
            
            policy_log_probs = torch.stack(seq_log_probs)
            
            if self.value_model is not None:
                values = self.value_model(input_ids, attention_mask)
            else:
                values = torch.zeros_like(policy_log_probs)
            
            advantages, returns = self._compute_gae(
                rewards,
                values,
                gamma=self.config.gamma,
                lam=self.config.lam
            )
            
            old_values = values.detach()
            
            loss, metrics = self.ppo_loss(
                policy_log_probs,
                old_log_probs,
                values,
                old_values,
                returns,
                advantages,
                mask
            )
            
            if self.config.use_reference and self.reference_model is not None:
                kl_penalty = self._compute_kl_penalty(policy_log_probs, ref_log_probs)
                loss = loss + self.config.reference_kl_weight * kl_penalty
                metrics['kl_penalty'] = kl_penalty.item()
            
            scaled_loss = loss / self.config.ppo_epochs
        
        self.scaler.scale(scaled_loss).backward()
        
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(),
            max_norm=self.config.max_grad_norm
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self.scheduler.step()
        
        self.global_step += 1
        
        return metrics
    
    def _save_checkpoint(self, step: int, metrics: Dict[str, float]):
        """保存检查点"""
        checkpoint = {
            'step': step,
            'policy_model_state_dict': self.policy_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__
        }
        
        checkpoint_path = os.path.join(self.config.output_dir, f'checkpoint_{step}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        latest_path = os.path.join(self.config.output_dir, 'checkpoint_latest.pt')
        torch.save(checkpoint, latest_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.policy_model.load_state_dict(checkpoint['policy_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.global_step = checkpoint['step']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def train(self):
        """执行完整的 PPO 训练流程"""
        if self.train_loader is None and self.train_dataset is not None:
            self._create_dataloader()
        
        if self.train_loader is None:
            raise ValueError("No training dataset provided")
        
        logger.info(f"Starting PPO training for {self.config.max_steps} steps")
        logger.info(f"Device: {self.device}")
        logger.info(f"Clip epsilon: {self.config.clip_epsilon}")
        
        progress_bar = tqdm(total=self.config.max_steps, desc="PPO Training")
        
        prompts_iterator = iter(self.train_loader)
        
        while self.global_step < self.config.max_steps:
            try:
                batch = next(prompts_iterator)
            except StopIteration:
                prompts_iterator = iter(self.train_loader)
                batch = next(prompts_iterator)
            
            prompts = batch['prompt']
            
            generated_ids, generated_texts = self._generate_rollout(
                prompts,
                max_length=self.config.max_response_length
            )
            
            rewards = self._compute_rewards(prompts, generated_texts)
            
            if self.global_step % self.config.log_interval == 0:
                metrics = {
                    'reward_mean': rewards.mean().item(),
                    'reward_std': rewards.std().item(),
                    'policy_loss': 0.0
                }
                progress_bar.set_postfix({
                    'reward': f"{rewards.mean().item():.4f}",
                    'std': f"{rewards.std().item():.4f}"
                })
            
            input_ids = generated_ids.to(self.device)
            
            with torch.no_grad():
                old_log_probs = []
                ref_log_probs = []
                
                attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
                
                for i in range(input_ids.size(0)):
                    seq_len = attention_mask[i].sum().item()
                    
                    outputs = self.policy_model(input_ids[i:i+1], attention_mask[i:i+1])
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                    
                    log_probs = F.log_softmax(logits[0, :seq_len-1], dim=-1)
                    target_ids = input_ids[i, 1:seq_len]
                    token_log_probs = log_probs.gather(
                        dim=-1,
                        index=target_ids.unsqueeze(-1)
                    ).squeeze(-1)
                    old_log_probs.append(token_log_probs.mean())
                
                old_log_probs = torch.stack(old_log_probs).to(self.device)
                
                ref_log_probs = self._compute_reference_log_probs(input_ids)
            
            for _ in range(self.config.ppo_epochs):
                metrics = self._ppo_train_step(
                    input_ids,
                    old_log_probs,
                    ref_log_probs,
                    rewards
                )
                
                progress_bar.update(1)
                
                if self.global_step % self.config.log_interval == 0:
                    progress_bar.set_postfix({
                        'policy_loss': f"{metrics['policy_loss']:.4f}",
                        'value_loss': f"{metrics['value_loss']:.4f}"
                    })
            
            if self.global_step % self.config.save_interval == 0:
                self._save_checkpoint(self.global_step, metrics)
        
        progress_bar.close()
        logger.info("PPO training completed!")
        
        self._save_checkpoint(self.global_step, metrics)
        
        return self.policy_model


class RewardShapingWrapper:
    """
    奖励塑形包装器
    
    在基础奖励上添加多种奖励信号，如:
    - 长度惩罚
    - 格式惩罚
    - 重复惩罚
    - KL 散度惩罚
    """
    
    def __init__(
        self,
        base_reward_fn: Callable,
        length_weight: float = 0.0,
        format_weight: float = 0.0,
        repetition_weight: float = 0.0,
        kl_weight: float = 0.1
    ):
        self.base_reward_fn = base_reward_fn
        self.length_weight = length_weight
        self.format_weight = format_weight
        self.repetition_weight = repetition_weight
        self.kl_weight = kl_weight
    
    def __call__(
        self,
        prompt: str,
        response: str,
        ref_response: Optional[str] = None,
        **kwargs
    ) -> float:
        """计算塑形后的奖励"""
        base_reward = self.base_reward_fn(prompt, response, **kwargs)
        
        length_reward = self._compute_length_reward(response)
        
        format_reward = self._compute_format_reward(response)
        
        repetition_reward = self._compute_repetition_reward(response)
        
        total_reward = base_reward
        total_reward += self.length_weight * length_reward
        total_reward += self.format_weight * format_reward
        total_reward += self.repetition_weight * repetition_reward
        
        if ref_response is not None and self.kl_weight > 0:
            kl_penalty = self._compute_kl_divergence(response, ref_response)
            total_reward -= self.kl_weight * kl_penalty
        
        return total_reward
    
    def _compute_length_reward(self, response: str) -> float:
        """长度奖励（惩罚过长或过短的响应）"""
        words = len(response.split())
        if 50 <= words <= 200:
            return 0.0
        elif words < 50:
            return -0.1 * (50 - words) / 50
        else:
            return -0.01 * (words - 200) / 100
    
    def _compute_format_reward(self, response: str) -> float:
        """格式奖励（奖励包含清晰结构的响应）"""
        score = 0.0
        
        if response.count('\n') >= 2:
            score += 0.1
        
        if any(marker in response for marker in ['1.', '2.', '3.', '-', '*']):
            score += 0.1
        
        if response.startswith(('The', 'First', 'In', 'To', 'For')):
            score += 0.05
        
        return score
    
    def _compute_repetition_reward(self, response: str) -> float:
        """重复惩罚"""
        words = response.lower().split()
        if len(words) < 3:
            return 0.0
        
        unique_words = len(set(words))
        repetition_ratio = 1 - unique_words / len(words)
        
        return -repetition_ratio
    
    def _compute_kl_divergence(self, response: str, ref_response: str) -> float:
        """计算 KL 散度（简化的基于长度的估计）"""
        return abs(len(response) - len(ref_response)) / max(len(ref_response), 1)


def create_ppo_trainer(
    policy_model: nn.Module,
    tokenizer,
    config: PPOConfig,
    reference_model: Optional[nn.Module] = None,
    value_model: Optional[nn.Module] = None,
    reward_model: Optional[nn.Module] = None,
    train_dataset: Optional[Dataset] = None
) -> PPOTrainer:
    """
    创建 PPO 训练器的工厂函数
    """
    return PPOTrainer(
        policy_model=policy_model,
        reference_model=reference_model,
        value_model=value_model,
        tokenizer=tokenizer,
        reward_model=reward_model,
        config=config,
        train_dataset=train_dataset
    )


if __name__ == '__main__':
    logger.info("PPO (Proximal Policy Optimization) Alignment Training Module")
    logger.info("Import this module to use the PPOTrainer class")
