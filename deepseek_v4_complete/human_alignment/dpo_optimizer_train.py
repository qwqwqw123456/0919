"""
DPO (Direct Preference Optimization) 训练模块

DPO 是一种直接优化偏好的方法，无需训练奖励模型。DPO 通过直接最大化
chosen 响应相对于 rejected 响应的概率来优化语言模型。

DPO 损失函数:
    L = -log(σ(r_θ(x, y_c) - r_θ(x, y_r)))
    
其中 r_θ(x, y) = β * (log_π_θ(y|x) - log_π_ref(y|x))

DPO 训练优势:
1. 无需单独训练奖励模型
2. 训练过程更稳定
3. 可以利用 KL 散度约束防止策略偏离参考模型太远
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ConstantLR, LinearLR
import json
import os
import logging
from pathlib import Path
from tqdm import tqdm
import copy
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    """DPO 训练配置参数"""
    beta: float = 0.1
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 1000
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    max_prompt_length: int = 512
    max_response_length: int = 1536
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    output_dir: str = "./dpo_output"
    use_amp: bool = True
    gradient_clip_norm: float = 1.0
    reference_free: bool = False
    label_smoothing: float = 0.0
    sync_ref_params: bool = True
    ref_batch_size: int = 1
    warmup_ratio: float = 0.1


class DPODataset(Dataset):
    """
    DPO 训练数据集
    
    每条数据包含: prompt, chosen_response, rejected_response
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        max_prompt_length: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.examples = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """加载偏好数据"""
        examples = []
        
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    examples.append(data)
        elif isinstance(data_path, list):
            examples = data_path
        else:
            raise ValueError(f"Invalid data_path: {data_path}")
        
        logger.info(f"Loaded {len(examples)} DPO examples from {data_path}")
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def _encode_with_prompt(
        self,
        prompt: str,
        response: str
    ) -> Dict[str, torch.Tensor]:
        """
        编码 prompt + response 序列
        
        格式: <bos> prompt <sep> response <eos>
        """
        bos_id = getattr(self.tokenizer, 'bos_token_id', None) or 151643
        sep_id = getattr(self.tokenizer, 'sep_token_id', None) or 151643
        eos_id = getattr(self.tokenizer, 'eos_token_id', None) or 151643
        
        prompt_tokens = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
            max_length=self.max_prompt_length
        )
        response_tokens = self.tokenizer.encode(
            response,
            add_special_tokens=False,
            max_length=self.max_length - len(prompt_tokens) - 4
        )
        
        full_tokens = [bos_id] + prompt_tokens + [sep_id] + response_tokens + [eos_id]
        
        if len(full_tokens) > self.max_length:
            full_tokens = full_tokens[:self.max_length]
        
        prompt_len = len(prompt_tokens) + 2
        
        return {
            'input_ids': torch.tensor(full_tokens, dtype=torch.long),
            'prompt_len': prompt_len,
            'response_len': len(response_tokens)
        }
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]
        
        prompt = example.get('prompt', '')
        chosen = example.get('chosen', example.get('chosen_response', ''))
        rejected = example.get('rejected', example.get('rejected_response', ''))
        
        chosen_enc = self._encode_with_prompt(prompt, chosen)
        rejected_enc = self._encode_with_prompt(prompt, rejected)
        
        return {
            'prompt': prompt,
            'chosen_input_ids': chosen_enc['input_ids'],
            'chosen_prompt_len': chosen_enc['prompt_len'],
            'rejected_input_ids': rejected_enc['input_ids'],
            'rejected_prompt_len': rejected_enc['prompt_len']
        }


def dpo_collate_fn(
    batch: List[Dict[str, Any]],
    pad_token_id: int = 0
) -> Dict[str, torch.Tensor]:
    """
    整理 DPO 批次数据
    """
    chosen_input_ids = torch.nn.utils.rnn.pad_sequence(
        [item['chosen_input_ids'] for item in batch],
        batch_first=True,
        padding_value=pad_token_id
    )
    
    rejected_input_ids = torch.nn.utils.rnn.pad_sequence(
        [item['rejected_input_ids'] for item in batch],
        batch_first=True,
        padding_value=pad_token_id
    )
    
    chosen_prompt_lens = torch.tensor(
        [item['chosen_prompt_len'] for item in batch],
        dtype=torch.long
    )
    rejected_prompt_lens = torch.tensor(
        [item['rejected_prompt_len'] for item in batch],
        dtype=torch.long
    )
    
    return {
        'chosen_input_ids': chosen_input_ids,
        'rejected_input_ids': rejected_input_ids,
        'chosen_prompt_lens': chosen_prompt_lens,
        'rejected_prompt_lens': rejected_prompt_lens
    }


class DPOLoss(nn.Module):
    """
    DPO 损失函数
    
    DPO 损失基于 Bradley-Terry 模型，但直接在策略模型和参考模型之间计算。
    
    损失函数:
        L = -E[(x,y_c,y_r) ~ D] [log σ(β * (log π_θ(y_c|x) - log π_θ(y_r|x) - 
                                        (log π_ref(y_c|x) - log π_ref(y_r|x))))]
    """
    
    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算 DPO 损失
        
        Args:
            policy_chosen_logps: 策略模型对 chosen 响应的 log 概率
            policy_rejected_logps: 策略模型对 rejected 响应的 log 概率
            reference_chosen_logps: 参考模型对 chosen 响应的 log 概率
            reference_rejected_logps: 参考模型对 rejected 响应的 log 概率
            
        Returns:
            loss: DPO 损失值
            metrics: 训练指标字典
        """
        chosen_log_ratio = policy_chosen_logps - reference_chosen_logps
        rejected_log_ratio = policy_rejected_logps - reference_rejected_logps
        
        logits = self.beta * (chosen_log_ratio - rejected_log_ratio)
        
        if self.label_smoothing > 0:
            loss = -F.logsigmoid(logits) * (1 - self.label_smoothing)
            loss = loss - self.label_smoothing * F.logsigmoid(-logits)
        else:
            loss = -F.logsigmoid(logits)
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        with torch.no_grad():
            chosen_probs = torch.sigmoid(logits)
            accuracy = (chosen_probs > 0.5).float().mean()
        
        metrics = {
            'dpo_loss': loss.item(),
            'logits_mean': logits.mean().item(),
            'logits_std': logits.std().item(),
            'chosen_log_ratio_mean': chosen_log_ratio.mean().item(),
            'rejected_log_ratio_mean': rejected_log_ratio.mean().item(),
            'accuracy': accuracy.item(),
            'chosen_prob_mean': chosen_probs.mean().item()
        }
        
        return loss, metrics


class AdaptiveDPO_loss(nn.Module):
    """
    自适应 DPO 损失
    
    根据偏好对的奖励差异自适应调整损失权重。
    """
    
    def __init__(
        self,
        beta: float = 0.1,
        margin_threshold: float = 0.5
    ):
        super().__init__()
        self.beta = beta
        self.margin_threshold = margin_threshold
    
    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算自适应 DPO 损失"""
        chosen_log_ratio = policy_chosen_logps - reference_chosen_logps
        rejected_log_ratio = policy_rejected_logps - reference_rejected_logps
        
        ref_diff = (reference_chosen_logps - reference_rejected_logps).detach()
        
        margin = torch.abs(ref_diff)
        weights = torch.where(
            margin < self.margin_threshold,
            torch.ones_like(margin),
            torch.exp(-margin + self.margin_threshold)
        )
        
        logits = self.beta * (chosen_log_ratio - rejected_log_ratio)
        
        loss = -weights * F.logsigmoid(logits)
        loss = loss.mean()
        
        metrics = {
            'adaptive_dpo_loss': loss.item(),
            'weights_mean': weights.mean().item(),
            'weights_std': weights.std().item()
        }
        
        return loss, metrics


class DPOTrainer:
    """
    DPO 训练器
    
    支持:
    - 冻结参考模型参数
    - 混合精度训练
    - 梯度累积
    - 学习率调度
    - 检查点保存与恢复
    """
    
    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: nn.Module,
        tokenizer,
        config: DPOConfig,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        device: Optional[torch.device] = None
    ):
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.config = config
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.policy_model.to(self.device)
        self.reference_model.to(self.device)
        
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        self.loss_fn = DPOLoss(
            beta=config.beta,
            label_smoothing=config.label_smoothing
        )
        
        self.optimizer = AdamW(
            self.policy_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
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
        self.best_loss = float('inf')
        
        self.train_loader = None
        self.eval_loader = None
        
        os.makedirs(config.output_dir, exist_ok=True)
    
    def _create_dataloaders(self):
        """创建数据加载器"""
        if self.train_dataset is not None:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=1,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                collate_fn=lambda x: dpo_collate_fn(
                    x,
                    pad_token_id=self.tokenizer.pad_token_id or 0
                )
            )
        
        if self.eval_dataset is not None:
            self.eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                collate_fn=lambda x: dpo_collate_fn(
                    x,
                    pad_token_id=self.tokenizer.pad_token_id or 0
                )
            )
    
    def _compute_log_probabilities(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        prompt_lens: torch.Tensor
    ) -> torch.Tensor:
        """
        计算序列的 log 概率
        
        只计算 response 部分的 log 概率（prompt_len 之后的部分）
        """
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False
        )
        
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        
        log_probs = F.log_softmax(logits, dim=-1)
        
        response_log_probs = []
        for i in range(input_ids.size(0)):
            seq_len = attention_mask[i].sum().item()
            prompt_len = min(prompt_lens[i].item(), seq_len - 1)
            
            start_idx = prompt_len
            end_idx = seq_len - 1
            
            if end_idx <= start_idx:
                response_log_probs.append(torch.tensor(0.0, device=input_ids.device))
                continue
            
            token_log_probs = log_probs[i, start_idx:end_idx]
            target_ids = input_ids[i, start_idx + 1:end_idx + 1]
            
            token_log_probs = token_log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
            
            seq_log_prob = token_log_probs.sum()
            response_log_probs.append(seq_log_prob)
        
        return torch.stack(response_log_probs)
    
    def _compute_log_probabilities_v2(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        prompt_lens: torch.Tensor
    ) -> torch.Tensor:
        """
        优化的 log 概率计算
        """
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False
        )
        
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        
        log_probs = F.log_softmax(logits, dim=-1)
        
        input_ids_shifted = input_ids[:, 1:].contiguous()
        log_probs_shifted = log_probs[:, :-1, :].contiguous()
        
        seq_lens = attention_mask.sum(dim=1)
        prompt_lens_clamped = torch.clamp(prompt_lens, min=0, max=seq_lens - 1)
        
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        response_log_probs_list = []
        
        for b in range(batch_size):
            prompt_len = prompt_lens_clamped[b].item()
            seq_len = seq_lens[b].item()
            
            start_idx = prompt_len
            end_idx = seq_len - 1
            
            if end_idx <= start_idx:
                response_log_probs_list.append(torch.tensor(0.0, device=device))
                continue
            
            token_log_probs = log_probs_shifted[b, start_idx:end_idx]
            target_ids = input_ids_shifted[b, start_idx:end_idx]
            
            token_log_probs = token_log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
            
            seq_log_prob = token_log_probs.sum()
            response_log_probs_list.append(seq_log_prob)
        
        return torch.stack(response_log_probs_list)
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """执行单个训练步骤"""
        self.policy_model.train()
        
        chosen_input_ids = batch['chosen_input_ids'].to(self.device)
        rejected_input_ids = batch['rejected_input_ids'].to(self.device)
        chosen_prompt_lens = batch['chosen_prompt_lens'].to(self.device)
        rejected_prompt_lens = batch['rejected_prompt_lens'].to(self.device)
        
        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
            policy_chosen_logps = self._compute_log_probabilities_v2(
                self.policy_model, chosen_input_ids, chosen_prompt_lens
            )
            policy_rejected_logps = self._compute_log_probabilities_v2(
                self.policy_model, rejected_input_ids, rejected_prompt_lens
            )
            
            with torch.no_grad():
                ref_chosen_logps = self._compute_log_probabilities_v2(
                    self.reference_model, chosen_input_ids, chosen_prompt_lens
                )
                ref_rejected_logps = self._compute_log_probabilities_v2(
                    self.reference_model, rejected_input_ids, rejected_prompt_lens
                )
            
            loss, metrics = self.loss_fn(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps
            )
            
            scaled_loss = loss / self.config.gradient_accumulation_steps
        
        self.scaler.scale(scaled_loss).backward()
        
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(),
                max_norm=self.config.gradient_clip_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()
        
        self.global_step += 1
        
        return metrics
    
    def _eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """执行单个评估步骤"""
        self.policy_model.eval()
        
        with torch.no_grad():
            chosen_input_ids = batch['chosen_input_ids'].to(self.device)
            rejected_input_ids = batch['rejected_input_ids'].to(self.device)
            chosen_prompt_lens = batch['chosen_prompt_lens'].to(self.device)
            rejected_prompt_lens = batch['rejected_prompt_lens'].to(self.device)
            
            policy_chosen_logps = self._compute_log_probabilities_v2(
                self.policy_model, chosen_input_ids, chosen_prompt_lens
            )
            policy_rejected_logps = self._compute_log_probabilities_v2(
                self.policy_model, rejected_input_ids, rejected_prompt_lens
            )
            
            ref_chosen_logps = self._compute_log_probabilities_v2(
                self.reference_model, chosen_input_ids, chosen_prompt_lens
            )
            ref_rejected_logps = self._compute_log_probabilities_v2(
                self.reference_model, rejected_input_ids, rejected_prompt_lens
            )
            
            _, metrics = self.loss_fn(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps
            )
        
        return metrics
    
    def _save_checkpoint(self, step: int, metrics: Dict[str, float], is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'step': step,
            'policy_model_state_dict': self.policy_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__,
            'best_loss': self.best_loss
        }
        
        checkpoint_path = os.path.join(self.config.output_dir, f'checkpoint_{step}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        latest_path = os.path.join(self.config.output_dir, 'checkpoint_latest.pt')
        torch.save(checkpoint, latest_path)
        
        if is_best:
            best_path = os.path.join(self.config.output_dir, 'checkpoint_best.pt')
            torch.save(checkpoint, best_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.policy_model.load_state_dict(checkpoint['policy_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.global_step = checkpoint['step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def evaluate(self) -> Dict[str, float]:
        """在评估集上评估模型"""
        if self.eval_loader is None:
            return {}
        
        self.policy_model.eval()
        total_metrics = {
            'dpo_loss': 0.0,
            'accuracy': 0.0,
            'logits_mean': 0.0
        }
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.eval_loader:
                metrics = self._eval_step(batch)
                for key in total_metrics:
                    if key in metrics:
                        total_metrics[key] += metrics[key]
                num_batches += 1
        
        avg_metrics = {
            f'eval_{key}': total_metrics[key] / num_batches
            for key in total_metrics
        }
        
        return avg_metrics
    
    def train(self):
        """执行完整的训练流程"""
        if self.train_loader is None and self.train_dataset is not None:
            self._create_dataloaders()
        
        if self.train_loader is None:
            raise ValueError("No training dataset provided")
        
        logger.info(f"Starting DPO training for {self.config.max_steps} steps")
        logger.info(f"Device: {self.device}")
        logger.info(f"Beta: {self.config.beta}")
        
        progress_bar = tqdm(total=self.config.max_steps, desc="DPO Training")
        
        while self.global_step < self.config.max_steps:
            for batch in self.train_loader:
                if self.global_step >= self.config.max_steps:
                    break
                
                metrics = self._train_step(batch)
                progress_bar.update(1)
                
                if self.global_step % self.config.log_interval == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({
                        'loss': f"{metrics['dpo_loss']:.4f}",
                        'acc': f"{metrics['accuracy']:.4f}",
                        'lr': f"{lr:.2e}"
                    })
                
                if self.global_step % self.config.eval_interval == 0 and self.eval_loader is not None:
                    eval_metrics = self.evaluate()
                    logger.info(f"Step {self.global_step} | Eval: {eval_metrics}")
                
                if self.global_step % self.config.save_interval == 0:
                    self._save_checkpoint(self.global_step, metrics)
        
        progress_bar.close()
        logger.info("DPO training completed!")
        
        self._save_checkpoint(self.global_step, metrics)
        
        return self.policy_model


class DPOwithRewardModel(DPOTrainer):
    """
    结合奖励模型的 DPO 训练器
    
    使用预训练的奖励模型来辅助 DPO 训练，提供更稳定的训练信号。
    """
    
    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: nn.Module,
        reward_model: nn.Module,
        tokenizer,
        config: DPOConfig,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        device: Optional[torch.device] = None
    ):
        super().__init__(
            policy_model,
            reference_model,
            tokenizer,
            config,
            train_dataset,
            eval_dataset,
            device
        )
        
        self.reward_model = reward_model
        self.reward_model.to(self.device)
        self.reward_model.eval()
    
    def _compute_reward_scores(
        self,
        input_ids: torch.Tensor,
        prompt_lens: torch.Tensor
    ) -> torch.Tensor:
        """计算奖励分数"""
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        with torch.no_grad():
            outputs = self.reward_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_hidden=False
            )
            
            if isinstance(outputs, torch.Tensor):
                rewards = outputs
            else:
                rewards = outputs[0]
            
            return rewards


class UnlikelihoodDPO_loss(nn.Module):
    """
    结合 Unlikelihood 的 DPO 损失
    
    在 DPO 损失基础上添加 unlikelihood 惩罚项，减少生成不期望 token 的概率。
    """
    
    def __init__(
        self,
        beta: float = 0.1,
        ul_weight: float = 0.1,
        ul_margin: float = 0.05
    ):
        super().__init__()
        self.beta = beta
        self.ul_weight = ul_weight
        self.ul_margin = ul_margin
    
    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        rejected_logits: Optional[torch.Tensor] = None,
        negative_token_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算结合 Unlikelihood 的 DPO 损失"""
        chosen_log_ratio = policy_chosen_logps - reference_chosen_logps
        rejected_log_ratio = policy_rejected_logps - reference_rejected_logps
        
        logits = self.beta * (chosen_log_ratio - rejected_log_ratio)
        dpo_loss = -F.logsigmoid(logits).mean()
        
        metrics = {
            'dpo_loss': dpo_loss.item(),
            'ul_loss': 0.0
        }
        
        if rejected_logits is not None and negative_token_ids is not None:
            neg_token_probs = rejected_logits.gather(dim=-1, index=negative_token_ids)
            ul_loss = neg_token_probs.clamp(max=self.ul_margin).mean()
            
            metrics['ul_loss'] = ul_loss.item()
            total_loss = dpo_loss + self.ul_weight * ul_loss
        else:
            total_loss = dpo_loss
        
        metrics['total_loss'] = total_loss.item()
        
        return total_loss, metrics


def create_dpo_trainer(
    policy_model: nn.Module,
    reference_model: nn.Module,
    tokenizer,
    config: Optional[DPOConfig] = None
) -> DPOTrainer:
    """
    创建 DPO 训练器的工厂函数
    """
    if config is None:
        config = DPOConfig()
    
    return DPOTrainer(
        policy_model=policy_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        config=config
    )


def generate_dpo_training_data(
    num_samples: int,
    output_path: str,
    prompts: Optional[List[str]] = None,
    quality_range: Tuple[float, float] = (0.0, 1.0)
) -> List[Dict[str, Any]]:
    """
    生成合成的 DPO 训练数据
    
    Args:
        num_samples: 样本数量
        output_path: 输出文件路径
        prompts: 可选的 prompt 列表
        quality_range: 响应质量分数范围
        
    Returns:
        生成的样本列表
    """
    import random
    
    if prompts is None:
        prompts = [
            "Explain the concept of machine learning.",
            "What are the benefits of renewable energy?",
            "How does blockchain technology work?",
            "Describe the process of photosynthesis.",
            "What is the theory of relativity?"
        ]
    
    response_templates = [
        "This is a {quality} answer that provides {depth} information about the topic.",
        "Let me explain that {quality}. The key points are {points}.",
        "In {depth} terms, this relates to {aspect} and involves {process}.",
        "The {quality} way to understand this is through {analogy}."
    ]
    
    quality_levels = ["excellent", "good", "basic", "poor"]
    
    samples = []
    
    for i in range(num_samples):
        prompt = random.choice(prompts)
        
        qualities = [
            random.uniform(*quality_range),
            random.uniform(*quality_range)
        ]
        
        if qualities[0] >= qualities[1]:
            chosen_quality = qualities[0]
            rejected_quality = qualities[1]
        else:
            chosen_quality = qualities[1]
            rejected_quality = qualities[0]
        
        chosen_response = f"Here is an excellent answer: {random.choice(response_templates).format(quality='excellent', depth='detailed', points='three main aspects', aspect='fundamental concepts', process='multiple steps')}"
        rejected_response = f"This is a basic response: {random.choice(response_templates).format(quality='basic', depth='simple', points='one point', aspect='related topics', process='a single step')}"
        
        sample = {
            'prompt': prompt,
            'chosen': chosen_response,
            'rejected': rejected_response,
            'quality_chosen': chosen_quality,
            'quality_rejected': rejected_quality
        }
        samples.append(sample)
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"Generated {num_samples} DPO training samples to {output_path}")
    return samples


if __name__ == '__main__':
    logger.info("DPO (Direct Preference Optimization) Training Module")
    logger.info("Import this module to use the DPOTrainer class")
