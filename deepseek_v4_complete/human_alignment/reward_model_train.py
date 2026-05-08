"""
Reward Model 训练模块
使用 Bradley-Terry 损失函数训练奖励模型，用于评估文本质量

Bradley-Terry 模型的核心思想：给定一对偏好数据 (x, y_winner, y_loser)，
模型学习使得 r(x, y_winner) > r(x, y_loser) 的概率最大化。

损失函数: L = -log(sigmoid(r_winner - r_loser))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import json
import os
import logging
from pathlib import Path
from tqdm import tqdm
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RewardModelConfig:
    """奖励模型配置参数"""
    hidden_size: int = 7168
    num_layers: int = 2
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 10000
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    output_dir: str = "./reward_model_output"
    eval_ratio: float = 0.1
    use_amp: bool = True
    warmup_ratio: float = 0.1


class RewardModel(nn.Module):
    """
    奖励模型基类
    基于预训练语言模型，在最后添加一个奖励头输出标量分数
    
    架构:
        Input -> Pretrained LM -> Mean Pooling -> Linear -> Reward Score
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: Optional[int] = None,
        reward_head_hidden_size: Optional[int] = None,
        use_mean_pooling: bool = True
    ):
        super().__init__()
        self.base_model = base_model
        self.use_mean_pooling = use_mean_pooling
        
        if hidden_size is None:
            hidden_size = base_model.config.d_model if hasattr(base_model, 'config') else 7168
        
        if reward_head_hidden_size is None:
            reward_head_hidden_size = hidden_size
        
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, reward_head_hidden_size),
            nn.GELU(),
            nn.Linear(reward_head_hidden_size, 1, bias=False)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化奖励头权重"""
        for module in self.reward_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs，形状 [batch_size, seq_len]
            attention_mask: 注意力掩码，形状 [batch_size, seq_len]
            return_hidden: 是否返回隐藏状态
            
        Returns:
            rewards: 奖励分数，形状 [batch_size]
            hidden_states: 隐藏状态（可选）
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
        
        if self.use_mean_pooling:
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_embeddings / sum_mask
            else:
                pooled = torch.mean(hidden_states, dim=1)
        else:
            pooled = hidden_states[:, -1, :]
        
        pooled = self.pooler(pooled)
        pooled = self.layer_norm(pooled)
        
        rewards = self.reward_head(pooled).squeeze(-1)
        
        if return_hidden:
            return rewards, hidden_states
        return rewards


class RewardModelWithScalarHead(nn.Module):
    """
    带有标量头的奖励模型变体
    直接在语言模型 logits 上添加奖励头
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        bias_vector: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.base_model = base_model
        
        hidden_size = base_model.config.d_model if hasattr(base_model, 'config') else 7168
        
        self.value_head = nn.Linear(hidden_size, 1, bias=False)
        
        if bias_vector is not None:
            self.register_buffer('bias_vector', bias_vector)
        else:
            self.register_buffer('bias_vector', torch.zeros(1))
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            logits: 语言模型logits
            rewards: 每个位置的奖励值
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_hidden=True
        )
        
        if isinstance(outputs, tuple):
            hidden_states = outputs[-1]
        else:
            hidden_states = outputs
        
        values = self.value_head(hidden_states).squeeze(-1)
        
        last_token_rewards = values[:, -1] + self.bias_vector
        
        return hidden_states, last_token_rewards


class PreferencePairDataset(Dataset):
    """
    偏好数据数据集
    每条数据包含: prompt, chosen_response, rejected_response
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        split: str = 'train'
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.examples = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """加载偏好数据"""
        examples = []
        
        if isinstance(data_path, (str, Path)) and os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    examples.append(data)
        elif isinstance(data_path, list):
            examples = data_path
        else:
            raise ValueError(f"Invalid data_path: {data_path}")
        
        logger.info(f"Loaded {len(examples)} preference examples from {data_path}")
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def _encode_with_prompt(
        self,
        prompt: str,
        response: str,
        max_length: int
    ) -> Dict[str, torch.Tensor]:
        """
        编码 prompt + response 序列
        
        格式: <bos> prompt <sep> response <eos>
        """
        sep_id = self.tokenizer.sep_token_id if hasattr(self.tokenizer, 'sep_token_id') else 151643
        bos_id = self.tokenizer.bos_token_id if hasattr(self.tokenizer, 'bos_token_id') else 151643
        eos_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else 151643
        
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
        
        full_tokens = [bos_id] + prompt_tokens + [sep_id] + response_tokens + [eos_id]
        
        if len(full_tokens) > max_length:
            full_tokens = full_tokens[:max_length]
        
        input_ids = torch.tensor(full_tokens, dtype=torch.long)
        attention_mask = torch.ones(len(full_tokens), dtype=torch.long)
        
        prompt_len = len(prompt_tokens) + 2
        labels = torch.full((len(full_tokens),), -100, dtype=torch.long)
        labels[prompt_len:] = torch.tensor(full_tokens[prompt_len:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        prompt = example.get('prompt', '')
        chosen = example.get('chosen', example.get('chosen_response', ''))
        rejected = example.get('rejected', example.get('rejected_response', ''))
        
        chosen_enc = self._encode_with_prompt(prompt, chosen, self.max_length)
        rejected_enc = self._encode_with_prompt(prompt, rejected, self.max_length)
        
        return {
            'prompt': prompt,
            'chosen_input_ids': chosen_enc['input_ids'],
            'chosen_attention_mask': chosen_enc['attention_mask'],
            'rejected_input_ids': rejected_enc['input_ids'],
            'rejected_attention_mask': rejected_enc['attention_mask']
        }


def collate_preference_batch(
    batch: List[Dict[str, Any]],
    pad_token_id: int = 0
) -> Dict[str, torch.Tensor]:
    """
    整理偏好数据批次
    
    Args:
        batch: 批次数据列表
        pad_token_id: 填充token ID
        
    Returns:
        整理后的批次字典
    """
    chosen_input_ids = torch.nn.utils.rnn.pad_sequence(
        [item['chosen_input_ids'] for item in batch],
        batch_first=True,
        padding_value=pad_token_id
    )
    
    chosen_attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item['chosen_attention_mask'] for item in batch],
        batch_first=True,
        padding_value=0
    )
    
    rejected_input_ids = torch.nn.utils.rnn.pad_sequence(
        [item['rejected_input_ids'] for item in batch],
        batch_first=True,
        padding_value=pad_token_id
    )
    
    rejected_attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item['rejected_attention_mask'] for item in batch],
        batch_first=True,
        padding_value=0
    )
    
    return {
        'chosen_input_ids': chosen_input_ids,
        'chosen_attention_mask': chosen_attention_mask,
        'rejected_input_ids': rejected_input_ids,
        'rejected_attention_mask': rejected_attention_mask
    }


class BradleyTerryLoss(nn.Module):
    """
    Bradley-Terry 损失函数
    
    给定一对偏好数据 (x, y_winner, y_loser)，损失函数定义为:
    L = -log(sigmoid(r_winner - r_loser))
    
    这等价于最大化 r_winner > r_loser 的概率。
    
    等价于二元交叉熵损失:
    L = -[y * log(sigmoid(r_winner - r_loser)) + (1-y) * log(sigmoid(r_loser - r_winner))]
    其中 y = 1 (chosen > rejected)
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
        label_smoothing: float = 0.0
    ) -> torch.Tensor:
        """
        计算 Bradley-Terry 损失
        
        Args:
            chosen_rewards: 被选中的响应的奖励分数，形状 [batch_size]
            rejected_rewards: 被拒绝的响应的奖励分数，形状 [batch_size]
            label_smoothing: 标签平滑系数
            
        Returns:
            损失值
        """
        reward_diff = chosen_rewards - rejected_rewards
        
        if label_smoothing > 0:
            loss = -F.logsigmoid(reward_diff) * (1 - label_smoothing)
            loss = loss - label_smoothing * F.logsigmoid(-reward_diff)
        else:
            loss = -F.logsigmoid(reward_diff)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class RewardModelLoss(nn.Module):
    """
    奖励模型综合损失
    
    包含:
    1. Bradley-Terry 偏好损失
    2. 奖励正则化损失（使奖励分数在合理范围内）
    3. 价值一致性损失（可选）
    """
    
    def __init__(
        self,
        weight_bt: float = 1.0,
        weight_reg: float = 0.01,
        reg_mean: float = 0.0,
        reg_std: float = 1.0,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.weight_bt = weight_bt
        self.weight_reg = weight_reg
        self.reg_mean = reg_mean
        self.reg_std = reg_std
        self.bt_loss = BradleyTerryLoss()
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
        return_metrics: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算综合损失
        
        Args:
            chosen_rewards: 被选中响应的奖励分数
            rejected_rewards: 被拒绝响应的奖励分数
            return_metrics: 是否返回详细指标
            
        Returns:
            total_loss: 总损失
            metrics: 详细指标字典
        """
        bt_loss = self.bt_loss(
            chosen_rewards,
            rejected_rewards,
            label_smoothing=self.label_smoothing
        )
        
        all_rewards = torch.cat([chosen_rewards, rejected_rewards], dim=0)
        reg_loss = F.mse_loss(
            all_rewards,
            torch.full_like(all_rewards, self.reg_mean),
            reduction='none'
        ).mean()
        
        total_loss = self.weight_bt * bt_loss + self.weight_reg * reg_loss
        
        if return_metrics:
            metrics = {
                'bt_loss': bt_loss.item(),
                'reg_loss': reg_loss.item(),
                'total_loss': total_loss.item(),
                'reward_chosen_mean': chosen_rewards.mean().item(),
                'reward_chosen_std': chosen_rewards.std().item(),
                'reward_rejected_mean': rejected_rewards.mean().item(),
                'reward_rejected_std': rejected_rewards.std().item(),
                'reward_diff_mean': (chosen_rewards - rejected_rewards).mean().item(),
                'accuracy': ((chosen_rewards > rejected_rewards).float()).mean().item()
            }
            return total_loss, metrics
        
        return total_loss


class RewardModelTrainer:
    """
    奖励模型训练器
    
    支持:
    - 混合精度训练 (AMP)
    - 梯度累积
    - 学习率调度
    - 检查点保存与恢复
    - 分布式训练
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: RewardModelConfig,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model.to(self.device)
        
        self.loss_fn = RewardModelLoss(
            label_smoothing=config.warmup_ratio * 0.1
        )
        
        self.optimizer = AdamW(
            self.model.parameters(),
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
        self.best_accuracy = 0.0
        
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
                collate_fn=lambda x: collate_preference_batch(
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
                collate_fn=lambda x: collate_preference_batch(
                    x,
                    pad_token_id=self.tokenizer.pad_token_id or 0
                )
            )
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """执行单个训练步骤"""
        self.model.train()
        
        chosen_input_ids = batch['chosen_input_ids'].to(self.device)
        chosen_attention_mask = batch['chosen_attention_mask'].to(self.device)
        rejected_input_ids = batch['rejected_input_ids'].to(self.device)
        rejected_attention_mask = batch['rejected_attention_mask'].to(self.device)
        
        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
            chosen_rewards = self.model(chosen_input_ids, chosen_attention_mask)
            rejected_rewards = self.model(rejected_input_ids, rejected_attention_mask)
            
            loss, metrics = self.loss_fn(
                chosen_rewards,
                rejected_rewards,
                return_metrics=True
            )
            
            scaled_loss = loss / self.config.gradient_accumulation_steps
        
        self.scaler.scale(scaled_loss).backward()
        
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()
        
        self.global_step += 1
        
        return metrics
    
    def _eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """执行单个评估步骤"""
        self.model.eval()
        
        with torch.no_grad():
            chosen_input_ids = batch['chosen_input_ids'].to(self.device)
            chosen_attention_mask = batch['chosen_attention_mask'].to(self.device)
            rejected_input_ids = batch['rejected_input_ids'].to(self.device)
            rejected_attention_mask = batch['rejected_attention_mask'].to(self.device)
            
            chosen_rewards = self.model(chosen_input_ids, chosen_attention_mask)
            rejected_rewards = self.model(rejected_input_ids, rejected_attention_mask)
            
            _, metrics = self.loss_fn(
                chosen_rewards,
                rejected_rewards,
                return_metrics=True
            )
        
        return metrics
    
    def _save_checkpoint(self, step: int, metrics: Dict[str, float], is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__,
            'best_accuracy': self.best_accuracy
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
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.global_step = checkpoint['step']
        self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def evaluate(self) -> Dict[str, float]:
        """在评估集上评估模型"""
        if self.eval_loader is None:
            return {}
        
        self.model.eval()
        total_metrics = {
            'total_loss': 0.0,
            'bt_loss': 0.0,
            'accuracy': 0.0,
            'reward_diff': 0.0
        }
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.eval_loader:
                metrics = self._eval_step(batch)
                total_metrics['total_loss'] += metrics['total_loss']
                total_metrics['bt_loss'] += metrics['bt_loss']
                total_metrics['accuracy'] += metrics['accuracy']
                total_metrics['reward_diff'] += metrics['reward_diff_mean']
                num_batches += 1
        
        avg_metrics = {
            'eval_loss': total_metrics['total_loss'] / num_batches,
            'eval_bt_loss': total_metrics['bt_loss'] / num_batches,
            'eval_accuracy': total_metrics['accuracy'] / num_batches,
            'eval_reward_diff': total_metrics['reward_diff'] / num_batches
        }
        
        return avg_metrics
    
    def train(self):
        """执行完整的训练流程"""
        if self.train_loader is None and self.train_dataset is not None:
            self._create_dataloaders()
        
        if self.train_loader is None:
            raise ValueError("No training dataset provided")
        
        logger.info(f"Starting reward model training for {self.config.max_steps} steps")
        logger.info(f"Device: {self.device}")
        logger.info(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        
        progress_bar = tqdm(total=self.config.max_steps, desc="Training")
        
        while self.global_step < self.config.max_steps:
            for batch in self.train_loader:
                if self.global_step >= self.config.max_steps:
                    break
                
                metrics = self._train_step(batch)
                progress_bar.update(1)
                
                if self.global_step % self.config.log_interval == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({
                        'loss': f"{metrics['total_loss']:.4f}",
                        'acc': f"{metrics['accuracy']:.4f}",
                        'lr': f"{lr:.2e}"
                    })
                
                if self.global_step % self.config.eval_interval == 0 and self.eval_loader is not None:
                    eval_metrics = self.evaluate()
                    logger.info(f"Step {self.global_step} | Eval: {eval_metrics}")
                    
                    if eval_metrics.get('eval_accuracy', 0) > self.best_accuracy:
                        self.best_accuracy = eval_metrics['eval_accuracy']
                        self._save_checkpoint(self.global_step, metrics, is_best=True)
                
                if self.global_step % self.config.save_interval == 0:
                    self._save_checkpoint(self.global_step, metrics)
        
        progress_bar.close()
        logger.info("Reward model training completed!")
        
        self._save_checkpoint(self.global_step, metrics)
        
        return self.model


class RewardModelEvaluator:
    """
    奖励模型评估器
    
    提供详细的模型评估功能，包括:
    - 准确率计算
    - AUC-ROC 计算
    - 奖励分布分析
    """
    
    def __init__(self, model: nn.Module, tokenizer, device: Optional[torch.device] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def score_single(
        self,
        prompt: str,
        response: str,
        max_length: int = 2048
    ) -> float:
        """
        对单个 (prompt, response) 对进行评分
        
        Args:
            prompt: 输入提示
            response: 响应文本
            max_length: 最大序列长度
            
        Returns:
            奖励分数
        """
        sep_id = self.tokenizer.sep_token_id if hasattr(self.tokenizer, 'sep_token_id') else 151643
        bos_id = self.tokenizer.bos_token_id if hasattr(self.tokenizer, 'bos_token_id') else 151643
        
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
        
        full_tokens = [bos_id] + prompt_tokens + [sep_id] + response_tokens
        
        if len(full_tokens) > max_length:
            full_tokens = full_tokens[:max_length]
        
        input_ids = torch.tensor([full_tokens], dtype=torch.long).to(self.device)
        attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            reward = self.model(input_ids, attention_mask)
        
        return reward.item()
    
    def score_pair(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        max_length: int = 2048
    ) -> Dict[str, float]:
        """
        对一对偏好数据进行评分
        
        Returns:
            包含两个响应的奖励分数和比较结果的字典
        """
        chosen_reward = self.score_single(prompt, chosen, max_length)
        rejected_reward = self.score_single(prompt, rejected, max_length)
        
        return {
            'chosen_reward': chosen_reward,
            'rejected_reward': rejected_reward,
            'reward_diff': chosen_reward - rejected_reward,
            'chosen_is_better': chosen_reward > rejected_reward
        }
    
    def evaluate_dataset(
        self,
        dataset: Dataset,
        batch_size: int = 8
    ) -> Dict[str, float]:
        """
        在整个数据集上评估模型
        
        Returns:
            评估指标字典
        """
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: collate_preference_batch(x, pad_token_id=0)
        )
        
        all_chosen_rewards = []
        all_rejected_rewards = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                chosen_input_ids = batch['chosen_input_ids'].to(self.device)
                chosen_attention_mask = batch['chosen_attention_mask'].to(self.device)
                rejected_input_ids = batch['rejected_input_ids'].to(self.device)
                rejected_attention_mask = batch['rejected_attention_mask'].to(self.device)
                
                chosen_rewards = self.model(chosen_input_ids, chosen_attention_mask)
                rejected_rewards = self.model(rejected_input_ids, rejected_attention_mask)
                
                all_chosen_rewards.extend(chosen_rewards.cpu().numpy())
                all_rejected_rewards.extend(rejected_rewards.cpu().numpy())
                all_labels.extend([1] * len(chosen_rewards))
        
        all_chosen_rewards = torch.tensor(all_chosen_rewards)
        all_rejected_rewards = torch.tensor(all_rejected_rewards)
        reward_diffs = all_chosen_rewards - all_rejected_rewards
        
        predictions = (reward_diffs > 0).float().numpy()
        labels = torch.ones(len(reward_diffs)).numpy()
        
        accuracy = (predictions == labels).mean()
        
        try:
            auc = roc_auc_score(labels, reward_diffs.numpy())
        except:
            auc = 0.0
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'mean_chosen_reward': all_chosen_rewards.mean().item(),
            'mean_rejected_reward': all_rejected_rewards.mean().item(),
            'mean_reward_diff': reward_diffs.mean().item(),
            'std_reward_diff': reward_diffs.std().item()
        }


def create_reward_model(
    base_model: nn.Module,
    config: Optional[RewardModelConfig] = None
) -> RewardModel:
    """
    创建奖励模型的工厂函数
    
    Args:
        base_model: 基础语言模型
        config: 奖励模型配置
        
    Returns:
        奖励模型实例
    """
    if config is None:
        config = RewardModelConfig()
    
    return RewardModel(
        base_model=base_model,
        hidden_size=config.hidden_size,
        use_mean_pooling=True
    )


def generate_synthetic_preference_data(
    num_samples: int,
    tokenizer,
    output_path: str,
    quality_range: Tuple[float, float] = (0.0, 1.0)
) -> List[Dict[str, Any]]:
    """
    生成合成的偏好数据用于测试训练流程
    
    Args:
        num_samples: 样本数量
        tokenizer: 分词器
        output_path: 输出文件路径
        quality_range: 质量分数范围
        
    Returns:
        生成的样本列表
    """
    import random
    
    templates = [
        ("What is {topic}?", [
            "The answer is complex and requires careful analysis.",
            "Let me explain {topic} in detail.",
            "{topic} is a fascinating subject with many aspects."
        ]),
        ("How does {topic} work?", [
            "{topic} operates through a series of interconnected mechanisms.",
            "The working principle of {topic} involves multiple steps.",
            "{topic} functions by integrating various components."
        ]),
        ("Why is {topic} important?", [
            "{topic} matters because it affects our daily lives.",
            "The importance of {topic} cannot be overstated.",
            "{topic} plays a crucial role in modern society."
        ])
    ]
    
    topics = ["artificial intelligence", "climate change", "quantum computing",
             "renewable energy", "space exploration", "medical research"]
    
    samples = []
    
    for i in range(num_samples):
        template, responses = random.choice(templates)
        topic = random.choice(topics)
        
        prompt = template.format(topic=topic)
        
        quality_scores = [
            random.uniform(*quality_range) for _ in responses
        ]
        
        chosen_idx = quality_scores.index(max(quality_scores))
        rejected_idx = quality_scores.index(min(quality_scores))
        
        sample = {
            'prompt': prompt,
            'chosen': responses[chosen_idx],
            'rejected': responses[rejected_idx],
            'quality_chosen': quality_scores[chosen_idx],
            'quality_rejected': quality_scores[rejected_idx]
        }
        samples.append(sample)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"Generated {num_samples} synthetic preference samples to {output_path}")
    return samples


if __name__ == '__main__':
    logger.info("Reward Model Training Module")
    logger.info("Import this module to use the RewardModelTrainer class")
