"""
IPO (Identity Preference Optimization) 训练模块

IPO 是 DPO 的改进版本，解决了 DPO 中存在的一些理论问题。
DPO 的目标是最大化 chosen 相对于 rejected 的概率比，但 IPO 发现这会导致
策略模型偏离参考模型太远，甚至反转某些样本的偏好。

IPO 损失函数:
    L = -E[(x,y_c,y_r) ~ D] [log σ(τ(log π_θ(y_c|x) - log π_θ(y_r|x) - 
                                  log π_ref(y_c|x) + log π_ref(y_r|x)) - 1/2τ)]

其中 τ 是一个温度参数。

IPO 的优势:
1. 具有更清晰的理论保证
2. 避免过度优化导致偏好反转
3. 训练更加稳定
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
import json
import os
import logging
from pathlib import Path
from tqdm import tqdm
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IPOConfig:
    """IPO 训练配置参数"""
    tau: float = 2.0
    beta: float = 0.1
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 1000
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    max_prompt_length: int = 512
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    output_dir: str = "./ipo_output"
    use_amp: bool = True
    gradient_clip_norm: float = 1.0
    label_smoothing: float = 0.0
    regularization_weight: float = 0.01
    warmup_ratio: float = 0.1


class IPODataset(Dataset):
    """
    IPO 训练数据集
    
    与 DPO 相同，每条数据包含: prompt, chosen_response, rejected_response
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
        
        logger.info(f"Loaded {len(examples)} IPO examples from {data_path}")
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


def ipo_collate_fn(
    batch: List[Dict[str, Any]],
    pad_token_id: int = 0
) -> Dict[str, torch.Tensor]:
    """
    整理 IPO 批次数据
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


class IPOLoss(nn.Module):
    """
    IPO 损失函数
    
    IPO 的核心思想是在Bradley-Terry模型下，直接优化策略模型的对数优势，
    而不是使用参考模型作为基准。
    
    IPO 损失函数:
    L = -log σ(τ * (log π_θ(y_c|x) - log π_θ(y_r|x) - 
                   log π_ref(y_c|x) + log π_ref(y_r|x)) - 1/(2τ))
    
    这可以看作是 DPO 损失的一个正则化版本，当 τ → ∞ 时，IPO 退化为 DPO。
    """
    
    def __init__(
        self,
        tau: float = 2.0,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        regularization_weight: float = 0.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.tau = tau
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.regularization_weight = regularization_weight
        self.reduction = reduction
    
    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算 IPO 损失
        
        Args:
            policy_chosen_logps: 策略模型对 chosen 响应的 log 概率
            policy_rejected_logps: 策略模型对 rejected 响应的 log 概率
            reference_chosen_logps: 参考模型对 chosen 响应的 log 概率
            reference_rejected_logps: 参考模型对 rejected 响应的 log 概率
            
        Returns:
            loss: IPO 损失值
            metrics: 训练指标字典
        """
        policy_diff = policy_chosen_logps - policy_rejected_logps
        reference_diff = reference_chosen_logps - reference_rejected_logps
        
        advantages = self.beta * (policy_diff - reference_diff)
        
        target = 1.0 / (2 * self.tau)
        logits = self.tau * advantages - target
        
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
            'ipo_loss': loss.item(),
            'logits_mean': logits.mean().item(),
            'logits_std': logits.std().item(),
            'policy_diff_mean': policy_diff.mean().item(),
            'reference_diff_mean': reference_diff.mean().item(),
            'advantage_mean': advantages.mean().item(),
            'accuracy': accuracy.item(),
            'chosen_prob_mean': chosen_probs.mean().item()
        }
        
        return loss, metrics


class SimpleIPOLoss(nn.Module):
    """
    简化版 IPO 损失
    
    不使用参考模型，直接在策略模型的对数概率上计算优势。
    """
    
    def __init__(
        self,
        tau: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.tau = tau
        self.reduction = reduction
    
    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算简化 IPO 损失"""
        policy_diff = policy_chosen_logps - policy_rejected_logps
        
        target = 1.0 / (2 * self.tau)
        logits = self.tau * policy_diff - target
        
        loss = -F.logsigmoid(logits)
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        metrics = {
            'simple_ipo_loss': loss.item(),
            'logits_mean': logits.mean().item(),
            'policy_diff_mean': policy_diff.mean().item(),
            'accuracy': (torch.sigmoid(logits) > 0.5).float().mean().item()
        }
        
        return loss, metrics


class RegularizedIPOLoss(nn.Module):
    """
    正则化 IPO 损失
    
    在 IPO 损失基础上添加 KL 散度正则项，防止策略偏离参考模型太远。
    """
    
    def __init__(
        self,
        tau: float = 2.0,
        beta: float = 0.1,
        kl_weight: float = 0.01,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.tau = tau
        self.beta = beta
        self.kl_weight = kl_weight
        self.reduction = reduction
        self.ipo_loss = IPOLoss(tau=tau, beta=beta, reduction='none')
    
    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算正则化 IPO 损失"""
        ipo_loss_val, base_metrics = self.ipo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps
        )
        
        kl_divergence = (reference_chosen_logps - policy_chosen_logps).mean() + \
                       (reference_rejected_logps - policy_rejected_logps).mean()
        
        total_loss = ipo_loss_val + self.kl_weight * kl_divergence
        
        if self.reduction == 'mean':
            total_loss = total_loss.mean()
        
        metrics = {
            'total_loss': total_loss.item(),
            'ipo_loss': base_metrics['ipo_loss'],
            'kl_divergence': kl_divergence.item()
        }
        
        return total_loss, metrics


class IPOTrainer:
    """
    IPO 训练器
    
    支持:
    - 冻结参考模型参数
    - 混合精度训练
    - 梯度累积
    - 学习率调度
    - 多种 IPO 损失变体
    """
    
    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: nn.Module,
        tokenizer,
        config: IPOConfig,
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
        
        self.loss_fn = IPOLoss(
            tau=config.tau,
            beta=config.beta,
            label_smoothing=config.label_smoothing,
            regularization_weight=config.regularization_weight
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
                collate_fn=lambda x: ipo_collate_fn(
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
                collate_fn=lambda x: ipo_collate_fn(
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
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """执行单个训练步骤"""
        self.policy_model.train()
        
        chosen_input_ids = batch['chosen_input_ids'].to(self.device)
        rejected_input_ids = batch['rejected_input_ids'].to(self.device)
        chosen_prompt_lens = batch['chosen_prompt_lens'].to(self.device)
        rejected_prompt_lens = batch['rejected_prompt_lens'].to(self.device)
        
        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
            policy_chosen_logps = self._compute_log_probabilities(
                self.policy_model, chosen_input_ids, chosen_prompt_lens
            )
            policy_rejected_logps = self._compute_log_probabilities(
                self.policy_model, rejected_input_ids, rejected_prompt_lens
            )
            
            with torch.no_grad():
                ref_chosen_logps = self._compute_log_probabilities(
                    self.reference_model, chosen_input_ids, chosen_prompt_lens
                )
                ref_rejected_logps = self._compute_log_probabilities(
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
            
            policy_chosen_logps = self._compute_log_probabilities(
                self.policy_model, chosen_input_ids, chosen_prompt_lens
            )
            policy_rejected_logps = self._compute_log_probabilities(
                self.policy_model, rejected_input_ids, rejected_prompt_lens
            )
            
            ref_chosen_logps = self._compute_log_probabilities(
                self.reference_model, chosen_input_ids, chosen_prompt_lens
            )
            ref_rejected_logps = self._compute_log_probabilities(
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
            'ipo_loss': 0.0,
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
        
        logger.info(f"Starting IPO training for {self.config.max_steps} steps")
        logger.info(f"Device: {self.device}")
        logger.info(f"Tau: {self.config.tau}, Beta: {self.config.beta}")
        
        progress_bar = tqdm(total=self.config.max_steps, desc="IPO Training")
        
        while self.global_step < self.config.max_steps:
            for batch in self.train_loader:
                if self.global_step >= self.config.max_steps:
                    break
                
                metrics = self._train_step(batch)
                progress_bar.update(1)
                
                if self.global_step % self.config.log_interval == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({
                        'loss': f"{metrics['ipo_loss']:.4f}",
                        'acc': f"{metrics['accuracy']:.4f}",
                        'lr': f"{lr:.2e}"
                    })
                
                if self.global_step % self.config.eval_interval == 0 and self.eval_loader is not None:
                    eval_metrics = self.evaluate()
                    logger.info(f"Step {self.global_step} | Eval: {eval_metrics}")
                
                if self.global_step % self.config.save_interval == 0:
                    self._save_checkpoint(self.global_step, metrics)
        
        progress_bar.close()
        logger.info("IPO training completed!")
        
        self._save_checkpoint(self.global_step, metrics)
        
        return self.policy_model


class SimpleIPOTrainer(IPOTrainer):
    """
    简化版 IPO 训练器
    
    不使用参考模型，直接在策略模型上计算损失。
    适用于没有参考模型或参考模型不可用的情况。
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = SimpleIPOLoss(tau=self.config.tau)
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """执行单个训练步骤（简化版）"""
        self.policy_model.train()
        
        chosen_input_ids = batch['chosen_input_ids'].to(self.device)
        rejected_input_ids = batch['rejected_input_ids'].to(self.device)
        chosen_prompt_lens = batch['chosen_prompt_lens'].to(self.device)
        rejected_prompt_lens = batch['rejected_prompt_lens'].to(self.device)
        
        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
            policy_chosen_logps = self._compute_log_probabilities(
                self.policy_model, chosen_input_ids, chosen_prompt_lens
            )
            policy_rejected_logps = self._compute_log_probabilities(
                self.policy_model, rejected_input_ids, rejected_prompt_lens
            )
            
            loss, metrics = self.loss_fn(
                policy_chosen_logps,
                policy_rejected_logps
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


class RegularizedIPOTrainer(IPOTrainer):
    """
    正则化 IPO 训练器
    
    使用带 KL 散度正则项的 IPO 损失。
    """
    
    def __init__(self, *args, kl_weight: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = RegularizedIPOLoss(
            tau=self.config.tau,
            beta=self.config.beta,
            kl_weight=kl_weight
        )


def create_ipo_trainer(
    policy_model: nn.Module,
    reference_model: nn.Module,
    tokenizer,
    config: Optional[IPOConfig] = None,
    trainer_type: str = 'standard'
) -> IPOTrainer:
    """
    创建 IPO 训练器的工厂函数
    
    Args:
        policy_model: 策略模型
        reference_model: 参考模型
        tokenizer: 分词器
        config: IPO 配置
        trainer_type: 训练器类型 ('standard', 'simple', 'regularized')
        
    Returns:
        IPO 训练器实例
    """
    if config is None:
        config = IPOConfig()
    
    if trainer_type == 'simple':
        return SimpleIPOTrainer(policy_model, reference_model, tokenizer, config)
    elif trainer_type == 'regularized':
        return RegularizedIPOTrainer(policy_model, reference_model, tokenizer, config)
    else:
        return IPOTrainer(policy_model, reference_model, tokenizer, config)


def generate_ipo_training_data(
    num_samples: int,
    output_path: str,
    prompts: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    生成合成的 IPO 训练数据
    
    Args:
        num_samples: 样本数量
        output_path: 输出文件路径
        prompts: 可选的 prompt 列表
        
    Returns:
        生成的样本列表
    """
    import random
    
    if prompts is None:
        prompts = [
            "Explain the concept of artificial intelligence.",
            "What are the main benefits of exercise?",
            "How does photosynthesis work?",
            "Describe the water cycle.",
            "What causes climate change?"
        ]
    
    response_templates_chosen = [
        "This is an excellent and comprehensive answer that covers all the key aspects of the topic.",
        "Let me provide a detailed explanation: the concept involves multiple interconnected components.",
        "An excellent response would include: first, a clear definition; second, practical examples; and third, supporting evidence."
    ]
    
    response_templates_rejected = [
        "This is a brief and incomplete answer.",
        "I don't have enough information to answer this question.",
        "Maybe you should look this up somewhere else."
    ]
    
    samples = []
    
    for i in range(num_samples):
        prompt = random.choice(prompts)
        
        chosen_response = random.choice(response_templates_chosen)
        rejected_response = random.choice(response_templates_rejected)
        
        sample = {
            'prompt': prompt,
            'chosen': chosen_response,
            'rejected': rejected_response
        }
        samples.append(sample)
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"Generated {num_samples} IPO training samples to {output_path}")
    return samples


if __name__ == '__main__':
    logger.info("IPO (Identity Preference Optimization) Training Module")
    logger.info("Import this module to use the IPOTrainer class")
