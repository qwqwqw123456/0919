import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Optional, Tuple, Any
import json
import os
from dataclasses import dataclass
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 10000
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    log_interval: int = 10
    save_interval: int = 1000
    output_dir: str = "./sft_output"


class SFTDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        examples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                examples.append(data)
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        messages = example.get('messages', [])
        
        text = self._format_conversation(messages)
        encoding = self.tokenizer.encode(text)
        
        if len(encoding) > self.max_length:
            encoding = encoding[:self.max_length]
        
        input_ids = encoding[:-1]
        labels = encoding[1:]
        
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.vocab['<pad>']] * padding_length
            labels = labels + [-100] * padding_length
        else:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.ones(len(input_ids), dtype=torch.long)
        }
    
    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        text = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                text += f"<system>{content}</system>"
            elif role == 'user':
                text += f"<user>{content}</user>"
            elif role == 'assistant':
                text += f"<assistant>{content}</assistant>"
        
        return text


class SFTLoss(nn.Module):
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.ignore_index
        )
        return loss


class SFTTrainer:
    def __init__(self, model: nn.Module, tokenizer, config: SFTConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        self.loss_fn = SFTLoss()
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_steps,
            eta_min=config.learning_rate * 0.1
        )
        
        self.global_step = 0
        self.best_loss = float('inf')
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        input_ids = batch['input_ids']
        labels = batch['labels']
        
        outputs = self.model(input_ids)
        
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        
        loss = self.loss_fn(logits, labels)
        
        metrics = {
            'loss': loss.item(),
            'ppl': torch.exp(loss).item()
        }
        
        return loss, metrics
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        
        loss, metrics = self._compute_loss(batch)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1
        
        return metrics
    
    def _save_checkpoint(self, step: int, metrics: Dict[str, float]):
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__
        }
        
        checkpoint_path = os.path.join(self.config.output_dir, f'checkpoint_{step}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        latest_path = os.path.join(self.config.output_dir, 'checkpoint_latest.pt')
        torch.save(checkpoint, latest_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"Starting SFT training for {self.config.max_steps} steps")
        
        while self.global_step < self.config.max_steps:
            for batch in train_loader:
                if self.global_step >= self.config.max_steps:
                    break
                
                metrics = self._train_step(batch)
                
                if self.global_step % self.config.log_interval == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    logger.info(
                        f"Step {self.global_step}/{self.config.max_steps} | "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"PPL: {metrics['ppl']:.4f} | "
                        f"LR: {lr:.2e}"
                    )
                
                if self.global_step % self.config.save_interval == 0:
                    self._save_checkpoint(self.global_step, metrics)
        
        logger.info("SFT training completed!")
        self._save_checkpoint(self.global_step, metrics)
        
        if eval_dataset is not None:
            eval_metrics = self.evaluate(eval_dataset)
            logger.info(f"Eval metrics: {eval_metrics}")
        
        return self.model
    
    def evaluate(self, eval_dataset: Dataset) -> Dict[str, float]:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4
        )
        
        self.model.eval()
        total_loss = 0.0
        total_ppl = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                loss, metrics = self._compute_loss(batch)
                total_loss += metrics['loss']
                total_ppl += metrics['ppl']
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_ppl = total_ppl / num_batches
        
        return {
            'eval_loss': avg_loss,
            'eval_ppl': avg_ppl
        }


def create_sft_trainer(model, tokenizer, config: Optional[SFTConfig] = None) -> SFTTrainer:
    if config is None:
        config = SFTConfig()
    return SFTTrainer(model, tokenizer, config)


def format_conversation_for_sft(messages: List[Dict[str, str]]) -> str:
    text = ""
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        
        if role == 'system':
            text += f"<system>{content}</system>"
        elif role == 'user':
            text += f"<user>{content}</user>"
        elif role == 'assistant':
            text += f"<assistant>{content}</assistant>"
    
    return text


def prepare_sft_data(input_file: str, output_file: str, tokenizer):
    examples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            messages = data.get('messages', [])
            
            formatted = format_conversation_for_sft(messages)
            tokens = tokenizer.encode(formatted)
            
            examples.append({
                'messages': messages,
                'text': formatted,
                'token_count': len(tokens)
            })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    logger.info(f"Prepared {len(examples)} SFT examples to {output_file}")