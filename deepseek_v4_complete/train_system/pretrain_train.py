import os
import sys
import argparse
import json
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.deepseek_v4_model import DeepSeekV4Multimodal, V4Config
from core.loss_collection import TotalLoss
from tokenizer.base_tokenizer import DeepSeekTokenizer
from data_engine.streaming_dataset import StreamingDataset
from data_engine.distributed_dataloader import DataLoaderWrapper, DataPrefetcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, args):
        self.args = args
        self.config = V4Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_step = 0
        self.epoch = 0
        
        self._init_distributed()
        self._init_model()
        self._init_optimizer()
        self._init_scheduler()
        self._init_dataloader()
        self._init_loss_fn()
        self._init_logging()
    
    def _init_distributed(self):
        if self.args.use_deepspeed:
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.world_size = int(os.environ.get('WORLD_SIZE', 1))
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend='nccl')
        elif self.args.distributed:
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.world_size = int(os.environ.get('WORLD_SIZE', 1))
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend='nccl')
        else:
            self.local_rank = 0
            self.world_size = 1
        
        self.is_main_process = self.local_rank == 0
    
    def _init_model(self):
        logger.info(f"Initializing model with config: {self.config.__dict__}")
        self.model = DeepSeekV4Multimodal(self.config).to(self.device)
        
        if self.args.load_checkpoint:
            self._load_checkpoint(self.args.load_checkpoint)
        
        if self.args.use_deepspeed:
            self.model, _, _, _ = deepspeed.initialize(
                model=self.model,
                model_parameters=self.model.parameters(),
                config=self.args.deepspeed_config
            )
        elif self.args.distributed and self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
    
    def _init_optimizer(self):
        if self.args.use_deepspeed:
            return
        
        params = self.model.parameters()
        self.optimizer = AdamW(
            params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.999)
        )
    
    def _init_scheduler(self):
        if self.args.use_deepspeed:
            return
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.max_steps,
            eta_min=self.args.min_lr
        )
        
        self.warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=self.args.warmup_factor,
            total_iters=self.args.warmup_steps
        )
    
    def _init_dataloader(self):
        transform = self._build_transform()
        dataset = StreamingDataset(
            data_paths=self.args.data_paths,
            transform=transform,
            shuffle=True,
            buffer_size=self.args.buffer_size
        )
        
        self.dataloader = DataLoaderWrapper(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True,
            distributed=self.args.distributed
        )
        
        if self.args.prefetch:
            self.dataloader = self.dataloader.to_prefetcher(self.device)
    
    def _build_transform(self):
        tokenizer = DeepSeekTokenizer(self.args.vocab_file)
        
        def transform(item):
            text = item.get('text', '')
            tokens = tokenizer.encode(text, add_special_tokens=True)
            
            if len(tokens) > self.config.max_seq_len:
                tokens = tokens[:self.config.max_seq_len]
            
            labels = tokens[:]
            labels[0] = -100
            
            return {
                'input_ids': torch.tensor(tokens, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'attention_mask': torch.ones(len(tokens), dtype=torch.long)
            }
        
        return transform
    
    def _init_loss_fn(self):
        self.loss_fn = TotalLoss(
            weight_mtp=self.args.weight_mtp,
            weight_aux=self.args.weight_aux,
            weight_cl=self.args.weight_cl
        )
    
    def _init_logging(self):
        if not self.is_main_process:
            return
        
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        
        self.log_file = os.path.join(self.args.output_dir, 'training.log')
        self.file_handler = logging.FileHandler(self.log_file)
        self.file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.file_handler.setFormatter(formatter)
        logger.addHandler(self.file_handler)
    
    def _load_checkpoint(self, path):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint)
            
            if 'optimizer' in checkpoint and not self.args.use_deepspeed:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            if 'scheduler' in checkpoint and not self.args.use_deepspeed:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            
            if 'global_step' in checkpoint:
                self.global_step = checkpoint['global_step']
            
            logger.info(f"Loaded checkpoint from {path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
    
    def _save_checkpoint(self, step):
        if not self.is_main_process:
            return
        
        checkpoint = {
            'model': self.model.state_dict() if not self.args.use_deepspeed else self.model.module.state_dict(),
            'global_step': step,
            'epoch': self.epoch
        }
        
        if not self.args.use_deepspeed:
            checkpoint['optimizer'] = self.optimizer.state_dict()
            checkpoint['scheduler'] = self.scheduler.state_dict()
        
        checkpoint_path = os.path.join(self.args.output_dir, f'checkpoint_{step}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        if step % self.args.save_interval == 0:
            latest_path = os.path.join(self.args.output_dir, 'checkpoint_latest.pt')
            torch.save(checkpoint, latest_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def _log_metrics(self, metrics, step):
        if not self.is_main_process:
            return
        
        log_str = f"Step {step} | "
        for key, value in metrics.items():
            log_str += f"{key}: {value:.6f} | "
        
        logger.info(log_str)
        
        if self.args.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            if not hasattr(self, 'writer'):
                self.writer = SummaryWriter(self.args.output_dir)
            
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)
    
    def train(self):
        logger.info(f"Starting training...")
        logger.info(f"Total steps: {self.args.max_steps}")
        logger.info(f"Batch size: {self.args.batch_size}")
        logger.info(f"Global batch size: {self.args.batch_size * self.world_size}")
        
        self.model.train()
        total_loss = 0.0
        loss_count = 0
        
        while self.global_step < self.args.max_steps:
            for batch in self.dataloader:
                if self.global_step >= self.args.max_steps:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                if self.args.use_deepspeed:
                    loss = self.model(input_ids, labels=labels)
                else:
                    logits, aux_loss = self.model(input_ids)
                    loss = self.loss_fn(logits, None, labels, aux_loss, None)
                
                if self.args.use_deepspeed:
                    self.model.backward(loss)
                    self.model.step()
                else:
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    if self.args.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    
                    self.optimizer.step()
                    
                    if self.global_step < self.args.warmup_steps:
                        self.warmup_scheduler.step()
                    else:
                        self.scheduler.step()
                
                total_loss += loss.item()
                loss_count += 1
                self.global_step += 1
                
                if self.global_step % self.args.log_interval == 0:
                    avg_loss = total_loss / loss_count
                    lr = self.optimizer.param_groups[0]['lr'] if not self.args.use_deepspeed else self.args.learning_rate
                    
                    metrics = {
                        'loss': avg_loss,
                        'lr': lr,
                        'step': self.global_step
                    }
                    
                    self._log_metrics(metrics, self.global_step)
                    total_loss = 0.0
                    loss_count = 0
                
                if self.global_step % self.args.save_interval == 0:
                    self._save_checkpoint(self.global_step)
                
                if self.args.eval_interval > 0 and self.global_step % self.args.eval_interval == 0:
                    self.evaluate()
            
            self.epoch += 1
        
        logger.info("Training completed!")
        self._save_checkpoint(self.global_step)
    
    def evaluate(self):
        logger.info(f"Evaluating at step {self.global_step}")
        self.model.eval()
        
        eval_loss = 0.0
        eval_count = 0
        
        with torch.no_grad():
            for batch in self.dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                if self.args.use_deepspeed:
                    loss = self.model(input_ids, labels=labels, training=False)
                else:
                    logits, aux_loss = self.model(input_ids)
                    loss = self.loss_fn(logits, None, labels, aux_loss, None)
                
                eval_loss += loss.item()
                eval_count += 1
                
                if eval_count >= self.args.eval_steps:
                    break
        
        avg_eval_loss = eval_loss / eval_count
        metrics = {'eval_loss': avg_eval_loss}
        self._log_metrics(metrics, self.global_step)
        
        self.model.train()

def parse_args():
    parser = argparse.ArgumentParser(description='DeepSeek V4 Pretraining')
    
    parser.add_argument('--data_paths', type=str, nargs='+', required=True, help='Path to training data files')
    parser.add_argument('--vocab_file', type=str, required=True, help='Path to vocabulary file')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--max_steps', type=int, default=1000000, help='Maximum training steps')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Gradient clipping norm')
    
    parser.add_argument('--warmup_steps', type=int, default=10000, help='Warmup steps')
    parser.add_argument('--warmup_factor', type=float, default=0.01, help='Warmup factor')
    
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    parser.add_argument('--save_interval', type=int, default=1000, help='Save interval')
    parser.add_argument('--eval_interval', type=int, default=0, help='Evaluation interval')
    parser.add_argument('--eval_steps', type=int, default=100, help='Evaluation steps')
    
    parser.add_argument('--weight_mtp', type=float, default=0.3, help='MTP loss weight')
    parser.add_argument('--weight_aux', type=float, default=0.01, help='Auxiliary loss weight')
    parser.add_argument('--weight_cl', type=float, default=0.1, help='Contrastive loss weight')
    
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data workers')
    parser.add_argument('--buffer_size', type=int, default=10000, help='Shuffle buffer size')
    parser.add_argument('--prefetch', action='store_true', help='Enable data prefetching')
    
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    parser.add_argument('--use_deepspeed', action='store_true', help='Use DeepSpeed')
    parser.add_argument('--deepspeed_config', type=str, default='./train_system/deepspeed_cfg/ds_zero3.json', help='DeepSpeed config file')
    
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard logging')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.use_deepspeed:
        deepspeed.init_distributed()
    
    trainer = Trainer(args)
    trainer.train()

if __name__ == '__main__':
    main()