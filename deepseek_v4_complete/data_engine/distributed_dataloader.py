import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from typing import List, Dict, Any, Optional, Callable
import math

class DistributedDataLoader:
    def __init__(self, dataset: Dataset, 
                 batch_size: int = 8,
                 shuffle: bool = True,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 drop_last: bool = True,
                 collate_fn: Optional[Callable] = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        self.sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
            drop_last=drop_last
        )
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=self.sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate_fn
        )
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)
    
    def set_epoch(self, epoch: int):
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
    
    def get_global_batch_size(self) -> int:
        return self.batch_size * self.world_size
    
    def get_local_batch_size(self) -> int:
        return self.batch_size

class DataPrefetcher:
    def __init__(self, loader, device: torch.device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == 'cuda' else None
        self.preload()
    
    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self.next_data = self._to_device(self.next_data)
    
    def _to_device(self, data):
        if isinstance(data, dict):
            return {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        elif isinstance(data, list):
            return [self._to_device(item) for item in data]
        elif isinstance(data, torch.Tensor):
            return data.to(self.device, non_blocking=True)
        return data
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        
        data = self.next_data
        if data is None:
            raise StopIteration
        
        self.preload()
        return data

class BatchSampler:
    def __init__(self, dataset: Dataset, 
                 batch_size: int = 8,
                 shuffle: bool = True,
                 seed: int = 42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        self.num_samples = len(dataset)
        self.num_batches = math.ceil(self.num_samples / batch_size)
    
    def set_epoch(self, epoch: int):
        self.epoch = epoch
    
    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.num_samples, generator=g).tolist()
        else:
            indices = list(range(self.num_samples))
        
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.num_samples)
            yield indices[start:end]
    
    def __len__(self):
        return self.num_batches

class DataLoaderWrapper:
    def __init__(self, dataset: Dataset,
                 batch_size: int = 8,
                 shuffle: bool = True,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 drop_last: bool = True,
                 collate_fn: Optional[Callable] = None,
                 distributed: bool = False):
        self.distributed = distributed
        
        if distributed and dist.is_initialized():
            self.loader = DistributedDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
                collate_fn=collate_fn
            )
        else:
            self.loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
                collate_fn=collate_fn
            )
    
    def __iter__(self):
        return iter(self.loader)
    
    def __len__(self):
        return len(self.loader)
    
    def set_epoch(self, epoch: int):
        if hasattr(self.loader, 'set_epoch'):
            self.loader.set_epoch(epoch)
    
    def to_prefetcher(self, device: torch.device):
        return DataPrefetcher(self.loader, device)