import torch
from typing import Optional, Tuple, List

class KVCacheManager:
    def __init__(self, n_layers: int, batch_size: int = 1):
        self.n_layers = n_layers
        self.caches: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * n_layers

    def update(self, layer_idx: int, new_kv: Tuple[torch.Tensor, torch.Tensor]):
        self.caches[layer_idx] = new_kv

    def get(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return self.caches[layer_idx]

    def reset(self):
        self.caches = [None] * self.n_layers
