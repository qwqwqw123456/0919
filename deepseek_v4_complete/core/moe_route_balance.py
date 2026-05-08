import torch.nn as nn
from .moe_architecture import DeepSeekMoE

class MoEWithBalance(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_shared: int = 2, n_routed: int = 384, top_k: int = 6, capacity_factor: float = 1.25):
        super().__init__()
        self.moe = DeepSeekMoE(d_model, d_ff, n_shared, n_routed, top_k)
        self.capacity_factor = capacity_factor

    def forward(self, x):
        return self.moe(x)
