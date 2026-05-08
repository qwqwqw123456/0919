import torch
import torch.nn as nn
from .attention_module import MultiHeadLatentAttention

try:
    from mamba_ssm import Mamba2 as Mamba2Block
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False

class HybridBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_state: int = 128, expand: int = 2):
        super().__init__()
        self.attn = MultiHeadLatentAttention(d_model, n_heads)
        if MAMBA_AVAILABLE:
            self.mamba = Mamba2Block(d_model, d_state=d_state, expand=expand)
        else:
            self.mamba = None
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, kv_cache=None, cache_pos=0, use_cache=False):
        attn_out, new_kv = self.attn(x, kv_cache, cache_pos, use_cache)
        if self.mamba is not None:
            mamba_out = self.mamba(x)
            gate = torch.sigmoid(self.gate)
            out = gate * attn_out + (1 - gate) * mamba_out
        else:
            out = attn_out
        return out, new_kv, None
