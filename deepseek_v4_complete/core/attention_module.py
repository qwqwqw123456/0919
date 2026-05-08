import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from .pos_encoding import RoPE
from .layer_norm import RMSNorm

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 131072,
                 q_lora_rank: int = 1536, kv_lora_rank: int = 512):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.wq_a = nn.Linear(d_model, q_lora_rank, bias=False)
        self.q_norm = RMSNorm(q_lora_rank)
        self.wq_b = nn.Linear(q_lora_rank, n_heads * self.head_dim, bias=False)

        self.wkv_a = nn.Linear(d_model, kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(kv_lora_rank)
        self.wk_b = nn.Linear(kv_lora_rank, n_heads * self.head_dim, bias=False)
        self.wv_b = nn.Linear(kv_lora_rank, n_heads * self.head_dim, bias=False)

        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.rope = RoPE(self.head_dim, max_seq_len)

    def forward(self, x: torch.Tensor,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                cache_pos: int = 0,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, L, D = x.shape
        q = self.wq_b(self.q_norm(self.wq_a(x))).view(B, L, self.n_heads, self.head_dim)
        kv = self.kv_norm(self.wkv_a(x))
        k = self.wk_b(kv).view(B, L, self.n_heads, self.head_dim)
        v = self.wv_b(kv).view(B, L, self.n_heads, self.head_dim)

        cos, sin = self.rope(cache_pos, cache_pos + L, x.device)
        q, k = RoPE.apply(q, k, cos, sin)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)

        new_kv = (k, v) if use_cache else None
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q.transpose(1, 2), k.transpose(1, 2).transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v.transpose(1, 2)).transpose(1, 2).contiguous().view(B, L, D)
        return self.wo(out), new_kv
