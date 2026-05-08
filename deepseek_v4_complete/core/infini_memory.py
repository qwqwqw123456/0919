import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InfiniAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_memory: int = 256):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_memory = d_memory
        self.register_buffer("M_k", torch.zeros(n_heads, self.head_dim, d_memory))
        self.register_buffer("M_v", torch.zeros(n_heads, d_memory, self.head_dim))
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, kv_cache=None, cache_pos=0, use_cache=False):
        B, L, D = x.shape
        q = k = v = x.view(B, L, self.n_heads, self.head_dim)
        mem_out = torch.einsum("bhlk,hkd->bhld", q, self.M_k)
        mem_out = torch.einsum("bhld,hdc->bhlc", mem_out, self.M_v)
        mem_out = mem_out.contiguous().view(B, L, D)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q.transpose(1,2), k.transpose(1,2).transpose(-2,-1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn_out = torch.matmul(attn, v.transpose(1,2)).transpose(1,2).contiguous().view(B, L, D)

        out = attn_out + 0.1 * mem_out

        if use_cache:
            with torch.no_grad():
                new_k = k.mean(dim=1)
                new_v = v.mean(dim=1)
                for b in range(B):
                    for h in range(self.n_heads):
                        self.M_k[h] = (1 - self.alpha) * self.M_k[h] + self.alpha * torch.outer(new_k[b,h], torch.ones(self.d_memory, device=x.device))
                        self.M_v[h] = (1 - self.alpha) * self.M_v[h] + self.alpha * torch.outer(torch.ones(self.d_memory, device=x.device), new_v[b,h])

        return self.wo(out), None
