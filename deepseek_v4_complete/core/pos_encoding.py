import torch
import torch.nn as nn
import math

class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 131072):
        super().__init__()
        self.head_dim = head_dim
        freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, freqs)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, start: int, end: int, device: torch.device):
        cos = self.cos[start:end].to(device).unsqueeze(0).unsqueeze(0)
        sin = self.sin[start:end].to(device).unsqueeze(0).unsqueeze(0)
        return cos, sin

    @staticmethod
    def apply(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        q_rot = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape_as(q)
        k_rot = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape_as(k)
        q_out = q * cos + q_rot * sin
        k_out = k * cos + k_rot * sin
        return q_out, k_out
