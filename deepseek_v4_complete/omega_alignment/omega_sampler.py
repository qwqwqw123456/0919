import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math

class OmegaSampler(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 131072):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len
        
        self.omega_proj = nn.Linear(d_model, n_heads * self.head_dim)
        self.alpha_proj = nn.Linear(d_model, n_heads)
        self.beta_proj = nn.Linear(d_model, n_heads)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor, 
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, L, D = x.shape
        
        omega = self.omega_proj(x).view(B, L, self.n_heads, self.head_dim)
        alpha = self.alpha_proj(x).view(B, L, self.n_heads, 1)
        beta = self.beta_proj(x).view(B, L, self.n_heads, 1)
        
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, omega], dim=1)
            v = torch.cat([past_v, omega], dim=1)
        else:
            k = v = omega
        
        if use_cache:
            new_kv = (k, v)
        else:
            new_kv = None
        
        scores = torch.matmul(omega, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        dynamic_mask = self._compute_dynamic_mask(alpha, beta, L, k.size(1))
        scores = scores + dynamic_mask
        
        attn = self.softmax(scores)
        output = torch.matmul(attn, v)
        
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        return output, new_kv
    
    def _compute_dynamic_mask(self, alpha: torch.Tensor, beta: torch.Tensor, 
                              query_len: int, key_len: int) -> torch.Tensor:
        device = alpha.device
        query_pos = torch.arange(query_len, device=device).view(1, query_len, 1, 1)
        key_pos = torch.arange(key_len, device=device).view(1, 1, 1, key_len)
        
        distance = torch.abs(query_pos - key_pos)
        alpha = alpha.expand(-1, -1, -1, key_len)
        beta = beta.expand(-1, -1, -1, key_len)
        
        mask = -alpha * distance - beta
        return mask

class OmegaAligner(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 131072):
        super().__init__()
        self.sampler = OmegaSampler(d_model, n_heads, max_seq_len)
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, 
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = x
        x = self.norm(x)
        sampled, new_kv = self.sampler(x, past_kv, use_cache)
        output = residual + self.output_proj(sampled)
        return output, new_kv

class AdaptiveOmegaLoss(nn.Module):
    def __init__(self, alpha_weight: float = 0.1, beta_weight: float = 0.1):
        super().__init__()
        self.alpha_weight = alpha_weight
        self.beta_weight = beta_weight
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, 
                alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        
        alpha_reg = alpha.abs().mean()
        beta_reg = beta.abs().mean()
        
        total_loss = ce_loss + self.alpha_weight * alpha_reg + self.beta_weight * beta_reg
        return total_loss