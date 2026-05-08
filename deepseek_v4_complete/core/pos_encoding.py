import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class RoPEConfig:
    """RoPE 配置"""
    head_dim: int = 128
    max_seq_len: int = 131072
    base: float = 10000.0
    factor: float = 1.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    use_ntk_scaling: bool = True


class StreamingRoPE(nn.Module):
    """
    流式 RoPE - 动态计算位置编码，不预存储
    支持超长序列（131072+）而不占用大量显存
    """
    def __init__(self, config: RoPEConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim
        self.base = config.base
        self.factor = config.factor
        self.beta_fast = config.beta_fast
        self.beta_slow = config.beta_slow
        
        self._compute_base_freqs()
    
    def _compute_base_freqs(self):
        """计算基础频率"""
        self.register_buffer("base_freqs", None, persistent=False)
    
    def _compute_freqs_cis(self, seq_len: int, start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        动态计算频率，不存储完整矩阵
        """
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cpu')
        
        positions = torch.arange(start_pos, start_pos + seq_len, device=device, dtype=torch.float32)
        
        freqs = self.base ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim)
        
        freqs = torch.outer(positions, freqs)
        freqs = torch.cat([freqs, freqs], dim=-1)
        
        cos = freqs.cos()
        sin = freqs.sin()
        
        return cos, sin
    
    def _ntk_aware_scaling(self, seq_len: int) -> float:
        """
        NTK-aware 缩放 - 改善长上下文外推能力
        """
        if seq_len <= self.config.max_seq_len:
            return 1.0
        
        ctx_len = seq_len / self.config.max_seq_len
        assert ctx_len > 1, "ctx_len should be greater than 1"
        
        base = self.config.base
        beta_fast = self.beta_fast
        beta_slow = self.beta_slow
        
        def _ntk_scaled_degree(alpha: float, seq_len: int) -> float:
            return (alpha * (seq_len - self.config.max_seq_len) / self.config.max_seq_len) + 1
        
        if ctx_len >= self.beta_fast:
            scale = self.factor * _ntk_scaled_degree(beta_fast, ctx_len)
        elif ctx_len >= self.beta_slow:
            scale = self.factor * _ntk_scaled_degree(beta_slow, ctx_len)
        else:
            scale = self.factor
        
        return scale
    
    def forward(self, seq_len: int, start_pos: int = 0, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回指定范围的 cos 和 sin
        
        Args:
            seq_len: 需要的长度
            start_pos: 起始位置
            device: 目标设备
        """
        scale = self._ntk_aware_scaling(start_pos + seq_len)
        
        if scale != 1.0:
            adjusted_base = self.base * scale
        else:
            adjusted_base = self.base
        
        positions = torch.arange(start_pos, start_pos + seq_len, device=device or torch.device('cpu'), dtype=torch.float32)
        
        freqs = adjusted_base ** (torch.arange(0, self.head_dim, 2, device=device or torch.device('cpu'), dtype=torch.float32) / self.head_dim)
        freqs = torch.outer(positions, freqs)
        freqs = torch.cat([freqs, freqs], dim=-1)
        
        cos = freqs.cos()
        sin = freqs.sin()
        
        return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)
    
    @staticmethod
    def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用 RoTA 旋转到 Q 和 K
        
        Args:
            q: [batch, heads, seq, head_dim]
            k: [batch, heads, seq, head_dim]
            cos: [1, 1, seq, head_dim]
            sin: [1, 1, seq, head_dim]
        """
        q_embed = q * cos + torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape_as(q) * sin
        k_embed = k * cos + torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape_as(k) * sin
        return q_embed, k_embed


class YaRNRoPE(nn.Module):
    """
    YaRN (Yet another RoPE extensioN) - 更好的长上下文外推
    论文: https://arxiv.org/abs/2309.00071
    """
    def __init__(self, config: RoPEConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim
        self.base = config.base
        self.factor = config.factor
        self.scale_factor = config.factor
        
        self.register_buffer("dim_mask", torch.arange(0, config.head_dim, 2), persistent=False)
    
    def forward(self, seq_len: int, start_pos: int = 0, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """YaRN 缩放的 RoPE"""
        device = device or next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cpu')
        
        positions = torch.arange(start_pos, start_pos + seq_len, device=device, dtype=torch.float32)
        
        dims = torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32)
        scale = (dims / self.head_dim) * (self.scale_factor - 1) + 1
        freqs = self.base ** (dims / self.head_dim)
        freqs = freqs * (math.pi * (positions + 0.5 * seq_len) / seq_len)
        
        freqs = torch.outer(positions, freqs) / scale
        freqs = torch.cat([freqs, freqs], dim=-1)
        
        return freqs.cos().unsqueeze(0).unsqueeze(0), freqs.sin().unsqueeze(0).unsqueeze(0)


class LinearRoPE(nn.Module):
    """
    线性 RoPE - 理论上支持无限长度
    使用线性变换代替三角函数
    """
    def __init__(self, head_dim: int, max_seq_len: int = 131072):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        
        self.rope_cache = nn.Linear(head_dim, head_dim, bias=False)
    
    def forward(self, seq_len: int, start_pos: int = 0, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回线性变换后的位置编码"""
        device = device or next(self.parameters()).device
        
        positions = torch.arange(start_pos, start_pos + seq_len, device=device)
        
        pos_emb = self.rope_cache.weight[:seq_len] if seq_len <= self.max_seq_len else self.rope_cache.weight
        
        return pos_emb.unsqueeze(0), None
    
    @staticmethod
    def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用线性 RoPE"""
        if sin is not None:
            return q + sin * cos, k
        return q, k


def create_rope(rope_type: str = "streaming", **kwargs) -> nn.Module:
    """
    创建 RoPE 模块的工厂函数
    
    Args:
        rope_type: RoPE 类型 ("streaming", "yarn", "linear", "base")
    
    Returns:
        RoPE 模块
    """
    config = RoPEConfig(**kwargs)
    
    if rope_type == "streaming":
        return StreamingRoPE(config)
    elif rope_type == "yarn":
        return YaRNRoPE(config)
    elif rope_type == "linear":
        return LinearRoPE(config.head_dim, config.max_seq_len)
    else:
        config.use_ntk_scaling = False
        return StreamingRoPE(config)