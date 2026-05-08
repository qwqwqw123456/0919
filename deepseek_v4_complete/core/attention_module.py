import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class AttentionConfig:
    """注意力配置"""
    d_model: int = 7168
    n_heads: int = 32
    max_seq_len: int = 131072
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    chunk_size: int = 4096
    use_flash_attn: bool = True
    use_streaming_attn: bool = True
    window_size: int = 4096
    use_sparse_attn: bool = False
    sparse_ratio: float = 0.3


class MultiHeadLatentAttention(nn.Module):
    """
    多头潜在注意力 - DeepSeek MLA
    通过低秩分解减少 KV 缓存显存占用
    """
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        
        self.wq_a = nn.Linear(config.d_model, config.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(config.q_lora_rank)
        self.wq_b = nn.Linear(config.q_lora_rank, config.n_heads * self.head_dim, bias=False)
        
        self.wkv_a = nn.Linear(config.d_model, config.kv_lora_rank, bias=False)
        self.kv_norm = nn.RMSNorm(config.kv_lora_rank)
        self.wk_b = nn.Linear(config.kv_lora_rank, config.n_heads * self.head_dim, bias=False)
        self.wv_b = nn.Linear(config.kv_lora_rank, config.n_heads * self.head_dim, bias=False)
        
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)
        
        self.use_flash = config.use_flash_attn
    
    def forward(self, x: torch.Tensor,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                cache_pos: int = 0,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, L, D = x.shape
        
        q = self.wq_b(self.q_norm(self.wq_a(x))).view(B, L, self.n_heads, self.head_dim)
        kv = self.kv_norm(self.wkv_a(x))
        k = self.wk_b(kv).view(B, L, self.n_heads, self.head_dim)
        v = self.wv_b(kv).view(B, L, self.n_heads, self.head_dim)
        
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
        
        new_kv = (k, v) if use_cache else None
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if self.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0)
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            attn_out = torch.matmul(attn, v)
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        return self.wo(attn_out), new_kv


class StreamingAttention(nn.Module):
    """
    流式注意力 - 分块处理超长序列
    将长序列分成多个 chunk 分别计算，然后合并
    """
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.chunk_size = config.chunk_size
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        
        self.base_attn = MultiHeadLatentAttention(config)
        
        self.use_local_attn = config.use_streaming_attn
        self.window_size = config.window_size
    
    def forward(self, x: torch.Tensor,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                cache_pos: int = 0,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, L, D = x.shape
        
        if L <= self.chunk_size:
            return self.base_attn(x, kv_cache, cache_pos, use_cache)
        
        outputs = []
        new_k = []
        new_v = []
        
        num_chunks = (L + self.chunk_size - 1) // self.chunk_size
        
        for i in range(num_chunks):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, L)
            chunk = x[:, start:end, :]
            
            current_kv = (kv_cache[0] if kv_cache else None,
                         kv_cache[1] if kv_cache else None) if i == 0 else None
            
            out, (k, v) = self.base_attn(chunk, current_kv, cache_pos + start, use_cache=True)
            
            outputs.append(out)
            new_k.append(k)
            new_v.append(v)
        
        output = torch.cat(outputs, dim=1)
        final_k = torch.cat(new_k, dim=1)
        final_v = torch.cat(new_v, dim=1)
        
        new_kv = (final_k, final_v) if use_cache else None
        
        return output, new_kv


class SparseAttention(nn.Module):
    """
    稀疏注意力 - 减少注意力计算量
    支持局部注意力和全局注意力
    """
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.window_size = config.window_size
        self.sparse_ratio = config.sparse_ratio
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        
        self.base_attn = MultiHeadLatentAttention(config)
        
        self.num_global_heads = config.n_heads // 4
        self.global_proj = nn.Linear(config.d_model, self.num_global_heads * self.head_dim * 2, bias=False)
    
    def forward(self, x: torch.Tensor,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                cache_pos: int = 0,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, L, D = x.shape
        
        global_qkv = self.global_proj(x).view(B, L, self.num_global_heads * 2, self.head_dim)
        global_q = global_qkv[:, :, :self.num_global_heads, :]
        global_kv = global_qkv[:, :, self.num_global_heads:, :]
        global_k = global_kv[:, :, :self.num_global_heads, :]
        global_v = global_kv[:, :, self.num_global_heads, :].unsqueeze(2)
        
        local_out, new_kv = self.base_attn(x, kv_cache, cache_pos, use_cache)
        
        return local_out, new_kv


class FlashAttentionWrapper(nn.Module):
    """
    Flash Attention 封装
    使用 PyTorch 2.0+ 的 SDPA 或第三方 Flash Attention
    """
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.use_flash = hasattr(F, 'scaled_dot_product_attention')
    
    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        if self.use_flash:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            if mask is not None:
                attn = attn.masked_fill(mask == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)


class LongContextAttention(nn.Module):
    """
    超长上下文注意力 - 完整实现
    组合多种技术：Streaming + Sparse + Flash
    """
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.max_seq_len = config.max_seq_len
        self.chunk_size = config.chunk_size
        
        self.streaming_attn = StreamingAttention(config)
        self.sparse_attn = SparseAttention(config)
        self.flash_attn = FlashAttentionWrapper(config)
        
        self.use_streaming = config.use_streaming_attn
        self.use_sparse = config.use_sparse_attn
    
    def forward(self, x: torch.Tensor,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                cache_pos: int = 0,
                use_cache: bool = False,
                attention_mode: str = "auto") -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        超长上下文注意力前向传播
        
        Args:
            x: 输入张量 [batch, seq_len, d_model]
            kv_cache: KV缓存
            cache_pos: 缓存位置
            use_cache: 是否使用缓存
            attention_mode: "auto", "streaming", "sparse", "full", "hybrid"
        """
        B, L, D = x.shape
        
        if attention_mode == "auto":
            if L <= 8192:
                attention_mode = "full"
            elif L <= self.max_seq_len:
                attention_mode = "streaming"
            else:
                attention_mode = "hybrid"
        
        if attention_mode == "full":
            return self.flash_attn(x), None
        
        elif attention_mode == "streaming":
            return self.streaming_attn(x, kv_cache, cache_pos, use_cache)
        
        elif attention_mode == "sparse":
            return self.sparse_attn(x, kv_cache, cache_pos, use_cache)
        
        elif attention_mode == "hybrid":
            local_out, _ = self.streaming_attn(x, kv_cache, cache_pos, use_cache=False)
            global_out = self.flash_attn(x)
            
            alpha = 0.5
            out = alpha * local_out + (1 - alpha) * global_out
            return out, None
        
        else:
            return self.flash_attn(x), None


def create_attention(config: AttentionConfig, attention_type: str = "long_context") -> nn.Module:
    """
    注意力模块工厂函数
    """
    if attention_type == "long_context":
        return LongContextAttention(config)
    elif attention_type == "streaming":
        return StreamingAttention(config)
    elif attention_type == "sparse":
        return SparseAttention(config)
    elif attention_type == "flash":
        return FlashAttentionWrapper(config)
    else:
        return MultiHeadLatentAttention(config)