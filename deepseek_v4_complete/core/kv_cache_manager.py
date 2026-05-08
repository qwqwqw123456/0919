import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from collections import OrderedDict
import math


@dataclass
class KVCacheConfig:
    """KV缓存配置"""
    max_seq_len: int = 131072
    max_batch_size: int = 64
    num_layers: int = 64
    num_heads: int = 32
    head_dim: int = 128
    dtype: torch.dtype = torch.float16
    device: str = "cuda"
    use_paged_cache: bool = True
    page_size: int = 1024
    prealloc_size: int = 16384
    enable_cpu_offload: bool = True
    offload_threshold: float = 0.9


class PagedKVCache:
    """
    分页 KV 缓存 - 类似于 vLLM 的 PagedAttention
    将 KV 缓存分成固定大小的页面，减少内存碎片
    """
    def __init__(self, config: KVCacheConfig):
        self.config = config
        self.page_size = config.page_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        
        self.cache: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {}
        self.block_tables: Dict[int, List[int]] = {}
        self.seq_lens: Dict[int, int] = {}
        self.max_seq_len = config.max_seq_len
    
    def _allocate_block(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """分配一个新的缓存块"""
        k_cache = torch.zeros(
            self.page_size, self.num_heads, self.head_dim,
            dtype=self.config.dtype, device=self.config.device
        )
        v_cache = torch.zeros(
            self.page_size, self.num_heads, self.head_dim,
            dtype=self.config.dtype, device=self.config.device
        )
        return k_cache, v_cache
    
    def get_block(self, block_id: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """获取指定块"""
        if 0 in self.cache and block_id in self.cache[0]:
            return self.cache[0][block_id]
        return None
    
    def allocate_sequence(self, seq_id: int, max_len: int) -> List[int]:
        """为新序列分配缓存页"""
        num_blocks = math.ceil(max_len / self.page_size)
        block_ids = []
        
        for _ in range(num_blocks):
            block_id = len(self.cache[0]) if 0 in self.cache else 0
            if block_id not in self.cache.get(0, {}):
                if 0 not in self.cache:
                    self.cache[0] = {}
                self.cache[0][block_id] = self._allocate_block()
            block_ids.append(block_id)
        
        self.block_tables[seq_id] = block_ids
        self.seq_lens[seq_id] = 0
        
        return block_ids
    
    def update(self, seq_id: int, position: int, k: torch.Tensor, v: torch.Tensor):
        """更新指定位置的 KV"""
        if seq_id not in self.block_tables:
            self.allocate_sequence(seq_id, self.max_seq_len)
        
        block_id = self.block_tables[seq_id][position // self.page_size]
        offset = position % self.page_size
        
        k_cache, v_cache = self.cache[0][block_id]
        k_cache[offset] = k.squeeze(0)
        v_cache[offset] = v.squeeze(0)
        
        self.seq_lens[seq_id] = max(self.seq_lens[seq_id], position + 1)
    
    def get(self, seq_id: int, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定范围的 KV"""
        if seq_id not in self.block_tables:
            return None, None
        
        seq_len = self.seq_lens.get(seq_id, 0)
        length = min(end, seq_len) - start
        
        if length <= 0:
            return None, None
        
        k_full = []
        v_full = []
        
        start_block = start // self.page_size
        end_block = math.ceil(end / self.page_size)
        
        for block_idx in range(start_block, end_block):
            block_id = self.block_tables[seq_id][block_idx]
            k_cache, v_cache = self.cache[0][block_id]
            
            block_start = max(0, start - block_idx * self.page_size)
            block_end = min(self.page_size, end - block_idx * self.page_size)
            
            k_full.append(k_cache[block_start:block_end])
            v_full.append(v_cache[block_start:block_end])
        
        return torch.cat(k_full, dim=0), torch.cat(v_full, dim=0)
    
    def free_sequence(self, seq_id: int):
        """释放序列的缓存"""
        if seq_id in self.block_tables:
            del self.block_tables[seq_id]
        if seq_id in self.seq_lens:
            del self.seq_lens[seq_id]


class StreamingKVCache:
    """
    流式 KV 缓存 - 支持流式推理
    预先分配固定大小的缓存，支持动态扩展
    """
    def __init__(self, config: KVCacheConfig):
        self.config = config
        self.max_seq_len = config.max_seq_len
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        
        self.prealloc_len = min(config.prealloc_size, config.max_seq_len)
        
        self.k_cache = nn.Parameter(
            torch.zeros(1, self.prealloc_len, self.num_heads, self.head_dim, dtype=config.dtype),
            requires_grad=False
        )
        self.v_cache = nn.Parameter(
            torch.zeros(1, self.prealloc_len, self.num_heads, self.head_dim, dtype=config.dtype),
            requires_grad=False
        )
        
        self.current_len = 0
    
    def update(self, k: torch.Tensor, v: torch.Tensor, position: int):
        """更新缓存"""
        if self.current_len + k.size(1) > self.prealloc_len:
            self._expand_cache()
        
        end_pos = position + k.size(1)
        self.k_cache[:, position:end_pos] = k
        self.v_cache[:, position:end_pos] = v
        self.current_len = max(self.current_len, end_pos)
    
    def _expand_cache(self):
        """动态扩展缓存"""
        new_size = min(self.current_len * 2, self.max_seq_len)
        if new_size <= self.prealloc_len:
            return
        
        new_k = torch.zeros(1, new_size, self.num_heads, self.head_dim, 
                           dtype=self.k_cache.dtype, device=self.k_cache.device)
        new_v = torch.zeros(1, new_size, self.num_heads, self.head_dim,
                           dtype=self.v_cache.dtype, device=self.v_cache.device)
        
        new_k[:, :self.current_len] = self.k_cache[:, :self.current_len]
        new_v[:, :self.current_len] = self.v_cache[:, :self.current_len]
        
        self.k_cache = nn.Parameter(new_k, requires_grad=False)
        self.v_cache = nn.Parameter(new_v, requires_grad=False)
        self.prealloc_len = new_size
    
    def get(self, start: int = 0, end: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取缓存"""
        if end is None:
            end = self.current_len
        return self.k_cache[:, start:end], self.v_cache[:, start:end]
    
    def reset(self):
        """重置缓存"""
        self.current_len = 0


class QuantizedKVCache:
    """
    量化 KV 缓存 - 减少显存占用
    支持 INT8/INT4 量化
    """
    def __init__(self, config: KVCacheConfig, quant_bits: int = 8):
        self.config = config
        self.quant_bits = quant_bits
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        
        self.k_scale = torch.ones(1, config.max_seq_len, config.num_heads, 1, dtype=torch.float32)
        self.v_scale = torch.ones(1, config.max_seq_len, config.num_heads, 1, dtype=torch.float32)
        
        self.k_cache = torch.zeros(1, config.max_seq_len, config.num_heads, self.head_dim, 
                                   dtype=torch.int8 if quant_bits == 8 else torch.int4)
        self.v_cache = torch.zeros(1, config.max_seq_len, config.num_heads, self.head_dim,
                                   dtype=torch.int8 if quant_bits == 8 else torch.int4)
    
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """量化"""
        scale = x.abs().max() / (2 ** (self.quant_bits - 1) - 1)
        x_quant = (x / scale).round().clamp(-2**(self.quant_bits-1), 2**(self.quant_bits-1)-1)
        return x_quant.to(torch.int8 if self.quant_bits == 8 else torch.int4), scale
    
    def dequantize(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """反量化"""
        return x.float() * scale
    
    def update(self, k: torch.Tensor, v: torch.Tensor, position: int):
        """更新量化缓存"""
        k_quant, k_scale = self.quantize(k)
        v_quant, v_scale = self.quantize(v)
        
        end_pos = position + k.size(1)
        self.k_cache[:, position:end_pos] = k_quant.squeeze(0)
        self.v_cache[:, position:end_pos] = v_quant.squeeze(0)
        self.k_scale[:, position:end_pos] = k_scale.squeeze(0)
        self.v_scale[:, position:end_pos] = v_scale.squeeze(0)
    
    def get(self, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取并反量化"""
        k = self.dequantize(self.k_cache[:, start:end].float(), self.k_scale[:, start:end])
        v = self.dequantize(self.v_cache[:, start:end].float(), self.v_scale[:, start:end])
        return k.unsqueeze(0), v.unsqueeze(0)


class KVCacheManager:
    """
    KV 缓存管理器 - 统一接口
    支持多种缓存策略：分页、流式、量化
    """
    def __init__(self, config: Optional[KVCacheConfig] = None):
        if config is None:
            config = KVCacheConfig()
        
        self.config = config
        self.n_layers = config.num_layers
        
        if config.use_paged_cache:
            self.cache = [PagedKVCache(config) for _ in range(config.num_layers)]
        else:
            self.cache = [StreamingKVCache(config) for _ in range(config.num_layers)]
        
        self.quant_cache = QuantizedKVCache(config) if config.dtype == torch.int8 else None
        self.use_quant = config.dtype == torch.int8
    
    def update(self, layer_idx: int, position: int, k: torch.Tensor, v: torch.Tensor):
        """更新指定层的缓存"""
        if 0 <= layer_idx < self.n_layers:
            self.cache[layer_idx].update(k, v, position)
    
    def get(self, layer_idx: int, start: int = 0, end: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定层的缓存"""
        if 0 <= layer_idx < self.n_layers:
            return self.cache[layer_idx].get(start, end)
        return None, None
    
    def get_all_layers(self, start: int = 0, end: Optional[int] = None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """获取所有层的缓存"""
        return [self.get(i, start, end) for i in range(self.n_layers)]
    
    def reset(self):
        """重置所有缓存"""
        for cache in self.cache:
            if hasattr(cache, 'reset'):
                cache.reset()
    
    def allocate_sequence(self, seq_id: int, max_len: int):
        """为新序列分配缓存"""
        for cache in self.cache:
            if hasattr(cache, 'allocate_sequence'):
                cache.allocate_sequence(seq_id, max_len)
    
    def free_sequence(self, seq_id: int):
        """释放序列缓存"""
        for cache in self.cache:
            if hasattr(cache, 'free_sequence'):
                cache.free_sequence(seq_id)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        total_params = 0
        for cache in self.cache:
            if hasattr(cache, 'k_cache') and hasattr(cache.k_cache, 'numel'):
                total_params += cache.k_cache.numel() + cache.v_cache.numel()
        
        bytes_per_param = 2 if self.config.dtype == torch.float16 else 1
        memory_mb = total_params * bytes_per_param / (1024 ** 2)
        
        return {
            "total_params": total_params,
            "memory_mb": memory_mb,
            "max_seq_len": self.config.max_seq_len,
            "num_layers": self.n_layers
        }


def create_kv_cache(cache_type: str = "paged", **kwargs) -> KVCacheManager:
    """
    创建 KV 缓存管理器的工厂函数
    
    Args:
        cache_type: 缓存类型 ("paged", "streaming", "quantized")
    """
    config = KVCacheConfig(**kwargs)
    
    if cache_type == "paged":
        return KVCacheManager(config)
    elif cache_type == "streaming":
        config.use_paged_cache = False
        return KVCacheManager(config)
    elif cache_type == "quantized":
        config.dtype = torch.int8
        return KVCacheManager(config)
    else:
        return KVCacheManager(config)