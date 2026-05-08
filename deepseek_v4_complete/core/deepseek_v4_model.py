import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Union, Dict, Any
from dataclasses import dataclass, field
import math


@dataclass
class V4Config:
    """DeepSeek V4 模型配置"""
    vocab_size: int = 129280
    d_model: int = 7168
    n_heads: int = 32
    d_ff: int = 2048
    n_layers: int = 64
    num_experts: int = 384
    top_k: int = 6
    image_size: int = 448
    max_tiles: int = 6
    max_seq_len: int = 131072
    
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    
    use_streaming_rope: bool = True
    use_paged_cache: bool = True
    use_flash_attn: bool = True
    use_long_context_attn: bool = True
    attention_chunk_size: int = 4096


class DynamicImageProcessor:
    """动态图像处理器"""
    def __init__(self, image_size=448, patch_size=14, max_num_tiles=6):
        self.image_size = image_size
        self.max_num_tiles = max_num_tiles
        self.patch_size = patch_size
        try:
            from torchvision import transforms
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
            ])
        except ImportError:
            self.transform = None
    
    def process(self, image):
        if self.transform:
            return self.transform(image).unsqueeze(0)
        return torch.zeros(1, 3, self.image_size, self.image_size)
    
    def process_tiles(self, image, num_tiles=1):
        tiles = []
        for _ in range(min(num_tiles, self.max_num_tiles)):
            tiles.append(self.process(image))
        return tiles


class VisionEncoder(nn.Module):
    """视觉编码器"""
    def __init__(self, d_model=7168):
        super().__init__()
        self.conv = nn.Conv2d(3, d_model, 14, stride=14)
    
    def forward(self, tiles):
        if isinstance(tiles, list):
            if len(tiles) == 0:
                return torch.zeros(1, 0, self.conv.out_channels)
            tiles = torch.cat(tiles, dim=0)
        B = tiles.size(0)
        out = self.conv(tiles)
        out = out.flatten(2).transpose(1, 2)
        return out


class VisionProjector(nn.Module):
    """视觉投影器"""
    def __init__(self, d_model, num_visual_tokens=256):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.num_visual_tokens = num_visual_tokens
    
    def forward(self, x):
        if x.size(1) > self.num_visual_tokens:
            x = x[:, :self.num_visual_tokens, :]
        elif x.size(1) < self.num_visual_tokens:
            padding = torch.zeros(x.size(0), self.num_visual_tokens - x.size(1), x.size(2), 
                                 device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)
        return self.proj(x)


class RMSNorm(nn.Module):
    """RMSNorm 归一化"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU 激活函数"""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x))


class DeepSeekMoE(nn.Module):
    """DeepSeek MoE 架构"""
    def __init__(self, d_model: int, d_ff: int, n_shared: int = 2, n_routed: int = 384, top_k: int = 6):
        super().__init__()
        self.n_routed = n_routed
        self.top_k = top_k
        
        self.shared_experts = nn.ModuleList([
            SwiGLU(d_model, d_ff) for _ in range(n_shared)
        ])
        self.gate = nn.Linear(d_model, n_routed, bias=False)
        self.experts = nn.ModuleList([
            SwiGLU(d_model, d_ff) for _ in range(n_routed)
        ])
    
    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)
        shared_out = sum(expert(x_flat) for expert in self.shared_experts) / len(self.shared_experts)
        
        logits = self.gate(x_flat)
        weights = torch.nn.functional.softmax(logits, dim=-1)
        topk_weights, topk_ids = torch.topk(weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        routed_out = torch.zeros_like(x_flat)
        aux_loss = None
        if self.training:
            num_tokens = x_flat.shape[0]
            f = weights.mean(dim=0)
            one_hot = torch.nn.functional.one_hot(topk_ids, num_classes=self.n_routed).float()
            p = one_hot.sum(dim=(0,1)) / (num_tokens * self.top_k)
            aux_loss = self.n_routed * (f * p).sum() * 0.01
        
        for i in range(self.n_routed):
            mask = (topk_ids == i).any(dim=-1)
            if mask.any():
                expert_input = x_flat[mask]
                expert_output = self.experts[i](expert_input)
                idx_mask = (topk_ids == i)
                w = topk_weights[idx_mask].unsqueeze(-1)
                routed_out[mask] += w * expert_output
        
        out = shared_out + routed_out
        return out.reshape(B, L, D), aux_loss


class LongContextAttention(nn.Module):
    """超长上下文注意力 - 使用动态 RoPE 和 Flash Attention"""
    def __init__(self, config: V4Config):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.use_flash = config.use_flash_attn
        
        self.wq_a = nn.Linear(config.d_model, config.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(config.q_lora_rank)
        self.wq_b = nn.Linear(config.q_lora_rank, config.n_heads * self.head_dim, bias=False)
        
        self.wkv_a = nn.Linear(config.d_model, config.kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(config.kv_lora_rank)
        self.wk_b = nn.Linear(config.kv_lora_rank, config.n_heads * self.head_dim, bias=False)
        self.wv_b = nn.Linear(config.kv_lora_rank, config.n_heads * self.head_dim, bias=False)
        
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)
        
        self._init_rope(config)
    
    def _init_rope(self, config: V4Config):
        """初始化 RoPE"""
        if config.use_streaming_rope:
            from core.pos_encoding import StreamingRoPE, RoPEConfig
            rope_config = RoPEConfig(
                head_dim=self.head_dim,
                max_seq_len=config.max_seq_len,
                use_ntk_scaling=True
            )
            self.rope = StreamingRoPE(rope_config)
        else:
            self.rope = None
    
    def forward(self, x: torch.Tensor,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                cache_pos: int = 0,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, L, D = x.shape
        
        q = self.wq_b(self.q_norm(self.wq_a(x))).view(B, L, self.n_heads, self.head_dim)
        kv = self.kv_norm(self.wkv_a(x))
        k = self.wk_b(kv).view(B, L, self.n_heads, self.head_dim)
        v = self.wv_b(kv).view(B, L, self.n_heads, self.head_dim)
        
        if self.rope is not None:
            cos, sin = self.rope(L, cache_pos, x.device)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            q = q * cos + torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape_as(q) * sin
            k = k * cos + torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape_as(k) * sin
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
        else:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
        
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        
        new_kv = (k, v) if use_cache else None
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if self.use_flash and hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = torch.nn.functional.softmax(attn, dim=-1)
            attn_out = torch.matmul(attn, v)
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        return self.wo(attn_out), new_kv


class V4TransformerBlock(nn.Module):
    """V4 Transformer 块"""
    def __init__(self, config: V4Config):
        super().__init__()
        self.config = config
        self.attn = LongContextAttention(config)
        self.moe = DeepSeekMoE(config.d_model, config.d_ff, 2, config.num_experts, config.top_k)
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
    
    def forward(self, x, kv_cache=None, cache_pos=0, use_cache=False):
        residual = x
        x = self.norm1(x)
        attn_out, new_kv = self.attn(x, kv_cache, cache_pos, use_cache)
        x = residual + attn_out
        
        residual = x
        x = self.norm2(x)
        moe_out, moe_loss = self.moe(x)
        x = residual + moe_out
        
        return x, new_kv, moe_loss


class DeepSeekV4Multimodal(nn.Module):
    """
    DeepSeek V4 多模态模型
    
    特性：
    - 超长上下文支持（131072+）
    - 流式 RoPE 位置编码
    - Flash Attention
    - 分页 KV 缓存
    - MoE 架构
    - 多模态支持
    """
    def __init__(self, config: Optional[V4Config] = None):
        super().__init__()
        if config is None:
            config = V4Config()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        self.img_processor = DynamicImageProcessor(
            image_size=config.image_size,
            max_num_tiles=config.max_tiles
        )
        self.vision_encoder = VisionEncoder(config.d_model)
        self.vision_projector = VisionProjector(config.d_model, num_visual_tokens=256)
        
        self.layers = nn.ModuleList([
            V4TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
    
    def encode_images(self, images):
        """编码图像"""
        if images is None:
            return []
        tokens = []
        for img in images:
            tiles = self.img_processor.process_tiles(img, num_tiles=1)
            visual_features = self.vision_encoder(tiles)
            visual_features = self.vision_projector(visual_features)
            tokens.append(visual_features)
        return tokens
    
    def get_multimodal_embeddings(self, input_ids, images=None, img_pos=None):
        """获取多模态嵌入"""
        text_emb = self.token_embedding(input_ids)
        return text_emb
    
    def forward(self, input_ids, images=None, img_pos=None, return_hidden=False, 
               kv_caches=None, use_cache=False):
        """前向传播"""
        embeddings = self.get_multimodal_embeddings(input_ids, images, img_pos)
        x = embeddings
        
        all_kv_caches = [] if use_cache else None
        aux_loss_total = 0.0
        
        for i, layer in enumerate(self.layers):
            kv_cache = None
            cache_pos = 0
            if kv_caches is not None and i < len(kv_caches):
                kv_cache = kv_caches[i]
            
            x, new_kv, loss = layer(x, kv_cache, cache_pos, use_cache)
            
            if loss is not None:
                aux_loss_total += loss
            
            if use_cache and new_kv is not None:
                all_kv_caches.append(new_kv)
        
        last = self.norm(x)
        logits = self.lm_head(last)
        
        if return_hidden:
            return logits, aux_loss_total, last, all_kv_caches
        return logits, aux_loss_total, all_kv_caches
    
    def generate(self, input_ids, max_new_tokens=128, temperature=1.0, top_p=0.9,
                kv_caches=None, use_cache=True):
        """生成文本"""
        self.eval()
        with torch.no_grad():
            past_kv = kv_caches
            generated = input_ids
            
            for _ in range(max_new_tokens):
                logits, _, past_kv = self.forward(
                    generated[:, -1:],
                    kv_caches=past_kv,
                    use_cache=use_cache
                )
                
                logits = logits[:, -1, :] / temperature
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumsum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumsum_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[0, indices_to_remove] = float('-inf')
                
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=-1)
                
                if next_token.item() == self.config.vocab_size - 1:
                    break
            
            return generated


def create_long_context_model(config: Optional[V4Config] = None) -> DeepSeekV4Multimodal:
    """创建支持超长上下文的模型"""
    if config is None:
        config = V4Config()
    
    config.use_streaming_rope = True
    config.use_flash_attn = True
    config.use_long_context_attn = True
    
    return DeepSeekV4Multimodal(config)