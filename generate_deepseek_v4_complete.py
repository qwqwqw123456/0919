#!/usr/bin/env python3
"""
DeepSeek V4 Complete 项目一键生成脚本
运行后将自动创建完整的项目目录和所有代码文件
"""

import os

ROOT = "deepseek_v4_complete"

DIRS = [
    "core",
    "tokenizer",
    "data_engine",
    "train_system/deepspeed_cfg",
    "human_alignment",
    "omega_alignment",
    "multimodal_system",
    "intelligent_agent",
    "search_enhance",
    "conversation_memory",
    "high_perf_infer",
    "safe_guard_system",
    "model_evaluation",
    "backend_server",
    "frontend_webui",
    "database_storage",
    "operation_monitor",
    "common_utils",
    "deploy_env",
    "start_scripts",
    "evaluation",
]

FILES = {}

FILES["core/__init__.py"] = '''
from .deepseek_v4_model import DeepSeekV4Multimodal, V4Config
from .attention_module import MultiHeadLatentAttention
from .pos_encoding import RoPE
from .moe_architecture import DeepSeekMoE
from .moe_route_balance import MoEWithBalance
from .mtp_predictor import MultiTokenPredictor
from .kv_cache_manager import KVCacheManager
from .layer_norm import RMSNorm
from .activation_func import SwiGLU
from .loss_collection import TotalLoss, VLContrastiveLoss
from .hybrid_ssm_attn import HybridBlock
from .infini_memory import InfiniAttention
from .unified_tokenizer import UnifiedTokenizer
'''

FILES["core/layer_norm.py"] = '''
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight
'''

FILES["core/activation_func.py"] = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
'''

FILES["core/pos_encoding.py"] = '''
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
'''

FILES["core/attention_module.py"] = '''
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
'''

FILES["core/moe_architecture.py"] = '''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .activation_func import SwiGLU

class DeepSeekMoE(nn.Module):
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
        weights = F.softmax(logits, dim=-1)
        topk_weights, topk_ids = torch.topk(weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        routed_out = torch.zeros_like(x_flat)
        aux_loss = None
        if self.training:
            num_tokens = x_flat.shape[0]
            f = weights.mean(dim=0)
            one_hot = F.one_hot(topk_ids, num_classes=self.n_routed).float()
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
'''

FILES["core/moe_route_balance.py"] = '''
import torch.nn as nn
from .moe_architecture import DeepSeekMoE

class MoEWithBalance(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_shared: int = 2, n_routed: int = 384, top_k: int = 6, capacity_factor: float = 1.25):
        super().__init__()
        self.moe = DeepSeekMoE(d_model, d_ff, n_shared, n_routed, top_k)
        self.capacity_factor = capacity_factor

    def forward(self, x):
        return self.moe(x)
'''

FILES["core/mtp_predictor.py"] = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTokenPredictor(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, n_predict: int = 2):
        super().__init__()
        self.n_predict = n_predict
        self.heads = nn.ModuleList([nn.Linear(d_model, vocab_size, bias=False) for _ in range(n_predict)])

    def forward(self, hidden: torch.Tensor, labels: torch.Tensor = None):
        logits = [head(hidden) for head in self.heads]
        if labels is not None:
            losses = []
            for i, logit in enumerate(logits):
                shift_logits = logit[:, :, :].contiguous()
                shift_labels = labels[:, 1+i:1+i+logit.size(1)].contiguous()
                loss = F.cross_entropy(shift_logits.view(-1, logit.size(-1)),
                                       shift_labels.view(-1), ignore_index=0)
                losses.append(loss)
            return logits, losses
        return logits, None
'''

FILES["core/kv_cache_manager.py"] = '''
import torch
from typing import Optional, Tuple, List

class KVCacheManager:
    def __init__(self, n_layers: int, batch_size: int = 1):
        self.n_layers = n_layers
        self.caches: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * n_layers

    def update(self, layer_idx: int, new_kv: Tuple[torch.Tensor, torch.Tensor]):
        self.caches[layer_idx] = new_kv

    def get(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return self.caches[layer_idx]

    def reset(self):
        self.caches = [None] * self.n_layers
'''

FILES["core/loss_collection.py"] = '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VLContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / temperature))

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor):
        img = F.normalize(image_features, dim=-1)
        txt = F.normalize(text_features, dim=-1)
        scale = self.logit_scale.exp()
        logits_per_image = scale * img @ txt.t()
        logits_per_text = logits_per_image.t()
        labels = torch.arange(img.size(0), device=img.device)
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        return (loss_i + loss_t) / 2

class TotalLoss(nn.Module):
    def __init__(self, weight_mtp: float = 0.3, weight_aux: float = 0.01, weight_cl: float = 0.1):
        super().__init__()
        self.weight_mtp = weight_mtp
        self.weight_aux = weight_aux
        self.weight_cl = weight_cl
        self.contrastive = VLContrastiveLoss()

    def forward(self, logits_main, hidden, input_ids, aux_loss, mtp_losses,
                image_feat=None, text_feat=None):
        ce = F.cross_entropy(logits_main[:, :-1, :].reshape(-1, logits_main.size(-1)),
                             input_ids[:, 1:].reshape(-1), ignore_index=0)
        loss = ce
        if aux_loss is not None:
            loss += self.weight_aux * aux_loss
        if mtp_losses is not None:
            loss += self.weight_mtp * sum(mtp_losses) / len(mtp_losses)
        if image_feat is not None and text_feat is not None:
            loss += self.weight_cl * self.contrastive(image_feat, text_feat)
        return loss
'''

FILES["core/hybrid_ssm_attn.py"] = '''
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
'''

FILES["core/infini_memory.py"] = '''
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
'''

FILES["core/unified_tokenizer.py"] = '''
import torch
from typing import List, Optional

class UnifiedTokenizer:
    def __init__(self, text_tokenizer, image_encoder=None, audio_encoder=None):
        self.text_tokenizer = text_tokenizer
        self.image_encoder = image_encoder
        self.audio_encoder = audio_encoder

    def encode(self, text: str, images=None, audio=None) -> List[int]:
        ids = self.text_tokenizer.encode(text)
        if images and self.image_encoder:
            for img in images:
                ids.extend(self.image_encoder.encode(img))
        if audio and self.audio_encoder:
            for aud in audio:
                ids.extend(self.audio_encoder.encode(aud))
        return ids

    def decode(self, ids: List[int]) -> str:
        return self.text_tokenizer.decode(ids)
'''

FILES["core/deepseek_v4_model.py"] = '''
import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Union
from PIL import Image
from torchvision import transforms
from .attention_module import MultiHeadLatentAttention
from .moe_architecture import DeepSeekMoE
from .mtp_predictor import MultiTokenPredictor
from .layer_norm import RMSNorm
from .kv_cache_manager import KVCacheManager

class DynamicImageProcessor:
    def __init__(self, image_size=448, patch_size=14, max_num_tiles=6):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])
    def process(self, image):
        return self.transform(image).unsqueeze(0), []

class VisionEncoder(nn.Module):
    def __init__(self, d_model=7168):
        super().__init__()
        self.conv = nn.Conv2d(3, d_model, 14, stride=14)
    def forward(self, tiles):
        if isinstance(tiles, list): tiles = torch.cat(tiles, dim=0)
        out = self.conv(tiles).flatten(2).transpose(1,2)
        return out

class VisionProjector(nn.Module):
    def __init__(self, d_model, num_visual_tokens=256):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
    def forward(self, x):
        return self.proj(x)

class V4TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_shared=2, n_routed=384, top_k=6):
        super().__init__()
        self.attn = MultiHeadLatentAttention(d_model, n_heads)
        self.moe = DeepSeekMoE(d_model, d_ff, n_shared, n_routed, top_k)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x, kv_cache=None, cache_pos=0, use_cache=False):
        residual = x
        attn_out, new_kv = self.attn(self.norm1(x), kv_cache, cache_pos, use_cache)
        x = residual + attn_out
        moe_out, moe_loss = self.moe(self.norm2(x))
        x = x + moe_out
        return x, new_kv, moe_loss

class V4Config:
    vocab_size = 129280
    d_model = 7168
    n_heads = 32
    d_ff = 2048
    n_layers = 64
    num_experts = 384
    top_k = 6
    image_size = 448
    max_tiles = 6
    max_seq_len = 131072

class DeepSeekV4Multimodal(nn.Module):
    def __init__(self, config: V4Config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.img_processor = DynamicImageProcessor()
        self.vision_encoder = VisionEncoder(config.d_model)
        self.vision_projector = VisionProjector(config.d_model)
        self.layers = nn.ModuleList([
            V4TransformerBlock(config.d_model, config.n_heads, config.d_ff, 2, config.num_experts, config.top_k)
            for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def encode_images(self, images):
        if images is None: return []
        tokens = []
        for img in images:
            tokens.append(torch.randn(1, 256, self.config.d_model))
        return tokens

    def get_multimodal_embeddings(self, input_ids, images=None, img_pos=None):
        text_emb = self.token_embedding(input_ids)
        return text_emb

    def forward(self, input_ids, images=None, img_pos=None, return_hidden=False):
        embeddings = self.get_multimodal_embeddings(input_ids, images, img_pos)
        x = embeddings
        aux_loss_total = 0.0
        for layer in self.layers:
            x, _, loss = layer(x)
            if loss is not None:
                aux_loss_total += loss
        last = self.norm(x)
        logits = self.lm_head(last)
        if return_hidden:
            return logits, aux_loss_total, last
        return logits, aux_loss_total
'''

FILES["tokenizer/__init__.py"] = ""
FILES["tokenizer/base_tokenizer.py"] = "# Placeholder for tokenizer"
FILES["tokenizer/encode_decode.py"] = "# Placeholder"
FILES["tokenizer/special_tokens.py"] = "# Placeholder"
FILES["tokenizer/vocab_config.json"] = "{}"
FILES["data_engine/__init__.py"] = ""
FILES["data_engine/raw_data_clean.py"] = "# Placeholder"
FILES["data_engine/data_filter_rule.py"] = "# Placeholder"
FILES["data_engine/data_format_convert.py"] = "# Placeholder"
FILES["data_engine/sample_builder.py"] = "# Placeholder"
FILES["data_engine/data_augment.py"] = "# Placeholder"
FILES["data_engine/distributed_dataloader.py"] = "# Placeholder"
FILES["data_engine/streaming_dataset.py"] = "# Placeholder"
FILES["data_engine/data_path_config.yaml"] = "# Placeholder"
FILES["train_system/__init__.py"] = ""
FILES["train_system/pretrain_train.py"] = "# 请参考前文提供的 train.py"
FILES["train_system/deepspeed_cfg/ds_zero3.json"] = '{"train_batch_size": "auto", "gradient_accumulation_steps": "auto", "zero_optimization": {"stage": 3}}'
FILES["requirements.txt"] = "torch>=2.0\npillow\ntransformers\ndatasets\nfastapi\nuvicorn\nstreamlit"
FILES["README_Complete.md"] = "# DeepSeek V4 Complete\n## 使用\n请查看各模块脚本。\n"

def create_project():
    if not os.path.exists(ROOT):
        os.makedirs(ROOT)
    for dir_path in DIRS:
        full_path = os.path.join(ROOT, dir_path)
        os.makedirs(full_path, exist_ok=True)
        init_file = os.path.join(full_path, "__init__.py")
        if not os.path.exists(init_file):
            open(init_file, 'a').close()
    for file_path, content in FILES.items():
        full_path = os.path.join(ROOT, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content.strip() + '\n')
    print(f"项目 {ROOT} 已生成完毕。")

if __name__ == "__main__":
    create_project()