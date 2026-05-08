"""
Multimodal Unified Encoder - 多模态统一编码器
==========================================

该模块实现统一的多模态编码器架构，能够同时处理文本、图像、音频等多种模态。

主要组件：
- UnifiedEncoderConfig: 统一编码器配置
- MultimodalUnifiedEncoder: 多模态统一编码器
- TextBranch: 文本分支
- ImageBranch: 图像分支
- AudioBranch: 音频分支

特点：
- 统一的编码架构
- 模态无关的表示学习
- 跨模态注意力机制
- 可扩展的模态支持
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class UnifiedEncoderConfig:
    """
    统一编码器配置

    参数:
        hidden_size: 隐藏层维度
        num_hidden_layers: 编码器层数
        num_attention_heads: 注意力头数
        intermediate_size: FFN中间层维度
        max_seq_len: 最大序列长度
        vocab_size: 词表大小
        dropout_prob: Dropout概率
        layer_norm_eps: LayerNorm epsilon
        use_rope: 是否使用旋转位置编码
        rope_theta: 旋转位置编码基础频率
        use_modality_embedding: 是否使用模态嵌入
        num_modalities: 模态数量
    """
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_seq_len: int = 8192
    vocab_size: int = 129280
    dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-6
    use_rope: bool = True
    rope_theta: float = 10000.0
    use_modality_embedding: bool = True
    num_modalities: int = 3


class UnifiedAttention(nn.Module):
    """
    统一注意力机制

    支持跨模态注意力和模态内注意力
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播

        参数:
            hidden_states: 隐藏状态 [B, L, D]
            attention_mask: 注意力掩码
            output_attentions: 是否输出注意力权重

        返回:
            context: 上下文向量
            attention_weights: 注意力权重（可选）
        """
        B, L, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, v)
        context = context.transpose(1, 2).contiguous().view(B, L, -1)
        context = self.o_proj(context)

        outputs = (context,)
        if output_attentions:
            outputs = (context, attention_probs)

        return outputs


class UnifiedTransformerBlock(nn.Module):
    """
    统一Transformer块

    包含注意力和前馈网络
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-6
    ):
        super().__init__()

        self.attention = UnifiedAttention(hidden_size, num_attention_heads, dropout)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """前向传播"""
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)

        attention_outputs = self.attention(hidden_states, attention_mask, output_attentions)
        attention_output = attention_outputs[0]

        hidden_states = residual + attention_output
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))

        outputs = (hidden_states,)
        if output_attentions:
            outputs = (hidden_states, attention_outputs[1])

        return outputs


class ModalityTypeEmbedding(nn.Module):
    """
    模态类型嵌入

    为不同模态添加可学习的嵌入标识
    """

    def __init__(
        self,
        hidden_size: int,
        num_modalities: int = 3
    ):
        super().__init__()
        self.modality_embeddings = nn.ModuleDict({
            'text': nn.Embedding(1, hidden_size),
            'image': nn.Embedding(1, hidden_size),
            'audio': nn.Embedding(1, hidden_size)
        })

        for modality in self.modality_embeddings.values():
            nn.init.trunc_normal_(modality.weight, std=0.02)

    def forward(
        self,
        modality_type: str,
        batch_size: int,
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        获取模态嵌入

        参数:
            modality_type: 模态类型
            batch_size: 批次大小
            seq_len: 序列长度
            device: 设备

        返回:
            模态嵌入张量 [B, L, D]
        """
        if modality_type not in self.modality_embeddings:
            modality_type = 'text'

        modality_idx = torch.zeros(batch_size, 1, device=device, dtype=torch.long)
        modality_emb = self.modality_embeddings[modality_type](modality_idx)
        modality_emb = modality_emb.expand(-1, seq_len, -1)

        return modality_emb


class TextBranch(nn.Module):
    """
    文本分支

    负责文本模态的编码
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int = 8192
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)

        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

        nn.init.trunc_normal_(self.token_embedding.weight, std=0.02)
        nn.init.trunc_normal_(self.position_embedding.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播

        参数:
            input_ids: 输入token IDs [B, L]
            attention_mask: 注意力掩码

        返回:
            text_features: 文本特征 [B, L, D]
        """
        B, L = input_ids.shape

        token_embeds = self.token_embedding(input_ids)

        position_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        position_embeds = self.position_embedding(position_ids)

        hidden_states = token_embeds + position_embeds
        hidden_states = self.dropout(self.norm(hidden_states))

        return hidden_states

    def get_output_dim(self) -> int:
        """获取输出维度"""
        return self.token_embedding.embedding_dim


class ImageBranch(nn.Module):
    """
    图像分支

    负责图像模态的编码
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        hidden_size: int = 768
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.patch_embedding = nn.Conv2d(
            3, hidden_size,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, hidden_size)
        )

        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            pixel_values: 像素值 [B, C, H, W]

        返回:
            image_features: 图像特征 [B, N+1, D]
        """
        B = pixel_values.shape[0]

        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        hidden_states = torch.cat([cls_tokens, patch_embeds], dim=1)

        hidden_states = hidden_states + self.position_embedding
        hidden_states = self.dropout(self.norm(hidden_states))

        return hidden_states

    def get_output_dim(self) -> int:
        """获取输出维度"""
        return self.patch_embedding.out_channels


class AudioBranch(nn.Module):
    """
    音频分支

    负责音频模态的编码
    """

    def __init__(
        self,
        n_mels: int = 80,
        n_fft: int = 400,
        hidden_size: int = 768
    ):
        super().__init__()
        self.n_mels = n_mels
        self.n_fft = n_fft

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.projection = nn.Linear(128 * (n_mels // 8), hidden_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.position_embedding = nn.Parameter(torch.zeros(1, 1000, hidden_size))

        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(
        self,
        audio_features: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播

        参数:
            audio_features: 音频特征 [B, 1, n_mels, T]

        返回:
            audio_features: 音频特征 [B, L, D]
        """
        B = audio_features.shape[0]

        x = self.conv_layers(audio_features)
        x = x.flatten(2).transpose(1, 2)

        x = self.projection(x)

        max_len = min(x.shape[1], self.position_embedding.shape[1])
        x = x[:, :max_len]
        pos_emb = self.position_embedding[:, :max_len]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        hidden_states = torch.cat([cls_tokens, x], dim=1)
        pos_emb = torch.cat([
            torch.zeros(1, 1, pos_emb.shape[-1], device=pos_emb.device),
            pos_emb
        ], dim=1)

        hidden_states = hidden_states + pos_emb
        hidden_states = self.dropout(self.norm(hidden_states))

        return hidden_states

    def get_output_dim(self) -> int:
        """获取输出维度"""
        return self.projection.out_features


class MultimodalUnifiedEncoder(nn.Module):
    """
    多模态统一编码器

    统一的编码器架构，能够同时处理文本、图像、音频等多种模态
    """

    def __init__(self, config: Optional[UnifiedEncoderConfig] = None, **kwargs):
        """
        初始化多模态统一编码器

        参数:
            config: 统一编码器配置
            **kwargs: 配置参数（当config为None时使用）
        """
        super().__init__()

        if config is None:
            config = UnifiedEncoderConfig(**kwargs)

        self.config = config

        self.text_branch = TextBranch(
            config.vocab_size,
            config.hidden_size,
            config.max_seq_len
        )

        self.image_branch = ImageBranch(
            hidden_size=config.hidden_size
        )

        self.audio_branch = AudioBranch(
            hidden_size=config.hidden_size
        )

        if config.use_modality_embedding:
            self.modality_embedding = ModalityTypeEmbedding(
                config.hidden_size,
                config.num_modalities
            )
        else:
            self.modality_embedding = None

        self.encoder_layers = nn.ModuleList([
            UnifiedTransformerBlock(
                config.hidden_size,
                config.num_attention_heads,
                config.intermediate_size,
                config.dropout_prob,
                config.layer_norm_eps
            )
            for _ in range(config.num_hidden_layers)
        ])

        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        编码文本

        参数:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码

        返回:
            text_features: 文本特征
        """
        return self.text_branch(input_ids, attention_mask)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        编码图像

        参数:
            pixel_values: 像素值

        返回:
            image_features: 图像特征
        """
        return self.image_branch(pixel_values)

    def encode_audio(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        编码音频

        参数:
            audio_features: 音频特征

        返回:
            audio_features_out: 音频特征
        """
        return self.audio_branch(audio_features)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        modality_type: str = "text",
        output_hidden_states: bool = False,
        output_attentions: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        统一前向传播

        参数:
            input_ids: 文本token IDs
            pixel_values: 图像像素值
            audio_features: 音频特征
            attention_mask: 注意力掩码
            modality_type: 当前处理的模态类型
            output_hidden_states: 是否返回所有隐藏状态
            output_attentions: 是否返回注意力权重

        返回:
            outputs: 编码器输出
        """
        if input_ids is not None:
            hidden_states = self.encode_text(input_ids, attention_mask)
        elif pixel_values is not None:
            hidden_states = self.encode_image(pixel_values)
        elif audio_features is not None:
            hidden_states = self.encode_audio(audio_features)
        else:
            raise ValueError("At least one modality input must be provided")

        if self.modality_embedding is not None:
            B = hidden_states.shape[0]
            L = hidden_states.shape[1]
            mod_emb = self.modality_embedding(modality_type, B, L, hidden_states.device)
            hidden_states = hidden_states + mod_emb

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer in self.encoder_layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask,
                output_attentions
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.final_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = {
            'last_hidden_state': hidden_states,
            'hidden_states': all_hidden_states,
            'attentions': all_attentions
        }

        return outputs

    def encode_multiple_modalities(
        self,
        text_input: Optional[torch.Tensor] = None,
        image_input: Optional[torch.Tensor] = None,
        audio_input: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        编码多个模态的输入

        参数:
            text_input: 文本输入
            image_input: 图像输入
            audio_input: 音频输入

        返回:
            各模态的编码结果
        """
        results = {}

        if text_input is not None:
            results['text'] = self.forward(
                input_ids=text_input,
                modality_type='text'
            )

        if image_input is not None:
            results['image'] = self.forward(
                pixel_values=image_input,
                modality_type='image'
            )

        if audio_input is not None:
            results['audio'] = self.forward(
                audio_features=audio_input,
                modality_type='audio'
            )

        return results

    def fuse_modalities(
        self,
        text_features: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        融合多个模态的特征

        参数:
            text_features: 文本特征
            image_features: 图像特征
            audio_features: 音频特征

        返回:
            fused: 融合后的特征
        """
        features = []

        if text_features is not None:
            features.append(text_features)
        if image_features is not None:
            features.append(image_features)
        if audio_features is not None:
            features.append(audio_features)

        if not features:
            raise ValueError("At least one modality feature must be provided")

        max_len = max(f.shape[1] for f in features)

        padded_features = []
        for f in features:
            if f.shape[1] < max_len:
                pad_len = max_len - f.shape[1]
                f = F.pad(f, (0, 0, 0, pad_len))
            padded_features.append(f)

        concatenated = torch.cat(padded_features, dim=1)

        hidden_states = concatenated
        for layer in self.encoder_layers:
            hidden_states, _ = layer(hidden_states, None, False)

        hidden_states = self.final_norm(hidden_states)

        return hidden_states

    def get_modality_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        获取指定模态的特征

        参数:
            input_ids: 文本输入
            pixel_values: 图像输入
            audio_features: 音频输入

        返回:
            特征向量（取CLS token）
        """
        if input_ids is not None:
            outputs = self.forward(input_ids=input_ids, modality_type='text')
        elif pixel_values is not None:
            outputs = self.forward(pixel_values=pixel_values, modality_type='image')
        elif audio_features is not None:
            outputs = self.forward(audio_features=audio_features, modality_type='audio')
        else:
            raise ValueError("At least one modality input must be provided")

        return outputs['last_hidden_state'][:, 0]


class CrossModalUnifiedEncoder(nn.Module):
    """
    跨模态统一编码器

    支持显式跨模态交互的统一编码器
    """

    def __init__(self, config: UnifiedEncoderConfig):
        super().__init__()
        self.config = config

        self.unified_encoder = MultimodalUnifiedEncoder(config)

        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(config.hidden_size, config.num_attention_heads)
            for _ in range(config.num_hidden_layers // 2)
        ])

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        跨模态前向传播

        参数:
            text_features: 文本特征
            image_features: 图像特征
            attention_mask: 注意力掩码

        返回:
            fused: 融合后的特征
        """
        text_hidden = text_features
        image_hidden = image_features

        for i, cross_attn in enumerate(self.cross_attention_layers):
            if i % 2 == 0:
                text_hidden = cross_attn(text_hidden, image_hidden)
                text_hidden = text_hidden + text_features
            else:
                image_hidden = cross_attn(image_hidden, text_hidden)
                image_hidden = image_hidden + image_features

        return text_hidden


class CrossAttentionLayer(nn.Module):
    """交叉注意力层"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int
    ):
        super().__init__()
        self.cross_attn = UnifiedAttention(hidden_size, num_heads)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor
    ) -> torch.Tensor:
        """前向传播"""
        attended, _ = self.cross_attn(query, None, False)
        output = self.norm(query + attended)
        return output


def create_unified_encoder(
    encoder_type: str = "standard",
    hidden_size: int = 768,
    **kwargs
) -> MultimodalUnifiedEncoder:
    """
    创建统一编码器的工厂函数

    参数:
        encoder_type: 编码器类型
        hidden_size: 隐藏层维度
        **kwargs: 其他配置参数

    返回:
        MultimodalUnifiedEncoder 实例
    """
    config = UnifiedEncoderConfig(
        hidden_size=hidden_size,
        **kwargs
    )

    if encoder_type == "cross_mododal":
        return CrossModalUnifiedEncoder(config)
    else:
        return MultimodalUnifiedEncoder(config)
