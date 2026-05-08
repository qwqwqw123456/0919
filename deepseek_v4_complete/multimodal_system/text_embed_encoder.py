"""
Text Embed Encoder - 文本嵌入编码器模块
=======================================

该模块提供完整的文本嵌入编码功能，用于多模态系统中的文本处理。

主要组件：
- TextEncoderConfig: 文本编码器配置类
- TextEmbedEncoder: 基础文本嵌入编码器
- PositionalTextEncoder: 带位置编码的文本编码器
- RotaryTextEncoder: 使用旋转位置编码的文本编码器
- TextEncoderWithProjection: 带投影层的文本编码器

特点：
- 支持多种位置编码方式（绝对位置、旋转位置、相对位置）
- 支持动态词表扩展
- 支持条件文本生成
- 支持GPU加速
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class TextEncoderConfig:
    """
    文本编码器配置

    参数:
        vocab_size: 词表大小
        hidden_size: 隐藏层维度
        max_position_embeddings: 最大位置嵌入长度
        num_hidden_layers: 编码器层数
        num_attention_heads: 注意力头数
        intermediate_size: FFN中间层维度
        hidden_dropout_prob: Dropout概率
        attention_probs_dropout_prob: 注意力Dropout概率
        layer_norm_eps: LayerNorm epsilon
        pad_token_id: 填充token ID
        bos_token_id: 起始token ID
        eos_token_id: 结束token ID
        use_rotary_position_embedding: 是否使用旋转位置编码
        use_relative_position: 是否使用相对位置编码
        relative_attention_max_distance: 相对注意力最大距离
        rope_theta: 旋转位置编码的基础频率
        rope_scaling: 旋转位置编码的缩放因子
        tie_word_embeddings: 是否共享词嵌入权重
    """
    vocab_size: int = 129280
    hidden_size: int = 7168
    max_position_embeddings: int = 131072
    num_hidden_layers: int = 64
    num_attention_heads: int = 32
    intermediate_size: int = 2048
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    layer_norm_eps: float = 1e-6
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    use_rotary_position_embedding: bool = True
    use_relative_position: bool = False
    relative_attention_max_distance: int = 128
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    tie_word_embeddings: bool = False


class LearnedPositionalEmbedding(nn.Module):
    """
    可学习的位置嵌入

    使用可学习的参数表示位置信息
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = 0):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.weight = nn.Parameter(torch.zeros(num_embeddings, embedding_dim))
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            input_ids: 输入token IDs [B, L]

        返回:
            position_embeddings: 位置嵌入 [B, L, D]
        """
        seq_len = input_ids.shape[1]

        if seq_len > self.num_embeddings:
            self._extend_embeddings(seq_len)

        mask = input_ids.ne(self.padding_idx).long()
        positions = torch.cumsum(mask, dim=1).clamp_(min=1) * mask
        positions = positions - 1

        return F.embedding(
            positions,
            self.weight,
            padding_idx=self.padding_idx
        )

    def _extend_embeddings(self, new_size: int):
        """扩展位置嵌入表"""
        if new_size <= self.num_embeddings:
            return

        old_weight = self.weight.data
        new_weight = torch.zeros(new_size, self.embedding_dim, device=old_weight.device)
        nn.init.trunc_normal_(new_weight, std=0.02)
        new_weight[:self.num_embeddings] = old_weight
        self.weight = nn.Parameter(new_weight)
        self.num_embeddings = new_size


class SinusoidalPositionalEmbedding(nn.Module):
    """
    正弦位置嵌入

    使用正弦和余弦函数生成位置编码，具有良好的外推能力
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        weight = self._generate_embedding(num_embeddings, embedding_dim)
        self.register_buffer('weight', weight)

    def _generate_embedding(
        self,
        num_embeddings: int,
        embedding_dim: int
    ) -> torch.Tensor:
        """生成位置编码矩阵"""
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float32).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).reshape(num_embeddings, -1)

        if embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            input_ids: 输入token IDs [B, L]

        返回:
            position_embeddings: 位置嵌入 [B, L, D]
        """
        seq_len = input_ids.shape[1]

        if seq_len > self.num_embeddings:
            weight = self._generate_embedding(seq_len, self.embedding_dim)
            self.register_buffer('weight', weight)
            self.num_embeddings = seq_len

        return self.weight[:seq_len].unsqueeze(0)


class RotaryPositionalEmbedding(nn.Module):
    """
    旋转位置编码 (RoPE)

    通过旋转操作将位置信息注入到Query和Key中
    支持线性缩放以处理更长的序列
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 32768,
        base: float = 10000.0,
        scaling_factor: float = 1.0
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_factor = scaling_factor

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        self._build_cos_sin_cache(max_seq_len)

    def _build_cos_sin_cache(self, max_seq_len: int):
        """预计算cos和sin值"""
        seq = torch.arange(max_seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        scaled_seq = seq / self.scaling_factor

        angles = torch.outer(scaled_seq, self.inv_freq)
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        self.register_buffer('cos_cached', emb, persistent=False)
        self.register_buffer('sin_cached', emb, persistent=False)

    def forward(
        self,
        seq_len: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取指定长度的cos和sin编码

        参数:
            seq_len: 序列长度
            device: 设备

        返回:
            cos, sin: 旋转编码的cos和sin值
        """
        if seq_len > self.max_seq_len:
            self._build_cos_sin_cache(seq_len)
            self.max_seq_len = seq_len

        return (
            self.cos_cached[:seq_len].to(device),
            self.sin_cached[:seq_len].to(device)
        )


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    应用旋转位置编码

    参数:
        q: Query张量 [B, H, L, D]
        k: Key张量 [B, H, L, D]
        cos: cos编码 [L, D]
        sin: sin编码 [L, D]

    返回:
        q, k: 应用旋转后的张量
    """
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)

    return q, k


class RelativePositionBias(nn.Module):
    """
    相对位置偏置

    在注意力分数中加入相对位置信息
    """

    def __init__(
        self,
        num_heads: int,
        max_distance: int = 128,
        bidirectional: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.bidirectional = bidirectional

        num_buckets = max_distance * 2
        if bidirectional:
            num_buckets = max_distance * 2 + 1

        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    def _relative_position_bucket(
        self,
        relative_position: torch.Tensor,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128
    ) -> torch.Tensor:
        """计算相对位置的bucket索引"""
        ret = 0
        n = -relative_position

        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) /
            math.log(max_distance / max_exact) *
            (num_buckets - max_exact)
        ).to(torch.long)

        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, seq_len: int, seq_len_q: int) -> torch.Tensor:
        """
        生成相对位置偏置

        参数:
            seq_len: Key序列长度
            seq_len_q: Query序列长度

        返回:
            relative_bias: 相对位置偏置 [1, H, L, L]
        """
        device = self.relative_attention_bias.weight.device
        q_pos = torch.arange(seq_len_q, dtype=torch.long, device=device)
        k_pos = torch.arange(seq_len, dtype=torch.long, device=device)

        relative_position = k_pos[None, :] - q_pos[:, None]
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.max_distance * 2 + 1 if self.bidirectional else self.max_distance,
            max_distance=self.max_distance
        )

        relative_bias = self.relative_attention_bias(relative_position_bucket)
        relative_bias = relative_bias.permute(2, 0, 1).unsqueeze(0)

        return relative_bias


class TextAttention(nn.Module):
    """
    文本注意力模块

    支持多头注意力和旋转位置编码
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float = 0.0,
        rope: Optional[RotaryPositionalEmbedding] = None,
        use_relative_position: bool = False,
        num_relative_distance: int = 128
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.all_head_size = num_attention_heads * self.head_dim

        self.q_proj = nn.Linear(hidden_size, self.all_head_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.all_head_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.all_head_size, bias=False)
        self.o_proj = nn.Linear(self.all_head_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.rope = rope
        self.use_relative_position = use_relative_position

        if use_relative_position:
            self.relative_bias = RelativePositionBias(
                num_attention_heads,
                num_relative_distance
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播

        参数:
            hidden_states: 输入隐藏状态 [B, L, D]
            attention_mask: 注意力掩码 [B, L] 或 [B, L, L]
            output_attentions: 是否输出注意力权重

        返回:
            context: 上下文向量 [B, L, D]
            attention_probs: 注意力权重 [B, H, L, L]（可选）
        """
        B, L, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(B, L, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_attention_heads, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            cos, sin = self.rope(L, hidden_states.device)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        scale = 1.0 / math.sqrt(self.head_dim)
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) * scale

        if self.use_relative_position:
            relative_bias = self.relative_bias(L, L)
            attention_scores = attention_scores + relative_bias

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
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


class TextMLP(nn.Module):
    """
    文本前馈网络

    包含两层全连接层和激活函数
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TransformerBlock(nn.Module):
    """
    Transformer编码器块

    包含自注意力和前馈网络
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-6,
        rope: Optional[RotaryPositionalEmbedding] = None,
        use_relative_position: bool = False
    ):
        super().__init__()

        self.attention = TextAttention(
            hidden_size,
            num_attention_heads,
            dropout,
            rope,
            use_relative_position
        )

        self.mlp = TextMLP(hidden_size, intermediate_size, dropout)
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """前向传播"""
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)

        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions
        )
        attention_output = attention_outputs[0]
        attention_probs = attention_outputs[1] if output_attentions else None

        hidden_states = residual + self.dropout1(attention_output)

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = residual + self.dropout2(self.mlp(hidden_states))

        outputs = (hidden_states,)
        if output_attentions:
            outputs = (hidden_states, attention_probs)

        return outputs


class TextEncoderLayer(nn.Module):
    """
    文本编码器层

    完整的Transformer编码器层序列
    """

    def __init__(
        self,
        config: TextEncoderConfig,
        rope: Optional[RotaryPositionalEmbedding] = None
    ):
        super().__init__()
        self.config = config

        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.hidden_size,
                config.num_attention_heads,
                config.intermediate_size,
                config.attention_probs_dropout_prob,
                config.layer_norm_eps,
                rope,
                config.use_relative_position
            )
            for _ in range(config.num_hidden_layers)
        ])

        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        前向传播

        参数:
            hidden_states: 输入隐藏状态 [B, L, D]
            attention_mask: 注意力掩码
            output_hidden_states: 是否返回所有隐藏状态
            output_attentions: 是否返回注意力权重

        返回:
            last_hidden_state: 最后一层隐藏状态
            hidden_states: 所有隐藏状态（可选）
            attentions: 所有注意力权重（可选）
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = block(
                hidden_states,
                attention_mask,
                output_attentions
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)

        return outputs


class TextEmbedEncoder(nn.Module):
    """
    文本嵌入编码器

    完整的文本编码器，包含词嵌入、位置编码和Transformer层
    """

    def __init__(self, config: Optional[TextEncoderConfig] = None, **kwargs):
        """
        初始化文本嵌入编码器

        参数:
            config: 文本编码器配置
            **kwargs: 配置参数（当config为None时使用）
        """
        super().__init__()

        if config is None:
            config = TextEncoderConfig(**kwargs)

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id
        )

        if config.use_rotary_position_embedding:
            rope_scaling = config.rope_scaling or {}
            self.rope = RotaryPositionalEmbedding(
                dim=config.hidden_size // config.num_attention_heads,
                max_seq_len=config.max_position_embeddings,
                base=config.rope_theta,
                scaling_factor=rope_scaling.get('factor', 1.0)
            )
        else:
            self.rope = None

        self.position_embedding = LearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.hidden_size,
            config.pad_token_id
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.encoder = TextEncoderLayer(config, self.rope)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        参数:
            input_ids: 输入token IDs [B, L]
            attention_mask: 注意力掩码 [B, L]
            position_ids: 位置IDs [B, L]
            output_hidden_states: 是否返回所有隐藏状态
            output_attentions: 是否返回注意力权重
            return_dict: 是否返回字典格式输出

        返回:
            output: 编码器输出
        """
        token_embeds = self.token_embedding(input_ids)

        if position_ids is None:
            position_ids = torch.arange(
                input_ids.shape[1],
                device=input_ids.device
            ).unsqueeze(0).expand_as(input_ids)

        position_embeds = self.position_embedding(position_ids)

        hidden_states = token_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask,
            output_hidden_states,
            output_attentions
        )

        sequence_output = encoder_outputs[0]

        if not return_dict:
            outputs = (sequence_output,)
            if output_hidden_states:
                outputs = outputs + (encoder_outputs[1],)
            if output_attentions:
                outputs = outputs + (encoder_outputs[2],)
            return outputs

        return {
            'last_hidden_state': sequence_output,
            'hidden_states': encoder_outputs[1] if output_hidden_states else None,
            'attentions': encoder_outputs[2] if output_attentions else None,
        }

    def get_input_embeddings(self) -> nn.Embedding:
        """获取输入嵌入层"""
        return self.token_embedding

    def set_input_embeddings(self, embeddings: nn.Embedding):
        """设置输入嵌入层"""
        self.token_embedding = embeddings


class PositionalTextEncoder(nn.Module):
    """
    带位置编码的文本编码器

    专门处理位置编码的文本编码器
    """

    def __init__(
        self,
        vocab_size: int = 129280,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 8192
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = SinusoidalPositionalEmbedding(max_seq_len, hidden_size)

        self.encoder_layers = nn.ModuleList([
            TransformerBlock(
                hidden_size,
                num_heads,
                hidden_size * 4,
                dropout=0.1
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播

        参数:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码

        返回:
            last_hidden_state: 最后隐藏状态
        """
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(input_ids)

        hidden_states = token_embeds + position_embeds

        for layer in self.encoder_layers:
            hidden_states, _ = layer(hidden_states, attention_mask)

        hidden_states = self.norm(hidden_states)

        return hidden_states


class RotaryTextEncoder(nn.Module):
    """
    使用旋转位置编码的文本编码器

    专门使用RoPE的文本编码器
    """

    def __init__(
        self,
        vocab_size: int = 129280,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 8192,
        rope_theta: float = 10000.0
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.rope = RotaryPositionalEmbedding(
            dim=hidden_size // num_heads,
            max_seq_len=max_seq_len,
            base=rope_theta
        )

        self.encoder_layers = nn.ModuleList([
            TransformerBlock(
                hidden_size,
                num_heads,
                hidden_size * 4,
                dropout=0.1,
                rope=self.rope
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播

        参数:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码

        返回:
            last_hidden_state: 最后隐藏状态
        """
        token_embeds = self.token_embedding(input_ids)

        hidden_states = token_embeds

        for layer in self.encoder_layers:
            hidden_states, _ = layer(hidden_states, attention_mask)

        hidden_states = self.norm(hidden_states)

        return hidden_states


class TextEncoderWithProjection(nn.Module):
    """
    带投影层的文本编码器

    在标准文本编码器后添加投影层，用于多模态融合
    """

    def __init__(
        self,
        encoder: TextEmbedEncoder,
        projection_dim: int = 768,
        use_gated_projection: bool = True
    ):
        super().__init__()
        self.encoder = encoder

        if use_gated_projection:
            self.projection = GatedProjection(
                encoder.config.hidden_size,
                projection_dim
            )
        else:
            self.projection = nn.Linear(
                encoder.config.hidden_size,
                projection_dim
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播

        参数:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码

        返回:
            projected: 投影后的特征
        """
        encoder_outputs = self.encoder(
            input_ids,
            attention_mask,
            return_dict=True
        )

        hidden_states = encoder_outputs['last_hidden_state']

        projected = self.projection(hidden_states)

        return projected


class GatedProjection(nn.Module):
    """
    门控投影层

    通过门控机制控制信息流
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)

        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        proj = self.proj(x)
        gate = torch.sigmoid(self.gate(x))
        return proj * gate


class MultiScaleTextEncoder(nn.Module):
    """
    多尺度文本编码器

    在不同粒度上编码文本，捕捉不同层次的语义信息
    """

    def __init__(
        self,
        config: TextEncoderConfig,
        num_scales: int = 3
    ):
        super().__init__()

        self.num_scales = num_scales
        self.config = config

        self.encoders = nn.ModuleList([
            TextEmbedEncoder(config) for _ in range(num_scales)
        ])

        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)

        self.fusion = nn.Linear(
            config.hidden_size * num_scales,
            config.hidden_size
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        多尺度编码

        参数:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码

        返回:
            fused: 融合后的特征
        """
        scale_outputs = []

        for encoder in self.encoders:
            outputs = encoder(
                input_ids,
                attention_mask,
                return_dict=True
            )
            scale_outputs.append(outputs['last_hidden_state'])

        weights = F.softmax(self.scale_weights, dim=0)
        weighted_outputs = [
            w * out for w, out in zip(weights, scale_outputs)
        ]

        concatenated = torch.cat(weighted_outputs, dim=-1)
        fused = self.fusion(concatenated)

        return fused


class ConditionalTextEncoder(nn.Module):
    """
    条件文本编码器

    支持条件输入的文本编码器，用于条件生成任务
    """

    def __init__(
        self,
        config: TextEncoderConfig,
        num_condition_types: int = 4
    ):
        super().__init__()

        self.config = config
        self.encoder = TextEmbedEncoder(config)

        self.condition_embedding = nn.ModuleDict({
            str(i): nn.Linear(config.hidden_size, config.hidden_size)
            for i in range(num_condition_types)
        })

        self.condition_scale = nn.ModuleDict({
            str(i): nn.Linear(config.hidden_size, config.hidden_size)
            for i in range(num_condition_types)
        })

    def forward(
        self,
        input_ids: torch.Tensor,
        condition_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        条件编码

        参数:
            input_ids: 输入token IDs
            condition_ids: 条件IDs
            attention_mask: 注意力掩码

        返回:
            encoded: 条件编码后的特征
        """
        outputs = self.encoder(
            input_ids,
            attention_mask,
            return_dict=True
        )

        hidden_states = outputs['last_hidden_state']

        if condition_ids is not None:
            for i, cond_id in enumerate(condition_ids):
                cond_emb = self.condition_embedding[str(cond_id)](hidden_states)
                cond_scale = self.condition_scale[str(cond_id)](hidden_states)
                hidden_states = hidden_states * (1 + cond_scale) + cond_emb

        return hidden_states


class TextEncoderPooler(nn.Module):
    """
    文本编码器池化器

    用于从文本编码器输出中提取固定长度的表示
    """

    def __init__(
        self,
        hidden_size: int,
        pooler_type: str = 'cls'
    ):
        super().__init__()
        self.pooler_type = pooler_type

        if pooler_type == 'cls':
            self.dense = nn.Linear(hidden_size, hidden_size)
            self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        池化

        参数:
            hidden_states: 编码器输出的隐藏状态 [B, L, D]

        返回:
            pooled: 池化后的表示 [B, D]
        """
        if self.pooler_type == 'cls':
            pooled = self.dense(hidden_states[:, 0])
            pooled = self.activation(pooled)
        elif self.pooler_type == 'mean':
            pooled = hidden_states.mean(dim=1)
        elif self.pooler_type == 'max':
            pooled = hidden_states.max(dim=1)[0]
        else:
            pooled = hidden_states[:, 0]

        return pooled


def create_text_encoder(
    encoder_type: str = 'standard',
    **kwargs
) -> TextEmbedEncoder:
    """
    创建文本编码器的工厂函数

    参数:
        encoder_type: 编码器类型 ('standard', 'rotary', 'positional')
        **kwargs: 配置参数

    返回:
        TextEmbedEncoder 实例
    """
    if encoder_type == 'standard':
        config = TextEncoderConfig(**kwargs)
        return TextEmbedEncoder(config)

    elif encoder_type == 'rotary':
        vocab_size = kwargs.get('vocab_size', 129280)
        hidden_size = kwargs.get('hidden_size', 768)
        num_layers = kwargs.get('num_layers', 12)
        num_heads = kwargs.get('num_heads', 12)
        max_seq_len = kwargs.get('max_seq_len', 8192)
        rope_theta = kwargs.get('rope_theta', 10000.0)

        return RotaryTextEncoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta
        )

    elif encoder_type == 'positional':
        vocab_size = kwargs.get('vocab_size', 129280)
        hidden_size = kwargs.get('hidden_size', 768)
        num_layers = kwargs.get('num_layers', 12)
        num_heads = kwargs.get('num_heads', 12)
        max_seq_len = kwargs.get('max_seq_len', 8192)

        return PositionalTextEncoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len
        )

    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
