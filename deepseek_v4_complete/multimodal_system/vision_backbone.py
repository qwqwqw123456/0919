"""
Vision Backbone - 视觉骨干网络模块
=====================================

该模块实现了完整的 Vision Transformer (ViT) 架构，用于图像特征提取。

主要组件：
- ViTConfig: ViT 模型配置类
- PatchEmbedding: 图像分块嵌入层
- ViTAttention: 多头自注意力机制
- ViTMLP: 前馈神经网络
- ViTBlock: Transformer 编码器块
- ViTEncoder: 完整 ViT 编码器
- VisionBackbone: 主干网络类
- ImageTokenizer: 图像分词器
- TileProcessor: 图像瓦片处理器

参考论文：https://arxiv.org/abs/2010.11929
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class ViTConfig:
    """
    Vision Transformer 配置类

    参数:
        image_size: 输入图像大小，默认 224
        patch_size: 每个 patch 的大小，默认 16
        in_channels: 输入通道数，默认 3 (RGB)
        hidden_size: 隐藏层维度，默认 768
        num_hidden_layers: Transformer 层数，默认 12
        num_attention_heads: 注意力头数，默认 12
        intermediate_size: FFN 中间层维度，默认 3072
        hidden_dropout_prob: Dropout 概率，默认 0.0
        attention_probs_dropout_prob: 注意力 dropout，默认 0.0
        use_memory_efficient_attention: 是否使用高效注意力，默认 False
        use_gradient_checkpointing: 是否使用梯度检查点，默认 False
        use_rotary_position_embedding: 是否使用旋转位置编码，默认 False
        layer_norm_eps: LayerNorm epsilon，默认 1e-6
        qkv_bias: QKV 是否有偏置，默认 True
    """
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    use_memory_efficient_attention: bool = False
    use_gradient_checkpointing: bool = False
    use_rotary_position_embedding: bool = False
    layer_norm_eps: float = 1e-6
    qkv_bias: bool = True

    num_patches: int = field(init=False)

    def __post_init__(self):
        self.num_patches = (self.image_size // self.patch_size) ** 2


class PatchEmbedding(nn.Module):
    """
    图像分块嵌入层

    将输入图像分割成固定大小的 patches，并将其嵌入到隐藏维度空间。
    同时添加可学习的 [CLS] token 和位置嵌入。

    过程:
        1. 使用卷积层将图像分割成 patches
        2. 展平 patches 并转置
        3. 添加 [CLS] token
        4. 添加可学习的位置嵌入
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.num_patches = config.num_patches
        self.patch_size = config.patch_size

        self.projection = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, config.hidden_size)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(
        self,
        pixel_values: torch.Tensor,
        interpolate_pos_embedding: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        参数:
            pixel_values: 输入图像张量，形状 [B, C, H, W]
            interpolate_pos_embedding: 是否插值位置嵌入（用于不同尺寸图像）

        返回:
            embeddings: patch embeddings 包含 [CLS] token，形状 [B, N+1, D]
            pos_embedding: 位置嵌入，形状 [1, N+1, D]
        """
        batch_size = pixel_values.shape[0]

        patch_embeddings = self.projection(pixel_values)
        patch_embeddings = patch_embeddings.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, patch_embeddings], dim=1)

        if interpolate_pos_embedding and pixel_values.shape[2:] != (
            self.config.image_size, self.config.image_size
        ):
            pos_embedding = self._interpolate_pos_embedding(
                self.position_embedding,
                original_size=(self.config.image_size // self.patch_size,
                              self.config.image_size // self.patch_size),
                target_size=(pixel_values.shape[2] // self.patch_size,
                            pixel_values.shape[3] // self.patch_size),
                batch_size=batch_size
            )
        else:
            pos_embedding = self.position_embedding

        embeddings = embeddings + pos_embedding
        embeddings = self.dropout(embeddings)

        return embeddings, pos_embedding

    def _interpolate_pos_embedding(
        self,
        pos_embedding: torch.Tensor,
        original_size: Tuple[int, int],
        target_size: Tuple[int, int],
        batch_size: int
    ) -> torch.Tensor:
        """插值位置嵌入以适应不同尺寸的图像"""
        cls_pos = pos_embedding[:, :1, :]

        original_grid_size = int(math.sqrt(pos_embedding.shape[1] - 1))
        target_grid_size = target_size[0] * target_size[1]

        original_pos = pos_embedding[:, 1:, :].reshape(
            1, original_grid_size, original_grid_size, -1
        ).permute(0, 3, 1, 2)

        target_pos = F.interpolate(
            original_pos,
            size=(target_size[0], target_size[1]),
            mode='bicubic',
            align_corners=False
        ).permute(0, 2, 3, 1).reshape(1, target_grid_size, -1)

        target_pos = torch.cat([cls_pos, target_pos], dim=1)
        target_pos = target_pos.expand(batch_size, -1, -1)

        return target_pos


class RotaryPositionEmbedding(nn.Module):
    """
    旋转位置编码 (RoPE)
    用于为注意力机制提供位置信息
    """

    def __init__(self, dim: int, max_seq_len: int = 8192):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(
        self,
        seq_len: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 cos 和 sin 编码"""
        positions = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        angles = torch.outer(positions, self.inv_freq)
        cos = angles.cos()
        sin = angles.sin()
        return cos, sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """应用旋转位置编码到 Q 和 K"""
    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * k * cos) + (rotate_half(k) * sin)
    return q, k


class ViTAttention(nn.Module):
    """
    Vision Transformer 多头自注意力模块

    实现标准的多头注意力机制，支持：
    - 可选的旋转位置编码
    - 高效注意力实现
    - 梯度检查点
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        if config.use_rotary_position_embedding:
            self.rotary_emb = RotaryPositionEmbedding(
                self.attention_head_size,
                max_seq_len=config.num_patches * 4
            )

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """变换形状以便于计算注意力分数"""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(new_shape).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播

        参数:
            hidden_states: 输入隐藏状态 [B, N, D]
            attention_mask: 注意力掩码 [B, N] 或 [B, N, N]
            output_attentions: 是否输出注意力权重

        返回:
            context_layer: 上下文层 [B, N, D]
            attention_probs: 注意力权重 [B, H, N, N]（可选）
        """
        batch_size, seq_len, _ = hidden_states.shape

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        if hasattr(self, 'rotary_emb') and self.training:
            cos, sin = self.rotary_emb(seq_len, hidden_states.device)
            query_layer, key_layer = apply_rotary_pos_emb(
                query_layer, key_layer, cos.unsqueeze(0), sin.unsqueeze(0)
            )

        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2)
        ) / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attention_mask = attention_mask.expand(-1, -1, seq_len, -1)
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(1, 2).contiguous()
        context_layer = context_layer.view(batch_size, seq_len, -1)

        outputs = (context_layer,)
        if output_attentions:
            outputs = (context_layer, attention_probs)

        return outputs


class ViTMLP(nn.Module):
    """
    Vision Transformer 前馈神经网络

    由两层线性变换组成，中间有 GELU 激活函数
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class ViTBlock(nn.Module):
    """
    Vision Transformer 编码器块

    包含：
    - Layer Normalization
    - 多头自注意力
    - 残差连接
    - Layer Normalization
    - 前馈神经网络
    - 残差连接
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.attention = ViTAttention(config)
        self.mlp = ViTMLP(config)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if config.use_gradient_checkpointing and self.training:
            self.gradient_checkpointing = True
        else:
            self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播

        参数:
            hidden_states: 输入隐藏状态
            attention_mask: 注意力掩码
            output_attentions: 是否输出注意力权重

        返回:
            hidden_states: 输出隐藏状态
            attention_probs: 注意力权重（可选）
        """
        if self.gradient_checkpointing:
            def create_forward():
                def forward_fn(hidden_states, attention_mask):
                    return self._forward_impl(
                        hidden_states, attention_mask, output_attentions
                    )
                return forward_fn

            outputs = checkpoint.checkpoint(
                create_forward(),
                hidden_states,
                attention_mask,
                use_reentrant=False
            )
        else:
            outputs = self._forward_impl(
                hidden_states, attention_mask, output_attentions
            )

        return outputs

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """实际的前向传播实现"""
        attention_outputs = self.attention(
            self.norm1(hidden_states),
            attention_mask,
            output_attentions
        )
        attention_output = attention_outputs[0]

        hidden_states = hidden_states + self.dropout(attention_output)

        mlp_output = self.mlp(self.norm2(hidden_states))
        hidden_states = hidden_states + mlp_output

        outputs = (hidden_states,)
        if output_attentions:
            outputs = (hidden_states, attention_outputs[1])

        return outputs


class ViTEncoder(nn.Module):
    """
    Vision Transformer 编码器

    由多个 ViTBlock 组成的编码器
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ViTBlock(config) for _ in range(config.num_hidden_layers)])

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
            hidden_states: 输入隐藏状态
            attention_mask: 注意力掩码
            output_hidden_states: 是否输出所有层的隐藏状态
            output_attentions: 是否输出所有层的注意力权重

        返回:
            last_hidden_state: 最后一层的隐藏状态
            hidden_states: 所有层的隐藏状态（可选）
            attentions: 所有层的注意力权重（可选）
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                output_attentions
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)

        return outputs


class VisionBackbone(nn.Module):
    """
    视觉骨干网络（完整 ViT 实现）

    该类是 Vision Transformer 的完整实现，用于从图像中提取特征。

    架构：
        1. Patch Embedding：将图像分割成 patches 并嵌入
        2. Transformer Encoder：多层自注意力
        3. 输出：CLS token 或所有 patch tokens 的特征
    """

    def __init__(self, config: Optional[ViTConfig] = None, **kwargs):
        """
        初始化视觉骨干网络

        参数:
            config: ViTConfig 配置对象，如果为 None 则使用 kwargs 创建
            **kwargs: 配置参数，当 config 为 None 时使用
        """
        super().__init__()

        if config is None:
            config = ViTConfig(**kwargs)
        self.config = config

        self.patch_embedding = PatchEmbedding(config)
        self.encoder = ViTEncoder(config)
        self.pooler = nn.Linear(config.hidden_size, 1) if config.hidden_size > 0 else None

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """初始化模型权重"""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        参数:
            pixel_values: 输入图像 [B, C, H, W]
            output_hidden_states: 是否返回所有隐藏状态
            output_attentions: 是否返回注意力权重
            return_dict: 是否返回字典格式输出

        返回:
            output: 包含 'last_hidden_state', 'pooler_output', 'hidden_states', 'attentions'
        """
        embedding_output, pos_embedding = self.patch_embedding(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=None,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )

        sequence_output = encoder_outputs[0]

        if self.pooler is not None:
            pooler_output = torch.sigmoid(self.pooler(sequence_output))
            pooler_output = (pooler_output * sequence_output).sum(dim=1) / pooler_output.sum(dim=1)
        else:
            pooler_output = sequence_output[:, 0]

        if not return_dict:
            outputs = (sequence_output, pooler_output)
            if output_hidden_states:
                outputs = outputs + (encoder_outputs[1],)
            if output_attentions:
                outputs = outputs + (encoder_outputs[2] if len(encoder_outputs) > 2 else (),)
            return outputs

        return {
            'last_hidden_state': sequence_output,
            'pooler_output': pooler_output,
            'hidden_states': encoder_outputs[1] if output_hidden_states else None,
            'attentions': encoder_outputs[2] if output_attentions else None,
        }

    def get_input_embeddings(self) -> PatchEmbedding:
        """获取输入嵌入层"""
        return self.patch_embedding

    def get_output_embeddings(self) -> Optional[nn.Module]:
        """获取输出嵌入层"""
        return self.encoder

    def resize_pos_embedding(self, new_size: int):
        """调整位置嵌入大小以适应不同的图像尺寸"""
        if new_size == self.config.image_size:
            return

        old_num_patches = self.config.num_patches
        self.config.image_size = new_size
        self.config.num_patches = (new_size // self.config.patch_size) ** 2

        new_pos_embedding = nn.Parameter(
            torch.zeros(1, self.config.num_patches + 1, self.config.hidden_size)
        )
        nn.init.trunc_normal_(new_pos_embedding, std=0.02)

        if old_num_patches < self.config.num_patches:
            new_pos_embedding[:, :old_num_patches + 1] = self.patch_embedding.position_embedding
        else:
            new_pos_embedding[:, :] = self.patch_embedding.position_embedding[:, :self.config.num_patches + 1]

        self.patch_embedding.position_embedding = new_pos_embedding


class ImageTokenizer(nn.Module):
    """
    图像分词器

    将图像转换为 token 序列，支持不同的粒度：
    - patch_tokens: 标准的 patch 分割
    - semantic_tokens: 语义分割后的区域
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_codebook_tokens: int = 8192,
        hidden_size: int = 768,
        commitment_loss_weight: float = 0.25
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Sequential(*[
                ResidualBlock(64, 128) if i == 0 else ResidualBlock(128, 128)
                for _ in range(4)
            ]),
            nn.Conv2d(128, hidden_size, kernel_size=1),
        )

        self.quantize = VectorQuantizer(
            num_codebook_tokens,
            hidden_size,
            commitment_loss_weight
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_size, 128, kernel_size=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        前向传播

        返回:
            quantized: 量化后的特征
            indices: token 索引
            loss_dict: 损失字典
        """
        h = self.encoder(x)
        h = h.flatten(2).transpose(1, 2)
        quantized, indices, loss_dict = self.quantize(h)
        return quantized, indices, loss_dict


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class VectorQuantizer(nn.Module):
    """向量量化器"""

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_loss_weight: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_loss_weight = commitment_loss_weight

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        量化

        参数:
            z: 输入特征 [B, N, D]

        返回:
            quantized: 量化后的特征
            indices: 最近邻索引
            loss_dict: 包含 vq_loss 和 commitment_loss
        """
        z_flattened = z.view(-1, self.embedding_dim)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())

        indices = torch.argmin(d, dim=1)
        quantized = self.embedding(indices).view(z.shape)

        loss = F.mse_loss(quantized.detach(), z)
        loss = loss + self.commitment_loss_weight * F.mse_loss(quantize, z)

        quantized = z + (quantized - z).detach()

        return quantized, indices, {'vq_loss': loss}


class TileProcessor(nn.Module):
    """
    图像瓦片处理器

    用于处理超高分辨率图像，将图像分割成多个 tiles 分别处理
    支持动态拼接和特征融合
    """

    def __init__(
        self,
        tile_size: int = 448,
        overlap: int = 32,
        hidden_size: int = 768,
        fusion_hidden_size: int = 768
    ):
        super().__init__()
        self.tile_size = tile_size
        self.overlap = overlap
        self.hidden_size = hidden_size

        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * 4, fusion_hidden_size),
            nn.GELU(),
            nn.Linear(fusion_hidden_size, hidden_size)
        )

    def split_into_tiles(
        self,
        image: torch.Tensor,
        tile_size: Optional[int] = None
    ) -> List[torch.Tensor]:
        """
        将图像分割成 tiles

        参数:
            image: 输入图像 [B, C, H, W]
            tile_size: tile 大小，如果为 None 则使用 self.tile_size

        返回:
            tiles: tile 列表
            positions: 每个 tile 的位置信息
        """
        if tile_size is None:
            tile_size = self.tile_size

        B, C, H, W = image.shape
        stride = tile_size - self.overlap

        tiles = []
        positions = []

        for y in range(0, H - self.overlap, stride):
            for x in range(0, W - self.overlap, stride):
                tile = image[
                    :,
                    :,
                    y:min(y + tile_size, H),
                    x:min(x + tile_size, W)
                ]

                if tile.shape[2] < tile_size or tile.shape[3] < tile_size:
                    tile = F.pad(tile, (0, tile_size - tile.shape[3], 0, tile_size - tile.shape[2]))

                tiles.append(tile)
                positions.append((y, x))

        return tiles, positions

    def merge_tiles(
        self,
        tile_features: List[torch.Tensor],
        positions: List[Tuple[int, int]],
        output_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        合并 tiles 的特征

        参数:
            tile_features: 每个 tile 的特征列表
            positions: 每个 tile 的位置
            output_size: 输出尺寸 (H, W)

        返回:
            merged_features: 合并后的特征
        """
        B = tile_features[0].shape[0]
        D = tile_features[0].shape[-1]

        H_out, W_out = output_size
        grid_size = self.tile_size // 14

        feature_map = torch.zeros(
            B, H_out // grid_size, W_out // grid_size, D,
            device=tile_features[0].device,
            dtype=tile_features[0].dtype
        )

        weight_map = torch.zeros(
            B, H_out // grid_size, W_out // grid_size, 1,
            device=tile_features[0].device
        )

        for feat, (y, x) in zip(tile_features, positions):
            y_start = y // grid_size
            x_start = x // grid_size
            y_end = y_start + feat.shape[1]
            x_end = x_start + feat.shape[2]

            feature_map[:, y_start:y_end, x_start:x_end] += feat
            weight_map[:, y_start:y_end, x_start:x_end] += 1

        merged_features = feature_map / (weight_map + 1e-8)
        return merged_features.permute(0, 3, 1, 2)

    def forward(
        self,
        image: torch.Tensor,
        backbone: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        处理图像瓦片

        参数:
            image: 输入图像
            backbone: 用于提取特征的骨干网络

        返回:
            output: 包含合并后的特征和原始 tile 特征
        """
        tiles, positions = self.split_into_tiles(image)

        tile_features = []
        for tile in tiles:
            with torch.no_grad():
                feat = backbone(tile)
                tile_features.append(feat['last_hidden_state'])

        max_seq_len = max(f.shape[1] for f in tile_features)
        tile_features_padded = []
        for f in tile_features:
            if f.shape[1] < max_seq_len:
                pad = torch.zeros(
                    *f.shape[:-1], max_seq_len - f.shape[1],
                    device=f.device, dtype=f.dtype
                )
                f = torch.cat([f, pad], dim=-1)
            tile_features_padded.append(f)

        merged = torch.stack(tile_features_padded).mean(dim=0)

        return {
            'merged_features': merged,
            'tile_features': tile_features,
            'positions': positions
        }


def create_vision_backbone(
    model_name: str = 'vit-base-patch16-224',
    pretrained: bool = False,
    **kwargs
) -> VisionBackbone:
    """
    创建视觉骨干网络的工厂函数

    参数:
        model_name: 预定义模型名称
        pretrained: 是否加载预训练权重
        **kwargs: 其他配置参数

    返回:
        VisionBackbone 实例
    """
    model_configs = {
        'vit-tiny-patch16-224': ViTConfig(
            image_size=224, patch_size=16, hidden_size=192,
            num_hidden_layers=12, num_attention_heads=3, intermediate_size=768
        ),
        'vit-small-patch16-224': ViTConfig(
            image_size=224, patch_size=16, hidden_size=384,
            num_hidden_layers=12, num_attention_heads=6, intermediate_size=1536
        ),
        'vit-base-patch16-224': ViTConfig(
            image_size=224, patch_size=16, hidden_size=768,
            num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072
        ),
        'vit-large-patch16-224': ViTConfig(
            image_size=224, patch_size=16, hidden_size=1024,
            num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096
        ),
    }

    if model_name in model_configs:
        config = model_configs[model_name]
        config.__dict__.update(kwargs)
    else:
        config = ViTConfig(**kwargs)

    model = VisionBackbone(config)

    if pretrained:
        try:
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            if config.hidden_size == 768 and config.patch_size == 16:
                torchvision_model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT if pretrained else None)
                model.load_state_dict(torchvision_model.state_dict(), strict=False)
        except Exception:
            pass

    return model
