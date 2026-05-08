"""
Cross Modal Fusion - 跨模态融合模块
==================================

该模块实现多种跨模态融合策略，用于将不同模态（文本、图像、音频等）的特征进行有效融合。

主要组件：
- CrossModalConfig: 跨模态融合配置
- CrossModalFusion: 基础跨模态融合模块
- GatedFusion: 门控融合
- AttentionFusion: 注意力融合
- BilinearFusion: 双线性融合
- ConcatenationFusion: 拼接融合
- HierarchicalFusion: 层次化融合

特点：
- 支持多种融合策略
- 支持可学习的融合权重
- 支持跨模态注意力机制
- 支持多模态交互
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class CrossModalConfig:
    """
    跨模态融合配置

    参数:
        hidden_size: 隐藏层维度
        num_attention_heads: 注意力头数
        fusion_type: 融合类型 ('gated', 'attention', 'bilinear', 'concat', 'hierarchical')
        dropout_prob: Dropout概率
        layer_norm_eps: LayerNorm epsilon
        use_cross_attention: 是否使用交叉注意力
        use_modality_embedding: 是否使用模态嵌入
        num_modalities: 模态数量
        fusion_layers: 融合层数
        fusion_strategy: 融合策略 ('early', 'late', 'hierarchical')
    """
    hidden_size: int = 768
    num_attention_heads: int = 12
    fusion_type: str = "gated"
    dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-6
    use_cross_attention: bool = True
    use_modality_embedding: bool = True
    num_modalities: int = 2
    fusion_layers: int = 2
    fusion_strategy: str = "early"


class ModalityEmbedding(nn.Module):
    """
    模态嵌入

    为不同模态添加可学习的嵌入标识
    """

    def __init__(self, hidden_size: int, num_modalities: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(1, hidden_size) for _ in range(num_modalities)
        ])
        for emb in self.embeddings:
            nn.init.trunc_normal_(emb.weight, std=0.02)

    def forward(self, x: torch.Tensor, modality_id: int) -> torch.Tensor:
        """
        添加模态嵌入

        参数:
            x: 输入张量
            modality_id: 模态ID

        返回:
            带模态嵌入的张量
        """
        modality_token = torch.zeros(x.shape[0], 1, device=x.device, dtype=torch.long)
        modality_emb = self.embeddings[modality_id](modality_token)
        return x + modality_emb


class GatedFusion(nn.Module):
    """
    门控融合模块

    通过门控机制动态控制不同模态信息的重要性
    """

    def __init__(self, hidden_size: int, num_gates: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size

        self.gate_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Sigmoid()
            )
            for _ in range(num_gates)
        ])

        self.transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            for _ in range(num_gates)
        ])

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor
    ) -> torch.Tensor:
        """
        门控融合

        参数:
            text_features: 文本特征 [B, L_t, D]
            image_features: 图像特征 [B, L_i, D]

        返回:
            fused: 融合后的特征 [B, L, D]
        """
        text_len = text_features.shape[1]
        image_len = image_features.shape[1]

        if text_len == image_len:
            combined = torch.cat([text_features, image_features], dim=-1)
        else:
            if text_len > image_len:
                image_expanded = F.interpolate(
                    image_features.transpose(1, 2),
                    size=text_len,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
                combined = torch.cat([text_features, image_expanded], dim=-1)
            else:
                text_expanded = F.interpolate(
                    text_features.transpose(1, 2),
                    size=image_len,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
                combined = torch.cat([text_expanded, image_features], dim=-1)

        hidden = combined
        for gate_fn, transform in zip(self.gate_networks, self.transforms):
            gate = gate_fn(hidden)
            hidden = transform(hidden) * gate + hidden * (1 - gate)

        output_len = min(text_len, image_len)
        output = hidden[:, :output_len]

        return output


class CrossAttention(nn.Module):
    """
    交叉注意力模块

    支持 Query-Key-Value 的跨模态注意力计算
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        交叉注意力前向传播

        参数:
            query: 查询张量 [B, L_q, D]
            key: 键张量 [B, L_k, D]
            value: 值张量 [B, L_v, D]
            attention_mask: 注意力掩码

        返回:
            output: 输出张量
            attention_weights: 注意力权重（可选）
        """
        B, L_q, _ = query.shape
        L_k = key.shape[1]

        q = self.q_proj(query).view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, v)
        context = context.transpose(1, 2).contiguous().view(B, L_q, -1)
        output = self.o_proj(context)

        return output, attention_probs


class AttentionFusion(nn.Module):
    """
    注意力融合模块

    使用交叉注意力机制进行跨模态融合
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.text_to_image_attn = nn.ModuleList([
            CrossAttention(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.image_to_text_attn = nn.ModuleList([
            CrossAttention(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.text_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])

        self.image_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])

        self.norm1_text = nn.LayerNorm(hidden_size)
        self.norm2_text = nn.LayerNorm(hidden_size)
        self.norm1_image = nn.LayerNorm(hidden_size)
        self.norm2_image = nn.LayerNorm(hidden_size)

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        注意力融合前向传播

        参数:
            text_features: 文本特征 [B, L_t, D]
            image_features: 图像特征 [B, L_i, D]
            attention_mask: 注意力掩码

        返回:
            text_output: 融合后的文本特征
            image_output: 融合后的图像特征
        """
        text_hidden = text_features
        image_hidden = image_features

        for i in range(self.num_layers):
            text_attn_out, _ = self.text_to_image_attn[i](
                self.norm1_text(text_hidden),
                self.norm1_image(image_hidden),
                self.norm1_image(image_hidden),
                attention_mask
            )
            text_hidden = text_hidden + text_attn_out
            text_hidden = text_hidden + self.text_ffn[i](self.norm2_text(text_hidden))

            image_attn_out, _ = self.image_to_text_attn[i](
                self.norm1_image(image_hidden),
                self.norm1_text(text_hidden),
                self.norm1_text(text_hidden),
                attention_mask
            )
            image_hidden = image_hidden + image_attn_out
            image_hidden = image_hidden + self.image_ffn[i](self.norm2_image(image_hidden))

        return text_hidden, image_hidden


class BilinearFusion(nn.Module):
    """
    双线性融合模块

    使用双线性变换进行跨模态特征交互
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size

        self.bilinear = nn.Bilinear(hidden_size, hidden_size, hidden_size)

        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor
    ) -> torch.Tensor:
        """
        双线性融合

        参数:
            text_features: 文本特征 [B, L_t, D]
            image_features: 图像特征 [B, L_i, D]

        返回:
            fused: 融合后的特征 [B, L, D]
        """
        seq_len = min(text_features.shape[1], image_features.shape[1])

        text_seq = text_features[:, :seq_len]
        image_seq = image_features[:, :seq_len]

        bilinear_out = self.bilinear(text_seq, image_seq)

        concat_features = torch.cat([text_seq, image_seq], dim=-1)
        gate_values = self.gate(concat_features)

        fused = bilinear_out * gate_values
        fused = self.projection(fused)

        return fused


class ConcatenationFusion(nn.Module):
    """
    拼接融合模块

    将不同模态特征拼接后通过 MLP 进行融合
    """

    def __init__(
        self,
        hidden_size: int,
        fusion_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        if fusion_dim is None:
            fusion_dim = hidden_size

        self.projection = nn.Sequential(
            nn.Linear(hidden_size * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, hidden_size),
            nn.Dropout(dropout)
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor
    ) -> torch.Tensor:
        """
        拼接融合

        参数:
            text_features: 文本特征 [B, L_t, D]
            image_features: 图像特征 [B, L_i, D]

        返回:
            fused: 融合后的特征 [B, L, D]
        """
        seq_len = min(text_features.shape[1], image_features.shape[1])

        text_seq = text_features[:, :seq_len]
        image_seq = image_features[:, :seq_len]

        concat = torch.cat([text_seq, image_seq], dim=-1)

        projected = self.projection(concat)
        gate = self.gate(concat)

        fused = projected * gate + text_seq * (1 - gate)

        return fused


class HierarchicalFusion(nn.Module):
    """
    层次化融合模块

    在多个层次上进行跨模态融合
    """

    def __init__(
        self,
        hidden_size: int,
        num_levels: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_levels = num_levels

        self.level_fusions = nn.ModuleList()
        for level in range(num_levels):
            level_hidden = hidden_size // (2 ** level) if level > 0 else hidden_size

            self.level_fusions.append(nn.ModuleDict({
                'cross_attn': CrossAttention(level_hidden, num_heads, dropout),
                'ffn': nn.Sequential(
                    nn.Linear(level_hidden, level_hidden * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(level_hidden * 4, level_hidden),
                    nn.Dropout(dropout)
                ),
                'norm1': nn.LayerNorm(level_hidden),
                'norm2': nn.LayerNorm(level_hidden)
            }))

        self.pool = nn.AdaptiveAvgPool1d(1)

    def _create_hierarchy(self, features: torch.Tensor) -> List[torch.Tensor]:
        """创建特征层次"""
        hierarchies = [features]

        current = features
        for level in range(self.num_levels - 1):
            pooled = F.adaptive_avg_pool1d(
                current.transpose(1, 2),
                output_size=max(1, current.shape[1] // 2)
            ).transpose(1, 2)
            hierarchies.append(pooled)
            current = pooled

        return hierarchies

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor
    ) -> torch.Tensor:
        """
        层次化融合

        参数:
            text_features: 文本特征 [B, L_t, D]
            image_features: 图像特征 [B, L_i, D]

        返回:
            fused: 融合后的特征 [B, L, D]
        """
        text_hierarchy = self._create_hierarchy(text_features)
        image_hierarchy = self._create_hierarchy(image_features)

        fused_features = []

        for level, level_modules in enumerate(self.level_fusions):
            if level >= len(text_hierarchy) or level >= len(image_hierarchy):
                break

            t_feat = text_hierarchy[level]
            i_feat = image_hierarchy[level]

            t2i_out, _ = level_modules['cross_attn'](
                level_modules['norm1'](t_feat),
                level_modules['norm1'](i_feat),
                level_modules['norm1'](i_feat)
            )

            t_feat = t_feat + t2i_out
            t_feat = t_feat + level_modules['ffn'](level_modules['norm2'](t_feat))

            fused_features.append(t_feat)

        if fused_features:
            fused = fused_features[0]
            for feat in fused_features[1:]:
                fused = fused + F.interpolate(
                    feat.transpose(1, 2),
                    size=fused.shape[1],
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            return fused
        else:
            return text_features


class CrossModalFusion(nn.Module):
    """
    跨模态融合主模块

    综合使用多种融合策略进行跨模态特征融合
    """

    def __init__(self, config: Optional[CrossModalConfig] = None, **kwargs):
        """
        初始化跨模态融合模块

        参数:
            config: 跨模态融合配置
            **kwargs: 配置参数（当config为None时使用）
        """
        super().__init__()

        if config is None:
            config = CrossModalConfig(**kwargs)

        self.config = config
        self.hidden_size = config.hidden_size

        if config.use_modality_embedding:
            self.modality_embedding = ModalityEmbedding(
                config.hidden_size,
                config.num_modalities
            )
        else:
            self.modality_embedding = None

        if config.fusion_type == 'gated':
            self.fusion = GatedFusion(
                config.hidden_size,
                num_gates=config.fusion_layers
            )
        elif config.fusion_type == 'attention':
            self.fusion = AttentionFusion(
                config.hidden_size,
                num_heads=config.num_attention_heads,
                num_layers=config.fusion_layers
            )
        elif config.fusion_type == 'bilinear':
            self.fusion = BilinearFusion(config.hidden_size)
        elif config.fusion_type == 'concat':
            self.fusion = ConcatenationFusion(config.hidden_size)
        elif config.fusion_type == 'hierarchical':
            self.fusion = HierarchicalFusion(
                config.hidden_size,
                num_levels=config.fusion_layers
            )
        else:
            self.fusion = GatedFusion(config.hidden_size)

        self.output_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        text_modality_id: int = 0,
        image_modality_id: int = 1,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        跨模态融合前向传播

        参数:
            text_features: 文本特征 [B, L_t, D]
            image_features: 图像特征 [B, L_i, D]
            text_modality_id: 文本模态ID
            image_modality_id: 图像模态ID
            return_attention: 是否返回注意力权重

        返回:
            fused_features: 融合后的特征
            attention_weights: 注意力权重（可选）
        """
        if self.modality_embedding is not None:
            text_features = self.modality_embedding(text_features, text_modality_id)
            image_features = self.modality_embedding(image_features, image_modality_id)

        if self.config.fusion_strategy == 'early':
            fused = self.fusion(text_features, image_features)
        elif self.config.fusion_strategy == 'late':
            text_pooled = text_features.mean(dim=1, keepdim=True)
            image_pooled = image_features.mean(dim=1, keepdim=True)
            fused = torch.cat([text_pooled, image_pooled], dim=1)
            fused = fused.mean(dim=1, keepdim=True).expand_as(text_features)
        else:
            fused = self.fusion(text_features, image_features)

        fused = self.output_norm(fused)

        outputs = {'fused_features': fused}

        if return_attention and hasattr(self.fusion, 'get_attention_weights'):
            outputs['attention_weights'] = self.fusion.get_attention_weights()

        return outputs

    def fuse_multiple_modalities(
        self,
        modality_features: List[torch.Tensor],
        modality_ids: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        融合多个模态的特征

        参数:
            modality_features: 多个模态的特征列表
            modality_ids: 模态ID列表

        返回:
            fused: 融合后的特征
        """
        if modality_ids is None:
            modality_ids = list(range(len(modality_features)))

        if len(modality_features) == 2:
            return self.forward(
                modality_features[0],
                modality_features[1],
                modality_ids[0],
                modality_ids[1]
            )['fused_features']

        fused = modality_features[0]
        for i in range(1, len(modality_features)):
            fused = self.forward(
                fused,
                modality_features[i],
                modality_ids[0] if i == 1 else 0,
                modality_ids[i]
            )['fused_features']

        return fused


class MultiHeadCrossModalAttention(nn.Module):
    """
    多头跨模态注意力

    在多个表示子空间中进行跨模态注意力计算
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        num_modalities: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_modalities = num_modalities
        self.head_dim = hidden_size // num_heads

        self.modality_projections = nn.ModuleDict({
            str(i): nn.Linear(hidden_size, hidden_size)
            for i in range(num_modalities)
        })

        self.q_proj = nn.ModuleDict({
            f'{i}_{j}': nn.Linear(hidden_size, hidden_size)
            for i in range(num_modalities)
            for j in range(num_modalities)
        })

        self.k_proj = nn.ModuleDict({
            f'{i}_{j}': nn.Linear(hidden_size, hidden_size)
            for i in range(num_modalities)
            for j in range(num_modalities)
        })

        self.v_proj = nn.ModuleDict({
            f'{i}_{j}': nn.Linear(hidden_size, hidden_size)
            for i in range(num_modalities)
            for j in range(num_modalities)
        })

        self.o_proj = nn.ModuleDict({
            str(i): nn.Linear(hidden_size, hidden_size)
            for i in range(num_modalities)
        })

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        modality_features: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        多头跨模态注意力前向传播

        参数:
            modality_features: 各模态特征列表
            attention_mask: 注意力掩码

        返回:
            outputs: 各模态的输出特征列表
        """
        outputs = []

        for target_modality in range(self.num_modalities):
            target_feat = modality_features[target_modality]
            B, L_t, _ = target_feat.shape

            q = self.q_proj[f'{target_modality}_{target_modality}'](target_feat)
            q = q.view(B, L_t, self.num_heads, self.head_dim).transpose(1, 2)

            k_list = []
            v_list = []

            for source_modality in range(self.num_modalities):
                source_feat = modality_features[source_modality]
                B_s, L_s, _ = source_feat.shape

                k = self.k_proj[f'{target_modality}_{source_modality}'](source_feat)
                v = self.v_proj[f'{target_modality}_{source_modality}'](source_feat)

                k = k.view(B_s, L_s, self.num_heads, self.head_dim).transpose(1, 2)
                v = v.view(B_s, L_s, self.num_heads, self.head_dim).transpose(1, 2)

                k_list.append(k)
                v_list.append(v)

            k_concat = torch.cat(k_list, dim=2)
            v_concat = torch.cat(v_list, dim=2)

            scale = 1.0 / math.sqrt(self.head_dim)
            attention_scores = torch.matmul(q, k_concat.transpose(-2, -1)) * scale

            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)

            context = torch.matmul(attention_probs, v_concat)
            context = context.transpose(1, 2).contiguous().view(B, L_t, -1)

            output = self.o_proj[str(target_modality)](context)
            outputs.append(output)

        return outputs


class TensorFusionNetwork(nn.Module):
    """
    张量融合网络

    使用外积操作进行高阶模态交互
    """

    def __init__(
        self,
        hidden_size: int,
        output_size: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        if output_size is None:
            output_size = hidden_size

        self.output_size = output_size
        self.hidden_size = hidden_size

        self.fusion_bias = nn.Parameter(torch.zeros(output_size))

        self.fc1 = nn.Linear(hidden_size * hidden_size, output_size)
        self.fc2 = nn.Linear(output_size, output_size)

        self.norm = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor
    ) -> torch.Tensor:
        """
        张量融合

        参数:
            text_features: 文本特征 [B, D]
            image_features: 图像特征 [B, D]

        返回:
            fused: 融合后的特征 [B, D]
        """
        text_expanded = text_features.unsqueeze(-1)
        image_expanded = image_features.unsqueeze(-2)

        tensor_product = torch.matmul(text_expanded, image_expanded)

        fused = tensor_product.view(text_features.shape[0], -1)

        fused = self.fc1(fused)
        fused = F.relu(fused)
        fused = self.dropout(fused)
        fused = self.fc2(fused)

        fused = fused + self.fusion_bias
        fused = self.norm(fused)

        return fused


class FactorizedTensorFusion(nn.Module):
    """
    因式分解张量融合

    使用低秩分解近似高阶张量融合，降低计算复杂度
    """

    def __init__(
        self,
        hidden_size: int,
        rank: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank

        self.text_factor = nn.Linear(hidden_size, rank, bias=False)
        self.image_factor = nn.Linear(hidden_size, rank, bias=False)

        self.fusion = nn.Linear(rank * rank, hidden_size)

        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor
    ) -> torch.Tensor:
        """
        因式分解张量融合

        参数:
            text_features: 文本特征 [B, D]
            image_features: 图像特征 [B, D]

        返回:
            fused: 融合后的特征 [B, D]
        """
        text_factors = self.text_factor(text_features)
        image_factors = self.image_factor(image_features)

        text_expanded = text_factors.unsqueeze(-1)
        image_expanded = image_factors.unsqueeze(-2)

        tensor_product = torch.matmul(text_expanded, image_expanded)

        fused = tensor_product.view(text_features.shape[0], -1)
        fused = self.fusion(fused)
        fused = self.dropout(fused)
        fused = self.norm(fused)

        return fused


class LowRankModalFusion(nn.Module):
    """
    低秩模态融合

    使用低秩矩阵分解进行高效的跨模态融合
    """

    def __init__(
        self,
        hidden_size: int,
        rank: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank

        self.text_projection = nn.Linear(hidden_size, rank)
        self.image_projection = nn.Linear(hidden_size, rank)

        self.fusion_weights = nn.Parameter(torch.randn(rank, hidden_size))

        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.fusion_weights)

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor
    ) -> torch.Tensor:
        """
        低秩模态融合

        参数:
            text_features: 文本特征 [B, D]
            image_features: 图像特征 [B, D]

        返回:
            fused: 融合后的特征 [B, D]
        """
        text_proj = self.text_projection(text_features)
        image_proj = self.image_projection(image_features)

        interaction = text_proj * image_proj

        fused = torch.matmul(interaction, self.fusion_weights)

        fused = self.dropout(fused)
        fused = self.norm(fused)

        return fused


def create_cross_modal_fusion(
    fusion_type: str = "gated",
    hidden_size: int = 768,
    **kwargs
) -> CrossModalFusion:
    """
    创建跨模态融合模块的工厂函数

    参数:
        fusion_type: 融合类型
        hidden_size: 隐藏层维度
        **kwargs: 其他配置参数

    返回:
        CrossModalFusion 实例
    """
    config = CrossModalConfig(
        hidden_size=hidden_size,
        fusion_type=fusion_type,
        **kwargs
    )

    return CrossModalFusion(config)
