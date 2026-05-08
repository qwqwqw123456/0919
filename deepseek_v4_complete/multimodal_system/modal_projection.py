"""
Modal Projection - 多模态投影层模块
===================================

该模块实现多种多模态投影策略，用于将不同模态的特征映射到统一的空间。

主要组件：
- ProjectionConfig: 投影配置类
- ModalProjection: 基础多模态投影模块
- MLPProjection: MLP投影
- CrossAttentionProjection: 交叉注意力投影
- LinearProjection: 线性投影
- ResidualProjection: 残差投影
- GatedProjection: 门控投影

特点：
- 支持多种投影方式
- 支持可学习的投影参数
- 支持模态特定的投影
- 支持残差连接
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass


@dataclass
class ProjectionConfig:
    """
    投影配置

    参数:
        input_dim: 输入维度
        output_dim: 输出维度
        hidden_dim: 隐藏层维度
        projection_type: 投影类型 ('linear', 'mlp', 'cross_attention', 'residual', 'gated')
        num_layers: 投影层数
        dropout_prob: Dropout概率
        layer_norm_eps: LayerNorm epsilon
        activation: 激活函数类型
        use_bias: 是否使用偏置
    """
    input_dim: int = 768
    output_dim: int = 768
    hidden_dim: int = 1536
    projection_type: str = "mlp"
    num_layers: int = 2
    dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-6
    activation: str = "gelu"
    use_bias: bool = True


class LinearProjection(nn.Module):
    """
    线性投影层

    最简单的投影方式，使用单个线性变换
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        use_bias: bool = True
    ):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=use_bias)
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.trunc_normal_(self.projection.weight, std=0.02)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入张量 [B, ..., D_in]

        返回:
            投影后的张量 [B, ..., D_out]
        """
        return self.projection(x)


class MLPProjection(nn.Module):
    """
    MLP投影层

    使用多层感知机进行投影
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if hidden_dim is None:
            hidden_dim = (input_dim + output_dim) // 2

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        activation_fn = self._get_activation(activation)

        layers = []
        in_dim = input_dim

        for i in range(num_layers):
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))

            if i < num_layers - 1:
                layers.append(nn.LayerNorm(out_dim, eps=1e-6))
                layers.append(activation_fn)
                layers.append(nn.Dropout(dropout))

            in_dim = out_dim

        self.projection = nn.Sequential(*layers)
        self._init_weights()

    def _get_activation(self, activation: str):
        """获取激活函数"""
        activations = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'silu': nn.SiLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid
        }
        return activations.get(activation, nn.GELU)()

    def _init_weights(self):
        """初始化权重"""
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入张量 [B, ..., D_in]

        返回:
            投影后的张量 [B, ..., D_out]
        """
        return self.projection(x)


class CrossAttentionProjection(nn.Module):
    """
    交叉注意力投影层

    使用交叉注意力机制进行投影
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        self.query_proj = nn.Linear(input_dim, output_dim)
        self.key_proj = nn.Linear(input_dim, output_dim)
        self.value_proj = nn.Linear(input_dim, output_dim)
        self.output_proj = nn.Linear(output_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入张量 [B, L, D_in]
            context: 上下文张量 [B, L_c, D_in]，如果为None则使用x

        返回:
            投影后的张量 [B, L, D_out]
        """
        if context is None:
            context = x

        B, L, _ = x.shape
        L_c = context.shape[1]

        q = self.query_proj(x)
        k = self.key_proj(context)
        v = self.value_proj(context)

        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L_c, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L_c, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, v)
        context = context.transpose(1, 2).contiguous().view(B, L, -1)
        output = self.output_proj(context)
        output = self.norm(output)

        return output


class ResidualProjection(nn.Module):
    """
    残差投影层

    使用残差连接进行投影
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if input_dim != output_dim:
            self.input_proj = nn.Linear(input_dim, output_dim)
        else:
            self.input_proj = None

        self.projection = MLPProjection(
            output_dim,
            output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入张量 [B, ..., D_in]

        返回:
            投影后的张量 [B, ..., D_out]
        """
        if self.input_proj is not None:
            x = self.input_proj(x)

        output = x + self.projection(x)
        return output


class GatedProjection(nn.Module):
    """
    门控投影层

    使用门控机制控制信息流
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if hidden_dim is None:
            hidden_dim = max(input_dim, output_dim)

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入张量 [B, ..., D_in]

        返回:
            投影后的张量 [B, ..., D_out]
        """
        projected = self.projection(x)
        gate_values = self.gate(x)
        gated = projected * gate_values
        output = self.output_proj(gated)
        return output


class MultiplicativeProjection(nn.Module):
    """
    乘法投影层

    使用乘法交互进行投影
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int = 4
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            nn.Linear(input_dim, output_dim)
            for _ in range(num_experts)
        ])

        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入张量 [B, ..., D_in]

        返回:
            投影后的张量 [B, ..., D_out]
        """
        B = x.shape[0]

        gate_values = self.gate(x)

        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))

        expert_outputs = torch.stack(expert_outputs, dim=0)

        gate_expanded = gate_values.view(B, 1, self.num_experts, 1)
        output = (gate_expanded * expert_outputs.permute(2, 0, 1, 3)).sum(dim=2)

        return output


class BiDirectionalProjection(nn.Module):
    """
    双向投影层

    同时进行前向和逆向投影
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.forward_proj = MLPProjection(
            input_dim,
            output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        self.backward_proj = MLPProjection(
            input_dim,
            output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入张量 [B, L, D_in]

        返回:
            投影后的张量 [B, L, D_out]
        """
        forward_out = self.forward_proj(x)

        reversed_x = torch.flip(x, dims=[1])
        backward_out = self.backward_proj(reversed_x)
        backward_out = torch.flip(backward_out, dims=[1])

        combined = torch.cat([forward_out, backward_out], dim=-1)
        output = self.fusion(combined)

        return output


class ModalProjection(nn.Module):
    """
    多模态投影主模块

    综合使用多种投影策略
    """

    def __init__(self, config: Optional[ProjectionConfig] = None, **kwargs):
        """
        初始化多模态投影模块

        参数:
            config: 投影配置
            **kwargs: 配置参数（当config为None时使用）
        """
        super().__init__()

        if config is None:
            config = ProjectionConfig(**kwargs)

        self.config = config
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim

        if config.projection_type == 'linear':
            self.projection = LinearProjection(
                config.input_dim,
                config.output_dim,
                config.use_bias
            )
        elif config.projection_type == 'mlp':
            self.projection = MLPProjection(
                config.input_dim,
                config.output_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                dropout=config.dropout_prob,
                activation=config.activation
            )
        elif config.projection_type == 'cross_attention':
            self.projection = CrossAttentionProjection(
                config.input_dim,
                config.output_dim,
                dropout=config.dropout_prob
            )
        elif config.projection_type == 'residual':
            self.projection = ResidualProjection(
                config.input_dim,
                config.output_dim,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout_prob
            )
        elif config.projection_type == 'gated':
            self.projection = GatedProjection(
                config.input_dim,
                config.output_dim,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout_prob
            )
        else:
            self.projection = MLPProjection(
                config.input_dim,
                config.output_dim,
                hidden_dim=config.hidden_dim
            )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入张量
            context: 上下文张量（用于交叉注意力投影）

        返回:
            投影后的张量
        """
        if isinstance(self.projection, CrossAttentionProjection):
            return self.projection(x, context)
        else:
            return self.projection(x)

    def get_output_dim(self) -> int:
        """获取输出维度"""
        return self.output_dim

    def get_input_dim(self) -> int:
        """获取输入维度"""
        return self.input_dim


class MultimodalProjector(nn.Module):
    """
    多模态投影器

    包含多个模态特定的投影器
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        output_dim: int,
        projection_type: str = "mlp"
    ):
        super().__init__()
        self.modality_dims = modality_dims
        self.output_dim = output_dim

        self.projectors = nn.ModuleDict({
            modality: ModalProjection(
                ProjectionConfig(
                    input_dim=dim,
                    output_dim=output_dim,
                    projection_type=projection_type
                )
            )
            for modality, dim in modality_dims.items()
        })

        self.default_projection = ModalProjection(
            ProjectionConfig(
                input_dim=list(modality_dims.values())[0],
                output_dim=output_dim,
                projection_type=projection_type
            )
        )

    def forward(
        self,
        features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        参数:
            features: 各模态的特征字典

        返回:
            投影后的特征字典
        """
        projected = {}

        for modality, feat in features.items():
            if modality in self.projectors:
                projected[modality] = self.projectors[modality](feat)
            else:
                projected[modality] = self.default_projection(feat)

        return projected

    def project(
        self,
        modality: str,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        投影指定模态的特征

        参数:
            modality: 模态名称
            features: 特征张量

        返回:
            投影后的特征
        """
        if modality in self.projectors:
            return self.projectors[modality](features)
        else:
            return self.default_projection(features)


class AdaptiveProjection(nn.Module):
    """
    自适应投影层

    根据输入内容自适应选择投影方式
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_strategies: int = 4,
        hidden_dim: Optional[int] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_strategies = num_strategies

        if hidden_dim is None:
            hidden_dim = (input_dim + output_dim) // 2

        self.strategy_projection = nn.ModuleList([
            MLPProjection(input_dim, output_dim, hidden_dim=hidden_dim)
            for _ in range(num_strategies)
        ])

        self.strategy_gate = nn.Sequential(
            nn.Linear(input_dim, num_strategies),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入张量 [B, ..., D_in]

        返回:
            投影后的张量 [B, ..., D_out]
        """
        gate_values = self.strategy_gate(x)

        strategy_outputs = []
        for strategy_proj in self.strategy_projection:
            strategy_outputs.append(strategy_proj(x))

        strategy_outputs = torch.stack(strategy_outputs, dim=0)

        B = x.shape[0]
        gate_expanded = gate_values.view(B, 1, self.num_strategies, 1)
        output = (gate_expanded * strategy_outputs.permute(1, 2, 0, 3)).sum(dim=2)

        return output


class ComposedProjection(nn.Module):
    """
    组合投影层

    将多个投影层组合在一起
    """

    def __init__(
        self,
        projections: List[nn.Module]
    ):
        super().__init__()
        self.projections = nn.ModuleList(projections)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入张量

        返回:
            投影后的张量
        """
        output = x
        for projection in self.projections:
            output = projection(output)
        return output


def create_projection(
    projection_type: str = "mlp",
    input_dim: int = 768,
    output_dim: int = 768,
    **kwargs
) -> ModalProjection:
    """
    创建投影模块的工厂函数

    参数:
        projection_type: 投影类型
        input_dim: 输入维度
        output_dim: 输出维度
        **kwargs: 其他配置参数

    返回:
        ModalProjection 实例
    """
    config = ProjectionConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        projection_type=projection_type,
        **kwargs
    )

    return ModalProjection(config)


def create_multimodal_projector(
    modality_dims: Dict[str, int],
    output_dim: int,
    projection_type: str = "mlp"
) -> MultimodalProjector:
    """
    创建多模态投影器的工厂函数

    参数:
        modality_dims: 各模态的输入维度
        output_dim: 输出维度
        projection_type: 投影类型

    返回:
        MultimodalProjector 实例
    """
    return MultimodalProjector(
        modality_dims=modality_dims,
        output_dim=output_dim,
        projection_type=projection_type
    )
