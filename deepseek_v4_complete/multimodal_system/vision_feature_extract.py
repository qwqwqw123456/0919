"""
Vision Feature Extract - 视觉特征提取模块
==========================================

该模块提供多种视觉特征提取策略，用于从图像中提取不同层次的视觉特征。

主要组件：
- VisionFeatureExtractor: 基础视觉特征提取器
- MultiScaleFeatureExtractor: 多尺度特征提取器
- HierarchicalFeatureExtractor: 层次化特征提取器
- FeaturePyramid: 特征金字塔

特点：
- 支持多尺度特征提取
- 支持层次化特征聚合
- 支持特征金字塔构建
- 支持GPU加速
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass


@dataclass
class FeatureExtractorConfig:
    """特征提取器配置"""
    backbone_channels: List[int] = None
    output_channels: int = 768
    use_fpn: bool = True
    use_pafpn: bool = False
    feature_dim: int = 256
    num_levels: int = 4

    def __post_init__(self):
        if self.backbone_channels is None:
            self.backbone_channels = [256, 512, 1024, 2048]


class ConvBottleneck(nn.Module):
    """卷积瓶颈块"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels * 4:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNetBackbone(nn.Module):
    """ResNet 骨干网络"""

    def __init__(
        self,
        in_channels: int = 3,
        layers: List[int] = [3, 4, 6, 3],
        out_indices: Tuple[int, ...] = (0, 1, 2, 3)
    ):
        super().__init__()
        self.out_indices = out_indices

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        self.layer1 = self._make_layer(64, 256, layers[0], stride=1)
        self.layer2 = self._make_layer(256, 512, layers[1], stride=2)
        self.layer3 = self._make_layer(512, 1024, layers[2], stride=2)
        self.layer4 = self._make_layer(1024, 2048, layers[3], stride=2)

        self.out_channels = [256, 512, 1024, 2048]

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int):
        layers = [ConvBottleneck(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ConvBottleneck(out_channels * 4, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []

        x = self.stem(x)

        x = self.layer1(x)
        if 0 in self.out_indices:
            outputs.append(x)

        x = self.layer2(x)
        if 1 in self.out_indices:
            outputs.append(x)

        x = self.layer3(x)
        if 2 in self.out_indices:
            outputs.append(x)

        x = self.layer4(x)
        if 3 in self.out_indices:
            outputs.append(x)

        return outputs


class FeaturePyramid(nn.Module):
    """
    特征金字塔网络 (FPN)

    将不同层级的特征进行上采样并融合，构建多尺度特征金字塔
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
        extra_convs: bool = True
    ):
        super().__init__()

        self.in_channels_list = in_channels_list
        self.out_channels = out_channels

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            lc = nn.Conv2d(in_channels, out_channels, 1)
            self.lateral_convs.append(lc)

            fc = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.fpn_convs.append(fc)

        if extra_convs:
            self.extra_convs = nn.ModuleList([
                nn.Conv2d(out_channels, out_channels, 3, 2, 1),
                nn.Conv2d(out_channels, out_channels, 3, 2, 1)
            ])
        else:
            self.extra_convs = None

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        前向传播

        参数:
            inputs: 多层级特征列表 [C2, C3, C4, C5]

        返回:
            outputs: FPN 特征列表 [P2, P3, P4, P5, P6, P7]
        """
        assert len(inputs) == len(self.in_channels_list)

        laterals = []
        for i, (x, lc) in enumerate(zip(inputs, self.lateral_convs)):
            laterals.append(lc(x))

        for i in range(len(laterals) - 1, 0, -1):
            prev_size = laterals[i - 1].shape[2:]
            curr_size = laterals[i].shape[2:]
            if prev_size != curr_size:
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i],
                    size=prev_size,
                    mode='bilinear',
                    align_corners=False
                )
            else:
                laterals[i - 1] = laterals[i - 1] + laterals[i]

        outs = []
        for i, fc in enumerate(self.fpn_convs):
            outs.append(fc(laterals[i]))

        if self.extra_convs is not None:
            outs.append(self.extra_convs[0](F.relu(outs[-1])))
            outs.append(self.extra_convs[1](F.relu(outs[-1])))

        return outs


class PathAggregationFPN(nn.Module):
    """
    路径聚合特征金字塔网络 (PAFPN)

    在 FPN 基础上增加自顶向下的路径，增强特征融合
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
        num_outs: int = 4
    ):
        super().__init__()

        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.num_outs = num_outs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            lc = nn.Conv2d(in_channels, out_channels, 1)
            fc = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.lateral_convs.append(lc)
            self.fpn_convs.append(fc)

        self.downsample_convs = nn.ModuleList()
        self.upsample_convs = nn.ModuleList()

        for _ in range(len(in_channels_list) - 1):
            self.downsample_convs.append(nn.Conv2d(out_channels, out_channels, 3, 2, 1))
            self.upsample_convs.append(nn.Conv2d(out_channels, out_channels, 1))

        self.extra_convs = None
        if num_outs > len(in_channels_list):
            self.extra_convs = nn.ModuleList([
                nn.Conv2d(out_channels, out_channels, 3, 2, 1)
                for _ in range(num_outs - len(in_channels_list))
            ])

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        前向传播

        参数:
            inputs: 多层级特征列表

        返回:
            outputs: PAFPN 特征列表
        """
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        for i in range(len(laterals) - 1, 0, -1):
            prev_size = laterals[i - 1].shape[2:]
            curr_size = laterals[i].shape[2:]
            if prev_size != curr_size:
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i],
                    size=prev_size,
                    mode='bilinear',
                    align_corners=False
                )
            else:
                laterals[i - 1] = laterals[i - 1] + laterals[i]

        for i in range(len(laterals)):
            laterals[i] = self.fpn_convs[i](laterals[i])

        for i in range(len(laterals) - 1):
            downsample = self.downsample_convs[i](laterals[i])
            upsample = F.interpolate(
                laterals[i + 1],
                size=downsample.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            laterals[i + 1] = laterals[i + 1] + self.upsample_convs[i](downsample + upsample)

        outs = laterals

        if self.extra_convs is not None:
            for extra_conv in self.extra_convs:
                outs.append(extra_conv(F.relu(outs[-1])))

        return outs


class AttentionPool2d(nn.Module):
    """注意力池化层"""

    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5
        )
        self.head_dim = embed_dim // num_heads
        self.scale = embed_dim ** -0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)

        x = x + self.positional_embedding[:, None, :].to(x.dtype)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(-1, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.reshape(-1, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.reshape(-1, self.num_heads, self.head_dim).transpose(0, 1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        x = torch.matmul(attn, v).transpose(0, 1).reshape(-1, self.head_dim * self.num_heads)
        x = self.c_proj(x)

        return x


class VisionFeatureExtractor(nn.Module):
    """
    视觉特征提取器

    封装骨干网络和特征金字塔，提供统一的特征提取接口
    """

    def __init__(
        self,
        config: Optional[FeatureExtractorConfig] = None,
        backbone_type: str = 'resnet',
        pretrained: bool = False
    ):
        super().__init__()

        if config is None:
            config = FeatureExtractorConfig()

        self.config = config

        if backbone_type == 'resnet':
            self.backbone = ResNetBackbone(
                out_indices=(0, 1, 2, 3)
            )
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

        if config.use_fpn:
            if config.use_pafpn:
                self.fpn = PathAggregationFPN(
                    in_channels_list=self.backbone.out_channels,
                    out_channels=config.feature_dim,
                    num_outs=config.num_levels
                )
            else:
                self.fpn = FeaturePyramid(
                    in_channels_list=self.backbone.out_channels,
                    out_channels=config.feature_dim
                )
        else:
            self.fpn = None

        self.projection = nn.ModuleList([
            nn.Conv2d(config.feature_dim, config.output_channels, 1)
            for _ in range(config.num_levels)
        ])

    def forward(
        self,
        x: torch.Tensor,
        return_all_levels: bool = True
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        前向传播

        参数:
            x: 输入图像 [B, C, H, W]
            return_all_levels: 是否返回所有层级的特征

        返回:
            features: 特征张量或特征列表
        """
        backbone_features = self.backbone(x)

        if self.fpn is not None:
            fpn_features = self.fpn(backbone_features)
        else:
            fpn_features = backbone_features

        projected_features = []
        for i, feat in enumerate(fpn_features[:self.config.num_levels]):
            projected_features.append(self.projection[i](feat))

        if not return_all_levels:
            return projected_features[-1]

        return projected_features

    def extract_patch_features(
        self,
        x: torch.Tensor,
        patch_size: int = 16
    ) -> torch.Tensor:
        """
        提取 patch 级别的特征

        参数:
            x: 输入图像 [B, C, H, W]
            patch_size: patch 大小

        返回:
            patch_features: [B, N, D] 其中 N = (H/p) * (W/p)
        """
        features = self.forward(x, return_all_levels=False)

        B, C, H, W = features.shape
        patch_h, patch_w = H // patch_size, W // patch_size

        features = F.adaptive_avg_pool2d(features, (patch_h, patch_w))

        features = features.flatten(2).transpose(1, 2)

        return features

    def extract_global_features(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        提取全局特征（池化后的特征）

        参数:
            x: 输入图像 [B, C, H, W]

        返回:
            global_features: [B, D]
        """
        features = self.forward(x, return_all_levels=False)

        global_features = F.adaptive_avg_pool2d(features, 1)
        global_features = global_features.flatten(1)

        return global_features


class MultiScaleFeatureExtractor(nn.Module):
    """
    多尺度特征提取器

    在多个尺度上提取特征并进行融合，提高对不同大小物体的检测能力
    """

    def __init__(
        self,
        base_extractor: VisionFeatureExtractor,
        scales: List[float] = [0.5, 0.75, 1.0, 1.25, 1.5],
        scale_factor: float = 2.0
    ):
        super().__init__()
        self.base_extractor = base_extractor
        self.scales = scales
        self.scale_factor = scale_factor

    def forward(
        self,
        x: torch.Tensor,
        max_scales: int = 3
    ) -> torch.Tensor:
        """
        多尺度特征提取

        参数:
            x: 输入图像 [B, C, H, W]
            max_scales: 最大使用的尺度数量

        返回:
            fused_features: 融合后的特征
        """
        use_scales = self.scales[:max_scales]

        features_list = []

        for scale in use_scales:
            if scale != 1.0:
                h, w = x.shape[2:]
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_x = F.interpolate(
                    x,
                    size=(new_h, new_w),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                scaled_x = x

            features = self.base_extractor(scaled_x, return_all_levels=False)

            if scale != 1.0:
                features = F.interpolate(
                    features,
                    size=x.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )

            features_list.append(features)

        fused_features = torch.stack(features_list, dim=0).mean(dim=0)

        return fused_features

    def extract_at_scale(
        self,
        x: torch.Tensor,
        scale: float
    ) -> torch.Tensor:
        """
        在指定尺度提取特征

        参数:
            x: 输入图像
            scale: 缩放因子

        返回:
            features: 特征
        """
        if scale != 1.0:
            h, w = x.shape[2:]
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_x = F.interpolate(
                x,
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            )
        else:
            scaled_x = x

        features = self.base_extractor(scaled_x, return_all_levels=False)

        if scale != 1.0:
            features = F.adaptive_max_pool2d(
                features,
                output_size=(x.shape[2] // 16, x.shape[3] // 16)
            )

        return features


class HierarchicalFeatureExtractor(nn.Module):
    """
    层次化特征提取器

    通过多层逐步聚合的方式提取特征，类似SIFT等传统特征描述子
    """

    def __init__(
        self,
        base_channels: int = 64,
        num_octaves: int = 3,
        layers_per_octave: int = 2,
        output_dim: int = 768
    ):
        super().__init__()

        self.num_octaves = num_octaves
        self.layers_per_octave = layers_per_octave

        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, base_channels, 7, 2, 3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        self.octave_layers = nn.ModuleList()

        channels = base_channels
        for octave_idx in range(num_octaves):
            octave_channels = channels * 2

            octave_convs = nn.ModuleList()
            for layer_idx in range(layers_per_octave):
                in_ch = channels if layer_idx == 0 else octave_channels
                octave_convs.append(nn.Sequential(
                    nn.Conv2d(in_ch, octave_channels, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(octave_channels),
                    nn.ReLU(inplace=True)
                ))

            self.octave_layers.append(octave_convs)
            channels = octave_channels

        self.out_conv = nn.Conv2d(channels, output_dim, 1)

        self.gaussian_kernel = self._create_gaussian_kernel(3)

    def _create_gaussian_kernel(self, size: int) -> torch.Tensor:
        """创建高斯核"""
        sigma = size / 6.0
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel_2d = g.outer(g)
        kernel_4d = kernel_2d.unsqueeze(0).unsqueeze(0) * torch.eye(3).unsqueeze(-1).unsqueeze(-1)
        return kernel_4d.squeeze(0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        层次化特征提取

        参数:
            x: 输入图像 [B, C, H, W]

        返回:
            output: 包含各层特征的字典
        """
        octave_features = []

        h, w = x.shape[2:]
        current_x = x

        for octave_idx, octave_convs in enumerate(self.octave_layers):
            octave_size = (h // (2 ** octave_idx), w // (2 ** octave_idx))

            if octave_idx > 0:
                current_x = F.avg_pool2d(current_x, 2)

            current_x = self.initial_conv(current_x) if octave_idx == 0 else current_x

            octave_feats = []
            for layer_idx, conv in enumerate(octave_convs):
                current_x = conv(current_x)
                octave_feats.append(current_x)

            octave_features.append({
                'octave': octave_idx,
                'features': octave_feats
            })

        final_features = self.out_conv(current_x)

        return {
            'final_features': final_features,
            'octave_features': octave_features,
            'pyramid': self._build_pyramid(octave_features)
        }

    def _build_pyramid(self, octave_features: List[Dict]) -> torch.Tensor:
        """构建特征金字塔"""
        pyramid_levels = []

        for octave in octave_features:
            for layer_feat in octave['features']:
                pyramid_levels.append(layer_feat)

        return torch.cat([F.adaptive_avg_pool2d(f, 1) for f in pyramid_levels], dim=1)


class SpatialAttention(nn.Module):
    """空间注意力模块"""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_map = self.conv(x)
        return x * attention_map


class ChannelAttention(nn.Module):
    """通道注意力模块"""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class CBAM(nn.Module):
    """
    卷积块注意力模块 (CBAM)

    结合空间注意力和通道注意力增强特征表达
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class AttentionEnhancedExtractor(nn.Module):
    """
    注意力增强的特征提取器

    在基础特征提取器上增加注意力机制，提升特征质量
    """

    def __init__(
        self,
        base_extractor: VisionFeatureExtractor,
        use_cbam: bool = True
    ):
        super().__init__()
        self.base_extractor = base_extractor
        self.use_cbam = use_cbam

        if use_cbam:
            feature_dim = base_extractor.config.feature_dim
            self.attention_modules = nn.ModuleList([
                CBAM(feature_dim) for _ in range(base_extractor.config.num_levels)
            ])

    def forward(
        self,
        x: torch.Tensor,
        return_all_levels: bool = True
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """前向传播"""
        features = self.base_extractor(x, return_all_levels=True)

        if self.use_cbam and len(features) == len(self.attention_modules):
            for i in range(len(features)):
                features[i] = self.attention_modules[i](features[i])

        if not return_all_levels:
            return features[-1]

        return features


class SemanticFeatureExtractor(nn.Module):
    """
    语义特征提取器

    提取包含语义信息的特征，用于图像描述等任务
    """

    def __init__(
        self,
        vision_extractor: VisionFeatureExtractor,
        semantic_dim: int = 512
    ):
        super().__init__()
        self.vision_extractor = vision_extractor

        self.semantic_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(vision_extractor.config.output_channels, semantic_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(semantic_dim * 2, semantic_dim)
        )

        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(
                vision_extractor.config.feature_dim,
                semantic_dim,
                1
            ),
            nn.BatchNorm2d(semantic_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        提取语义特征

        返回:
            包含全局语义特征和空间语义特征的字典
        """
        spatial_features = self.vision_extractor(x, return_all_levels=False)

        global_semantic = self.semantic_proj(spatial_features)

        spatial_semantic = self.spatial_encoder(spatial_features)

        return {
            'global_features': global_semantic,
            'spatial_features': spatial_semantic,
            'combined_features': global_semantic.unsqueeze(-1).unsqueeze(-1) + F.interpolate(
                spatial_semantic,
                size=(1, 1),
                mode='bilinear',
                align_corners=False
            ).squeeze(-1).squeeze(-1)
        }


def create_feature_extractor(
    extractor_type: str = 'standard',
    **kwargs
) -> VisionFeatureExtractor:
    """
    创建特征提取器的工厂函数

    参数:
        extractor_type: 提取器类型 ('standard', 'multiscale', 'hierarchical')
        **kwargs: 其他参数

    返回:
        VisionFeatureExtractor 实例
    """
    if extractor_type == 'standard':
        config = FeatureExtractorConfig(**kwargs)
        return VisionFeatureExtractor(config)

    elif extractor_type == 'multiscale':
        base_config = FeatureExtractorConfig(**kwargs)
        base_extractor = VisionFeatureExtractor(base_config)
        scales = kwargs.get('scales', [0.5, 0.75, 1.0, 1.25, 1.5])
        return MultiScaleFeatureExtractor(base_extractor, scales=scales)

    elif extractor_type == 'hierarchical':
        base_channels = kwargs.get('base_channels', 64)
        output_dim = kwargs.get('output_dim', 768)
        return HierarchicalFeatureExtractor(
            base_channels=base_channels,
            output_dim=output_dim
        )

    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}")
