"""
DeepSeek V4 多模态系统
======================

该模块提供完整的多模态处理能力，包括：
- 视觉骨干网络 (Vision Backbone)
- 视觉特征提取 (Vision Feature Extraction)
- 视觉缓存预计算 (Vision Cache Precomputation)
- 文本嵌入编码 (Text Embedding Encoding)
- 跨模态融合 (Cross-Modal Fusion)
- 多模态投影 (Modal Projection)
- 多模态推理 (Multi-Modal Inference)
- 统一编码器 (Unified Encoder)
- 多模态处理器 (Multimodal Processor)

Author: DeepSeek V4 Team
"""

from .vision_backbone import (
    VisionBackbone,
    ViTConfig,
    PatchEmbedding,
    ViTEncoder,
    ViTAttention,
    ViTMLP,
    ViTBlock,
    ImageTokenizer,
    TileProcessor,
)

from .vision_feature_extract import (
    VisionFeatureExtractor,
    MultiScaleFeatureExtractor,
    HierarchicalFeatureExtractor,
    FeaturePyramid,
)

from .vision_cache_precompute import (
    VisionCachePrecompute,
    VisionCacheConfig,
    CacheManager,
    TileCacheManager,
)

from .text_embed_encoder import (
    TextEmbedEncoder,
    TextEncoderConfig,
    PositionalTextEncoder,
    RotaryTextEncoder,
    TextEncoderWithProjection,
)

from .cross_modal_fusion import (
    CrossModalFusion,
    CrossModalConfig,
    GatedFusion,
    AttentionFusion,
    BilinearFusion,
    ConcatenationFusion,
    HierarchicalFusion,
)

from .modal_projection import (
    ModalProjection,
    ProjectionConfig,
    MLPProjection,
    CrossAttentionProjection,
    LinearProjection,
    ResidualProjection,
    GatedProjection,
)

from .multi_modal_infer import (
    MultiModalInference,
    InferenceConfig,
    BatchInference,
    StreamingInference,
    generate_multimodal,
)

from .multimodal_unified_encoder import (
    MultimodalUnifiedEncoder,
    UnifiedEncoderConfig,
    TextBranch,
    ImageBranch,
    AudioBranch,
)

from .multimodal_processor import (
    ImageProcessor,
    VisionEncoder,
    MultimodalProjector,
    AudioProcessor,
    MultimodalCollator,
    MultimodalTokenizer,
    DynamicImageProcessor,
    VisionProjector,
    V4TransformerBlock,
    V4Config,
    DeepSeekV4Multimodal,
)

__all__ = [
    # Vision Backbone
    'VisionBackbone',
    'ViTConfig',
    'PatchEmbedding',
    'ViTEncoder',
    'ViTAttention',
    'ViTMLP',
    'ViTBlock',
    'ImageTokenizer',
    'TileProcessor',

    # Vision Feature Extraction
    'VisionFeatureExtractor',
    'MultiScaleFeatureExtractor',
    'HierarchicalFeatureExtractor',
    'FeaturePyramid',

    # Vision Cache
    'VisionCachePrecompute',
    'VisionCacheConfig',
    'CacheManager',
    'TileCacheManager',

    # Text Encoder
    'TextEmbedEncoder',
    'TextEncoderConfig',
    'PositionalTextEncoder',
    'RotaryTextEncoder',
    'TextEncoderWithProjection',

    # Cross Modal Fusion
    'CrossModalFusion',
    'CrossModalConfig',
    'GatedFusion',
    'AttentionFusion',
    'BilinearFusion',
    'ConcatenationFusion',
    'HierarchicalFusion',

    # Modal Projection
    'ModalProjection',
    'ProjectionConfig',
    'MLPProjection',
    'CrossAttentionProjection',
    'LinearProjection',
    'ResidualProjection',
    'GatedProjection',

    # Multi-Modal Inference
    'MultiModalInference',
    'InferenceConfig',
    'BatchInference',
    'StreamingInference',
    'generate_multimodal',

    # Unified Encoder
    'MultimodalUnifiedEncoder',
    'UnifiedEncoderConfig',
    'TextBranch',
    'ImageBranch',
    'AudioBranch',

    # Multimodal Processor
    'ImageProcessor',
    'VisionEncoder',
    'MultimodalProjector',
    'AudioProcessor',
    'MultimodalCollator',
    'MultimodalTokenizer',
    'DynamicImageProcessor',
    'VisionProjector',
    'V4TransformerBlock',
    'V4Config',
    'DeepSeekV4Multimodal',
]
