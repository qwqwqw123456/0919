"""
Multimodal Processor - 多模态处理器
=================================

该模块提供完整的多模态处理功能，包括图像处理、视觉编码、投影、音频处理等功能。

主要组件：
- ImageProcessor: 图像处理器
- VisionEncoder: 视觉编码器
- VisionProjector: 视觉投影器
- DynamicImageProcessor: 动态图像处理器
- AudioProcessor: 音频处理器
- MultimodalProjector: 多模态投影器
- MultimodalCollator: 多模态数据整理器
- MultimodalTokenizer: 多模态分词器
- V4Config: DeepSeek V4 配置
- V4TransformerBlock: V4 Transformer 块
- DeepSeekV4Multimodal: DeepSeek V4 多模态模型

特点：
- 完整的端到端多模态处理流程
- 支持多种图像预处理策略
- 支持多模态数据整理和对齐
- 统一的模型架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Dict, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from PIL import Image
import numpy as np


@dataclass
class V4Config:
    """
    DeepSeek V4 配置

    参数:
        vision_config: 视觉配置
        text_config: 文本配置
        hidden_size: 隐藏层维度
        num_hidden_layers: 隐藏层数
        num_attention_heads: 注意力头数
        intermediate_size: 中间层大小
        vocab_size: 词表大小
        max_position_embeddings: 最大位置嵌入
        dropout_prob: Dropout概率
    """
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    vocab_size: int = 129280
    max_position_embeddings: int = 8192
    dropout_prob: float = 0.1
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3


class ImageProcessor(nn.Module):
    """
    图像处理器

    负责图像的预处理和增强
    """

    def __init__(
        self,
        image_size: int = 224,
        mean: List[float] = None,
        std: List[float] = None,
        resize: bool = True
    ):
        super().__init__()
        self.image_size = image_size
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]
        self.resize = resize

    def preprocess(
        self,
        images: Union[Image.Image, List[Image.Image], torch.Tensor]
    ) -> torch.Tensor:
        """
        预处理图像

        参数:
            images: 输入图像

        返回:
            处理后的图像张量 [B, C, H, W]
        """
        if isinstance(images, Image.Image):
            images = [images]

        processed = []
        for img in images:
            if isinstance(img, Image.Image):
                if self.resize:
                    img = img.resize((self.image_size, self.image_size))

                img_array = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)

                for c in range(3):
                    img_tensor[c] = (img_tensor[c] - self.mean[c]) / self.std[c]

                processed.append(img_tensor)
            elif isinstance(img, torch.Tensor):
                processed.append(img)

        batch = torch.stack(processed, dim=0)
        return batch

    def forward(self, images: Union[Image.Image, List[Image.Image], torch.Tensor]) -> torch.Tensor:
        """前向传播"""
        return self.preprocess(images)


class VisionEncoder(nn.Module):
    """
    视觉编码器

    负责将图像编码为视觉特征
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072
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

        self.transformer_layers = nn.ModuleList([
            TransformerLayer(hidden_size, num_attention_heads, intermediate_size)
            for _ in range(num_hidden_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        参数:
            pixel_values: 像素值 [B, C, H, W]
            output_hidden_states: 是否输出所有隐藏状态

        返回:
            编码结果
        """
        B = pixel_values.shape[0]

        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        hidden_states = torch.cat([cls_tokens, patch_embeds], dim=1)

        hidden_states = hidden_states + self.position_embedding

        all_hidden_states = () if output_hidden_states else None

        for layer in self.transformer_layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states = layer(hidden_states)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return {
            'last_hidden_state': hidden_states,
            'hidden_states': all_hidden_states,
            'pooler_output': hidden_states[:, 0]
        }


class TransformerLayer(nn.Module):
    """
    Transformer 层

    包含自注意力和前馈网络
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_attention_heads,
            dropout=dropout,
            batch_first=True
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)

        attn_output, _ = self.attention(hidden_states, hidden_states, hidden_states)
        hidden_states = residual + attn_output

        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))

        return hidden_states


class VisionProjector(nn.Module):
    """
    视觉投影器

    将视觉特征投影到语言模型空间
    """

    def __init__(
        self,
        vision_hidden_size: int = 768,
        llm_hidden_size: int = 768,
        projection_type: str = "mlp"
    ):
        super().__init__()

        if projection_type == "linear":
            self.projection = nn.Linear(vision_hidden_size, llm_hidden_size)
        elif projection_type == "mlp":
            self.projection = nn.Sequential(
                nn.Linear(vision_hidden_size, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size)
            )
        else:
            self.projection = nn.Linear(vision_hidden_size, llm_hidden_size)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            vision_features: 视觉特征 [B, N, D_v]

        返回:
            投影后的特征 [B, N, D_l]
        """
        return self.projection(vision_features)


class DynamicImageProcessor(nn.Module):
    """
    动态图像处理器

    支持动态调整图像大小和比例
    """

    def __init__(
        self,
        target_sizes: List[Tuple[int, int]] = None,
        mean: List[float] = None,
        std: List[float] = None
    ):
        super().__init__()
        self.target_sizes = target_sizes or [(224, 224), (336, 336), (504, 504)]
        self.mean = mean or [0.48145466, 0.4578275, 0.40821073]
        self.std = std or [0.26862954, 0.26130258, 0.27577711]

    def forward(
        self,
        images: Union[Image.Image, List[Image.Image]],
        target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        动态处理图像

        参数:
            images: 输入图像
            target_size: 目标尺寸

        返回:
            处理后的图像张量
        """
        if target_size is None:
            target_size = self.target_sizes[0]

        processed = []
        for img in images if isinstance(images, list) else [images]:
            if isinstance(img, Image.Image):
                img = img.resize(target_size)
                img_array = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)

                for c in range(3):
                    img_tensor[c] = (img_tensor[c] - self.mean[c]) / self.std[c]

                processed.append(img_tensor)

        return torch.stack(processed, dim=0)


class AudioProcessor(nn.Module):
    """
    音频处理器

    负责音频特征的提取和预处理
    """

    def __init__(
        self,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        sample_rate: int = 16000
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sample_rate = sample_rate

        self.mel_filters = self._create_mel_filters()

    def _create_mel_filters(self) -> torch.Tensor:
        """创建梅尔滤波器"""
        import numpy as np

        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        fmin = 0
        fmax = self.sample_rate / 2
        n_mels = self.n_mels

        mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
        hz = mel_to_hz(mels)

        freq_bins = np.floor((self.n_fft + 1) * hz / self.sample_rate).astype(np.int32)

        filters = np.zeros((n_mels, self.n_fft // 2 + 1))
        for m in range(1, n_mels + 1):
            f_m_minus = freq_bins[m - 1]
            f_m = freq_bins[m]
            f_m_plus = freq_bins[m + 1]

            for k in range(f_m_minus, f_m):
                filters[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                filters[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

        return torch.from_numpy(filters).float()

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        处理音频

        参数:
            audio: 原始音频波形 [B, T]

        返回:
            梅尔频谱图 [B, n_mels, T']
        """
        B = audio.shape[0]

        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft, device=audio.device),
            return_complex=True
        )

        magnitude = torch.abs(stft)

        mel_spec = torch.matmul(self.mel_filters.to(audio.device), magnitude)

        mel_spec = torch.log(mel_spec + 1e-9)

        return mel_spec


class MultimodalProjector(nn.Module):
    """
    多模态投影器

    统一管理所有模态的投影
    """

    def __init__(
        self,
        vision_hidden_size: int = 768,
        audio_hidden_size: int = 768,
        llm_hidden_size: int = 768
    ):
        super().__init__()

        self.vision_projector = VisionProjector(
            vision_hidden_size,
            llm_hidden_size,
            projection_type="mlp"
        )

        self.audio_projector = VisionProjector(
            audio_hidden_size,
            llm_hidden_size,
            projection_type="mlp"
        )

    def project_vision(self, vision_features: torch.Tensor) -> torch.Tensor:
        """投影视觉特征"""
        return self.vision_projector(vision_features)

    def project_audio(self, audio_features: torch.Tensor) -> torch.Tensor:
        """投影音频特征"""
        return self.audio_projector(audio_features)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播

        参数:
            features: 各模态的特征字典

        返回:
            投影后的特征字典
        """
        projected = {}

        if 'vision' in features:
            projected['vision'] = self.project_vision(features['vision'])

        if 'audio' in features:
            projected['audio'] = self.project_audio(features['audio'])

        return projected


class MultimodalCollator:
    """
    多模态数据整理器

    用于整理和填充多模态批次数据
    """

    def __init__(
        self,
        tokenizer,
        padding: bool = True,
        padding_length: int = 2048
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.padding_length = padding_length

    def __call__(
        self,
        batch: List[Dict]
    ) -> Dict[str, torch.Tensor]:
        """
        整理批次数据

        参数:
            batch: 样本列表

        返回:
            整理后的批次数据
        """
        texts = []
        images = []
        image_sizes = []

        for item in batch:
            if 'text' in item:
                texts.append(item['text'])
            if 'image' in item:
                images.append(item['image'])
                image_sizes.append(item['image'].size if hasattr(item['image'], 'size') else None)

        encoded = self.tokenizer(
            texts,
            padding=self.padding,
            max_length=self.padding_length,
            truncation=True,
            return_tensors='pt'
        )

        result = {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }

        if images:
            result['pixel_values'] = torch.stack([
                img if isinstance(img, torch.Tensor) else torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
                for img in images
            ])

        return result


class MultimodalTokenizer:
    """
    多模态分词器

    扩展标准分词器以支持多模态输入
    """

    def __init__(
        self,
        tokenizer,
        image_token: str = "<image>",
        image_pad_token: str = "<image_pad>",
        num_image_tokens: int = 256
    ):
        self.tokenizer = tokenizer
        self.image_token = image_token
        self.image_pad_token = image_pad_token
        self.num_image_tokens = num_image_tokens

        special_tokens = [image_token, image_pad_token]
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': special_tokens
        })

        self.image_token_id = self.tokenizer.convert_tokens_to_ids(image_token)
        self.image_pad_token_id = self.tokenizer.convert_tokens_to_ids(image_pad_token)

    def __call__(
        self,
        text: Union[str, List[str]],
        images: Optional[List] = None,
        padding: bool = True,
        max_length: int = 2048,
        truncation: bool = True,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """
        分词

        参数:
            text: 输入文本
            images: 图像列表
            padding: 是否填充
            max_length: 最大长度
            truncation: 是否截断
            return_tensors: 返回张量类型

        返回:
            分词结果
        """
        if isinstance(text, str):
            text = [text]

        processed_texts = []
        for t in text:
            if self.image_token in t:
                processed_t = t.replace(self.image_token, self.image_pad_token * self.num_image_tokens)
            else:
                processed_t = t
            processed_texts.append(processed_t)

        encoded = self.tokenizer(
            processed_texts,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_tensors=return_tensors
        )

        return encoded

    def decode(self, token_ids: torch.Tensor) -> str:
        """解码"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)


class V4TransformerBlock(nn.Module):
    """
    DeepSeek V4 Transformer 块

    改进的 Transformer 层，支持多模态输入
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        dropout: float = 0.1,
        use_gated_attention: bool = True
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.use_gated_attention = use_gated_attention

        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_attention_heads,
            dropout=dropout,
            batch_first=True
        )

        if use_gated_attention:
            self.gate_q = nn.Linear(hidden_size, hidden_size)
            self.gate_k = nn.Linear(hidden_size, hidden_size)
            self.gate_v = nn.Linear(hidden_size, hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size)
        )

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播"""
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)

        if self.use_gated_attention:
            q = hidden_states + torch.tanh(self.gate_q(hidden_states)) * hidden_states
            k = hidden_states + torch.tanh(self.gate_k(hidden_states)) * hidden_states
            v = hidden_states + torch.tanh(self.gate_v(hidden_states)) * hidden_states
        else:
            q = k = v = hidden_states

        attn_output, _ = self.attention(q, k, v, attn_mask=attention_mask)
        hidden_states = residual + self.dropout1(attn_output)

        hidden_states = hidden_states + self.dropout2(self.mlp(self.norm2(hidden_states)))

        return hidden_states


class DeepSeekV4Multimodal(nn.Module):
    """
    DeepSeek V4 多模态模型

    完整的多模态模型，整合所有组件
    """

    def __init__(
        self,
        config: Optional[V4Config] = None,
        vision_encoder: Optional[VisionEncoder] = None,
        vision_projector: Optional[VisionProjector] = None,
        language_model: Optional[nn.Module] = None
    ):
        """
        初始化 DeepSeek V4 多模态模型

        参数:
            config: V4 配置
            vision_encoder: 视觉编码器
            vision_projector: 视觉投影器
            language_model: 语言模型
        """
        super().__init__()

        if config is None:
            config = V4Config()

        self.config = config

        self.image_processor = ImageProcessor(
            image_size=config.image_size
        )

        if vision_encoder is None:
            self.vision_encoder = VisionEncoder(
                image_size=config.image_size,
                hidden_size=config.hidden_size,
                num_hidden_layers=config.num_hidden_layers
            )
        else:
            self.vision_encoder = vision_encoder

        if vision_projector is None:
            self.vision_projector = VisionProjector(
                vision_hidden_size=config.hidden_size,
                llm_hidden_size=config.hidden_size
            )
        else:
            self.vision_projector = vision_projector

        self.language_model = language_model

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        编码图像

        参数:
            pixel_values: 像素值 [B, C, H, W]

        返回:
            编码后的图像特征
        """
        vision_outputs = self.vision_encoder(pixel_values)
        vision_features = vision_outputs['last_hidden_state']

        projected_features = self.vision_projector(vision_features)

        return projected_features

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        参数:
            input_ids: 输入 token IDs
            pixel_values: 像素值
            attention_mask: 注意力掩码
            labels: 标签

        返回:
            模型输出
        """
        vision_features = None

        if pixel_values is not None:
            vision_features = self.encode_images(pixel_values)

        if self.language_model is not None:
            outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                vision_features=vision_features,
                labels=labels
            )
        else:
            outputs = {
                'logits': None,
                'loss': None
            }

        return outputs

    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        max_length: int = 2048,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """
        生成文本

        参数:
            input_ids: 输入 token IDs
            pixel_values: 像素值
            max_length: 最大生成长度
            temperature: 温度
            top_p: Top-p 采样

        返回:
            生成的 token IDs
        """
        self.eval()
        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.forward(
                    input_ids=generated,
                    pixel_values=pixel_values
                )

                logits = outputs['logits'][:, -1, :] / temperature

                probs = F.softmax(logits, dim=-1)

                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

                    sorted_indices_to_remove = cumsum_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    probs[indices_to_remove] = 0

                next_token = torch.multinomial(probs, num_samples=1)

                generated = torch.cat([generated, next_token], dim=-1)

                if (next_token == 2).all():
                    break

        return generated


def create_multimodal_processor(
    config: Optional[V4Config] = None,
    **kwargs
) -> DeepSeekV4Multimodal:
    """
    创建多模态处理器的工厂函数

    参数:
        config: V4 配置
        **kwargs: 其他配置参数

    返回:
        DeepSeekV4Multimodal 实例
    """
    if config is None:
        config = V4Config(**kwargs)

    return DeepSeekV4Multimodal(config)


class MultimodalFeatureExtractor:
    """
    多模态特征提取器

    提取和融合多模态特征
    """

    def __init__(
        self,
        vision_encoder: VisionEncoder,
        text_encoder: nn.Module,
        projection: VisionProjector
    ):
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.projection = projection

    @torch.no_grad()
    def extract_vision_features(
        self,
        pixel_values: torch.Tensor
    ) -> torch.Tensor:
        """提取视觉特征"""
        vision_outputs = self.vision_encoder(pixel_values)
        vision_features = vision_outputs['last_hidden_state']
        return self.projection(vision_features)

    @torch.no_grad()
    def extract_text_features(
        self,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """提取文本特征"""
        if hasattr(self.text_encoder, 'forward'):
            text_outputs = self.text_encoder(input_ids)
            if isinstance(text_outputs, dict):
                return text_outputs.get('last_hidden_state', text_outputs.get('pooler_output'))
            return text_outputs
        return self.text_encoder(input_ids)

    def compute_similarity(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """计算视觉和文本特征的相似度"""
        vision_features = F.normalize(vision_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        similarity = torch.matmul(vision_features, text_features.transpose(-2, -1))

        return similarity


def create_feature_extractor(
    vision_config: Dict,
    text_config: Dict,
    projection_type: str = "mlp"
) -> MultimodalFeatureExtractor:
    """
    创建多模态特征提取器

    参数:
        vision_config: 视觉编码器配置
        text_config: 文本编码器配置
        projection_type: 投影类型

    返回:
        MultimodalFeatureExtractor 实例
    """
    vision_encoder = VisionEncoder(**vision_config)

    text_encoder = nn.Linear(text_config['hidden_size'], text_config['hidden_size'])

    projection = VisionProjector(
        vision_hidden_size=vision_config['hidden_size'],
        llm_hidden_size=text_config['hidden_size'],
        projection_type=projection_type
    )

    return MultimodalFeatureExtractor(vision_encoder, text_encoder, projection)
