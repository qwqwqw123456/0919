import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from typing import List, Optional, Tuple, Dict, Any
import base64
import io

class ImageProcessor:
    def __init__(self, image_size: int = 448, patch_size: int = 14):
        self.image_size = image_size
        self.patch_size = patch_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def process(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image).unsqueeze(0)
    
    def process_batch(self, images: List[Image.Image]) -> torch.Tensor:
        tensors = [self.transform(img) for img in images]
        return torch.stack(tensors)
    
    def decode_base64(self, base64_str: str) -> Image.Image:
        image_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_data))
    
    def encode_base64(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

class VisionEncoder(nn.Module):
    def __init__(self, d_model: int = 7168, num_layers: int = 12, num_heads: int = 16):
        super().__init__()
        self.conv_proj = nn.Conv2d(3, d_model, kernel_size=14, stride=14)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, num_heads, d_model * 4)
            for _ in range(num_layers)
        ])
        
        self.proj = nn.Linear(d_model, d_model)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, C, H, W = images.shape
        patches = self.conv_proj(images).flatten(2).transpose(1, 2)
        patches = self.layer_norm(patches)
        
        for layer in self.transformer_layers:
            patches = layer(patches)
        
        cls_token = patches[:, 0, :]
        return self.proj(cls_token)

class MultimodalProjector(nn.Module):
    def __init__(self, visual_dim: int, text_dim: int, output_dim: int):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, output_dim)
        self.text_proj = nn.Linear(text_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, visual_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        visual = self.visual_proj(visual_embeds)
        text = self.text_proj(text_embeds)
        return self.norm(visual + text)

class AudioProcessor:
    def __init__(self, sample_rate: int = 16000, max_length: int = 30):
        self.sample_rate = sample_rate
        self.max_length = max_length
    
    def load_audio(self, file_path: str) -> torch.Tensor:
        try:
            import torchaudio
            waveform, sr = torchaudio.load(file_path)
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            return waveform.mean(dim=0)[:self.sample_rate * self.max_length]
        except ImportError:
            return torch.zeros(self.sample_rate * self.max_length)

class MultimodalCollator:
    def __init__(self, tokenizer, image_processor: ImageProcessor = None):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = []
        images = []
        image_positions = []
        
        for i, example in enumerate(examples):
            text = example.get('text', '')
            text = text.replace('<image>', '').replace('</image>', '')
            
            if 'image' in example and self.image_processor:
                img = example['image']
                if isinstance(img, str):
                    img = self.image_processor.decode_base64(img)
                images.append(self.image_processor.process(img))
                image_positions.append(i)
            
            texts.append(text)
        
        tokenized = self.tokenizer.batch_encode_plus(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        batch = {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask']
        }
        
        if images:
            batch['images'] = torch.cat(images, dim=0)
            batch['image_positions'] = torch.tensor(image_positions)
        
        return batch

class MultimodalTokenizer:
    def __init__(self, text_tokenizer, image_processor: ImageProcessor = None):
        self.text_tokenizer = text_tokenizer
        self.image_processor = image_processor
        self.image_token = '<image>'
    
    def encode(self, text: str, images: List[Image.Image] = None) -> List[int]:
        tokens = []
        parts = text.split(self.image_token)
        
        for i, part in enumerate(parts):
            if part:
                tokens.extend(self.text_tokenizer.encode(part, add_special_tokens=False))
            if i < len(parts) - 1:
                tokens.append(self.text_tokenizer.vocab.get(self.image_token, 128000))
        
        if images and self.image_processor:
            for img in images:
                img_tensor = self.image_processor.process(img)
                tokens.append(self.text_tokenizer.vocab.get(self.image_token, 128000))
        
        return tokens
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        text = []
        for id_ in ids:
            token = self.text_tokenizer.id_to_token.get(id_, '<unk>')
            if skip_special_tokens and token in ['<image>', '<pad>', '<s>', '</s>']:
                continue
            text.append(token)
        return ''.join(text).replace('@@ ', '')