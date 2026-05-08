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
