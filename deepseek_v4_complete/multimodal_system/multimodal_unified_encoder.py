import torch.nn as nn
class MultimodalUnifiedEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.text_enc = TextEmbedEncoder(10000, d_model)
        self.image_enc = VisionBackbone(d_model)
    def forward(self, text_ids, image):
        t = self.text_enc(text_ids)
        i = self.image_enc(image)
        return t + i.mean(dim=1, keepdim=True)
