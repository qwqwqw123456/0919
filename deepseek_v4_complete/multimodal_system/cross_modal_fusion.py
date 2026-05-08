import torch
import torch.nn as nn

class CrossModalFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Linear(d_model*2, d_model)
    def forward(self, text_feat, image_feat):
        combined = torch.cat([text_feat, image_feat], dim=-1)
        return torch.sigmoid(self.gate(combined)) * text_feat + (1 - torch.sigmoid(self.gate(combined))) * image_feat
