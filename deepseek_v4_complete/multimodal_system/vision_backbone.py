import torch.nn as nn
class VisionBackbone(nn.Module):
    def __init__(self, d_model=7168):
        super().__init__()
        self.conv = nn.Conv2d(3, d_model, 14, stride=14)
    def forward(self, x):
        return self.conv(x).flatten(2).transpose(1,2)
