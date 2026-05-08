import torch.nn as nn
class ModalProjection(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)
    def forward(self, x):
        return self.proj(x)
