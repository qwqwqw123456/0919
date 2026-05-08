import torch.nn as nn
class TextEmbedEncoder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, input_ids):
        return self.embed(input_ids)
