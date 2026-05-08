import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTokenPredictor(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, n_predict: int = 2):
        super().__init__()
        self.n_predict = n_predict
        self.heads = nn.ModuleList([nn.Linear(d_model, vocab_size, bias=False) for _ in range(n_predict)])

    def forward(self, hidden: torch.Tensor, labels: torch.Tensor = None):
        logits = [head(hidden) for head in self.heads]
        if labels is not None:
            losses = []
            for i, logit in enumerate(logits):
                shift_logits = logit[:, :, :].contiguous()
                shift_labels = labels[:, 1+i:1+i+logit.size(1)].contiguous()
                loss = F.cross_entropy(shift_logits.view(-1, logit.size(-1)),
                                       shift_labels.view(-1), ignore_index=0)
                losses.append(loss)
            return logits, losses
        return logits, None
