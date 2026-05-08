import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VLContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / temperature))

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor):
        img = F.normalize(image_features, dim=-1)
        txt = F.normalize(text_features, dim=-1)
        scale = self.logit_scale.exp()
        logits_per_image = scale * img @ txt.t()
        logits_per_text = logits_per_image.t()
        labels = torch.arange(img.size(0), device=img.device)
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        return (loss_i + loss_t) / 2

class TotalLoss(nn.Module):
    def __init__(self, weight_mtp: float = 0.3, weight_aux: float = 0.01, weight_cl: float = 0.1):
        super().__init__()
        self.weight_mtp = weight_mtp
        self.weight_aux = weight_aux
        self.weight_cl = weight_cl
        self.contrastive = VLContrastiveLoss()

    def forward(self, logits_main, hidden, input_ids, aux_loss, mtp_losses,
                image_feat=None, text_feat=None):
        ce = F.cross_entropy(logits_main[:, :-1, :].reshape(-1, logits_main.size(-1)),
                             input_ids[:, 1:].reshape(-1), ignore_index=0)
        loss = ce
        if aux_loss is not None:
            loss += self.weight_aux * aux_loss
        if mtp_losses is not None:
            loss += self.weight_mtp * sum(mtp_losses) / len(mtp_losses)
        if image_feat is not None and text_feat is not None:
            loss += self.weight_cl * self.contrastive(image_feat, text_feat)
        return loss
