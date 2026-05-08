import torch
import torch.nn as nn
import torch.nn.functional as F
from .activation_func import SwiGLU

class DeepSeekMoE(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_shared: int = 2, n_routed: int = 384, top_k: int = 6):
        super().__init__()
        self.n_routed = n_routed
        self.top_k = top_k

        self.shared_experts = nn.ModuleList([
            SwiGLU(d_model, d_ff) for _ in range(n_shared)
        ])
        self.gate = nn.Linear(d_model, n_routed, bias=False)
        self.experts = nn.ModuleList([
            SwiGLU(d_model, d_ff) for _ in range(n_routed)
        ])

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)
        shared_out = sum(expert(x_flat) for expert in self.shared_experts) / len(self.shared_experts)

        logits = self.gate(x_flat)
        weights = F.softmax(logits, dim=-1)
        topk_weights, topk_ids = torch.topk(weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        routed_out = torch.zeros_like(x_flat)
        aux_loss = None
        if self.training:
            num_tokens = x_flat.shape[0]
            f = weights.mean(dim=0)
            one_hot = F.one_hot(topk_ids, num_classes=self.n_routed).float()
            p = one_hot.sum(dim=(0,1)) / (num_tokens * self.top_k)
            aux_loss = self.n_routed * (f * p).sum() * 0.01

        for i in range(self.n_routed):
            mask = (topk_ids == i).any(dim=-1)
            if mask.any():
                expert_input = x_flat[mask]
                expert_output = self.experts[i](expert_input)
                idx_mask = (topk_ids == i)
                w = topk_weights[idx_mask].unsqueeze(-1)
                routed_out[mask] += w * expert_output

        out = shared_out + routed_out
        return out.reshape(B, L, D), aux_loss
