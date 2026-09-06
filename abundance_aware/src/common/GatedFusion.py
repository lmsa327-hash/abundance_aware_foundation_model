import torch
from torch import nn


class GatedFusion(nn.Module):
    def __init__(self, hidden_size, init_bias=-6.0):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        # Start each bias's contribution near-zero.
        nn.init.zeros_(self.gate[-2].weight)
        nn.init.constant_(self.gate[-2].bias, init_bias)

    def forward(self, token_embeddings, aux_embeddings):
        interaction = token_embeddings * aux_embeddings
        x = torch.cat([token_embeddings, aux_embeddings, interaction], dim=-1)
        gate = self.gate(x)
        fused = token_embeddings + gate * aux_embeddings
        return fused, gate