import math

import torch
from torch import nn


class AbundanceEncoder(nn.Module):

    def __init__(self, hidden_size: int, num_freqs: int = 16):
        super().__init__()
        freqs = torch.exp(torch.linspace(0, math.log(1000.0), num_freqs))
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_freqs, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, abundance: torch.Tensor) -> torch.Tensor:
        x = abundance.unsqueeze(-1) * self.freqs
        feats = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return self.mlp(feats)