import math

import torch
from torch import nn


class HeadWiseAbundanceBias(nn.Module):

    def __init__(self, hidden_size, num_heads, rank=16):
        super().__init__()

        self.num_heads = num_heads
        self.rank = rank

        self.query_proj = nn.Linear(hidden_size, num_heads * rank)


        self.key_proj = nn.Linear(hidden_size, num_heads * rank)

        nn.init.zeros_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)

        nn.init.zeros_(self.key_proj.weight)
        nn.init.zeros_(self.key_proj.bias)

    def forward(self, abundance_embeddings):

        B, N, _ = abundance_embeddings.shape

        q = self.query_proj(abundance_embeddings)
        k = self.key_proj(abundance_embeddings)

        q = q.view(B, N, self.num_heads, self.rank)

        k = k.view(B, N, self.num_heads, self.rank)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        # [B,H,N,r] x [B,H,r,N]
        bias = torch.matmul(
            q,
            k.transpose(-1, -2)
        )

        return bias

class AbundanceAttentionBias(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, abundance_embeddings):

        B, N, D = abundance_embeddings.shape

        left = abundance_embeddings.unsqueeze(2)
        right = abundance_embeddings.unsqueeze(1)

        left = left.expand(-1, -1, N, -1)
        right = right.expand(-1, N, -1, -1)

        pair = torch.cat([left, right], dim=-1)

        bias = self.net(pair).squeeze(-1)

        return bias



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