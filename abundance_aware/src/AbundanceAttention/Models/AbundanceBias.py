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

class RelativeAbundanceBias(nn.Module):

    def __init__(
        self,
        num_heads,
        hidden_dim=32,
        epsilon=1e-8,
    ):
        super().__init__()

        self.epsilon = epsilon

        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )

    def forward(self, abundances):

        log_a = torch.log(
            abundances + self.epsilon
        )

        ai = log_a.unsqueeze(-1)
        aj = log_a.unsqueeze(-2)

        relative = ai - aj

        features = torch.cat(
            [
                relative.unsqueeze(-1),
                relative.abs().unsqueeze(-1),
            ],
            dim=-1,
        )

        # [B, N, N, H]
        bias = self.mlp(features)

        # [B, H, N, N]
        return bias.permute(0, 3, 1, 2)

# class RelativeAbundanceBias(nn.Module):
#     """
#     Converts pairwise relative abundance into an
#     attention bias for each attention head.
#
#     Input:
#         abundances: [batch, seq_len]
#
#     Output:
#         bias: [batch, num_heads, seq_len, seq_len]
#     """
#
#     def __init__(
#         self,
#         num_heads: int,
#         hidden_dim: int = 32,
#         epsilon: float = 1e-8,
#     ):
#         super().__init__()
#
#         self.num_heads = num_heads
#         self.epsilon = epsilon
#
#         self.mlp = nn.Sequential(
#             nn.Linear(1, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, num_heads),
#         )
#
#     def forward(self, abundances):
#
#         # [B, N]
#         log_abundance = torch.log(
#             abundances + self.epsilon
#         )
#
#         # [B, N, 1]
#         ai = log_abundance.unsqueeze(-1)
#
#         # [B, 1, N]
#         aj = log_abundance.unsqueeze(-2)
#
#         # [B, N, N]
#         relative_abundance = ai - aj
#
#         # [B, N, N, 1]
#         relative_abundance = relative_abundance.unsqueeze(-1)
#
#         # [B, N, N, H]
#         bias = self.mlp(relative_abundance)
#
#         # [B, H, N, N]
#         bias = bias.permute(0, 3, 1, 2)
#
#         return bias