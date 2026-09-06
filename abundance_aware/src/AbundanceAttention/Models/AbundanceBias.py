import math

import torch
from torch import nn

from abundance_aware.src.common.ComputeBias import register_bias, AttentionBias



ABUNDANCE_BIAS = "abundances"
@register_bias("abundance")
class HeadWiseAbundanceBias(AttentionBias):
    input_key = ABUNDANCE_BIAS

    def __init__(self, hidden_size, num_heads, rank=16, encoder=None,
                 use_gated_fusion=False, fusion_init_bias=-6.0):
        super().__init__(use_gated_fusion=use_gated_fusion, hidden_size=hidden_size,
                          fusion_init_bias=fusion_init_bias)
        self.num_heads = num_heads
        self.rank = rank
        self.encoder = encoder

        self.query_proj = nn.Linear(hidden_size, num_heads * rank)
        self.key_proj = nn.Linear(hidden_size, num_heads * rank)
        nn.init.zeros_(self.query_proj.weight); nn.init.zeros_(self.query_proj.bias)
        nn.init.zeros_(self.key_proj.weight); nn.init.zeros_(self.key_proj.bias)

    def encode(self, abundance_embeddings):
        return self.encoder(abundance_embeddings) if self.encoder is not None else abundance_embeddings

    def forward(self, abundance_embeddings):
        B, N, _ = abundance_embeddings.shape
        q = self.query_proj(abundance_embeddings).view(B, N, self.num_heads, self.rank).transpose(1, 2)
        k = self.key_proj(abundance_embeddings).view(B, N, self.num_heads, self.rank).transpose(1, 2)
        return torch.matmul(q, k.transpose(-1, -2))

@register_bias("abundance")
class AbundanceBias(AttentionBias):

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




@register_bias("relative_abundance")
class RelativeAbundanceBias(nn.Module):
    def __init__(self,num_heads, hidden_dim=32,epsilon=1e-8, encoder=None,
                 use_gated_fusion=False, fusion_init_bias=-6.0):
        super().__init__(use_gated_fusion=use_gated_fusion, hidden_size=hidden_dim,
                         fusion_init_bias=fusion_init_bias)
        self.epsilon = epsilon
        self.encoder = encoder
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )

    def encode(self, abundance_embeddings):
        return self.encoder(abundance_embeddings) if self.encoder is not None else abundance_embeddings
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
