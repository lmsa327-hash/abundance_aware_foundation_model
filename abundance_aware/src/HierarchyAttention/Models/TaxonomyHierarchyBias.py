import torch
from torch import nn

from abundance_aware.src.common.ComputeBias import AttentionBias, register_bias


HIERARCHY_BIAS = 'hierarchies'
@register_bias("hierarchy")
class TaxonomyHierarchyBias(AttentionBias):
    input_key = HIERARCHY_BIAS

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

    def encode(self, hierarchy_embeddings):
        return self.encoder(hierarchy_embeddings) if self.encoder is not None else hierarchy_embeddings

    def forward(self, hierarchy_embeddings):
        B, N, _ = hierarchy_embeddings.shape
        q = self.query_proj(hierarchy_embeddings).view(B, N, self.num_heads, self.rank).transpose(1, 2)
        k = self.key_proj(hierarchy_embeddings).view(B, N, self.num_heads, self.rank).transpose(1, 2)
        return torch.matmul(q, k.transpose(-1, -2))