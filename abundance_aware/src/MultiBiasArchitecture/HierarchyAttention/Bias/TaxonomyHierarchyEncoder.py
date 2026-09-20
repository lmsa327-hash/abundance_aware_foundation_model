import torch
import torch.nn as nn

from abundance_aware.src.MultiBiasArchitecture.HierarchyAttention.utils.Utils import load_vocab_sizes
from tokenize_hierarchy import TAXONOMIC_LEVELS


class TaxonomyHierarchyEncoder(nn.Module):
    """
    Learn a dense representation for each taxon from its
    complete taxonomic hierarchy.

    Input:
        taxonomy_ids:
            [batch, num_taxa, num_levels]
    Levels:
        species
        genus
        family
        order
        class
        phylum
    """

    def __init__(
        self,
        hidden_size,
        level_dim=128,
        pad_idx=0,
    ):
        super().__init__()
        self.levels = TAXONOMIC_LEVELS
        self.vocab_sizes = load_vocab_sizes()
        self.embeddings = nn.ModuleDict({
            level: nn.Embedding(
                self.vocab_sizes[level],
                level_dim,
                padding_idx=pad_idx,
            )
            for level in self.levels
        })

        self.projection = nn.Sequential(
            nn.Linear(
                len(self.levels) * level_dim,
                hidden_size,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_size,
                hidden_size,
            ),
        )

    def forward(self, taxonomy_ids):

        representations = []

        for level_idx, level in enumerate(self.levels):

            ids = taxonomy_ids[:, :, level_idx]

            embedding = self.embeddings[level](ids)

            representations.append(embedding)

        # [B, N, 6 * level_dim]
        hierarchy = torch.cat(
            representations,
            dim=-1,
        )

        # [B, N, hidden_size]
        hierarchy = self.projection(hierarchy)

        return hierarchy