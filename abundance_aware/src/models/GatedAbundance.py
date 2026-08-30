import torch.nn as nn
import torch

class GatedAbundanceFusion(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        #init and make a negative bias
        nn.init.zeros_(self.gate[-2].weight)
        nn.init.constant_(self.gate[-2].bias, -6.0)

    def forward(self, token_embeddings, abundance_embeddings):

        combined = torch.cat(
            [token_embeddings, abundance_embeddings],
            dim=-1,
        )

        gate = self.gate(combined)

        fused = token_embeddings + gate * abundance_embeddings

        return fused, gate

class GatedAbundanceFusionWithInteraction(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, token_embeddings, abundance_embeddings):

        interaction = (
            token_embeddings * abundance_embeddings
        )

        x = torch.cat(
            [
                token_embeddings,
                abundance_embeddings,
                interaction,
            ],
            dim=-1,
        )

        gate = self.gate(x)

        fused = (
            token_embeddings
            + gate * abundance_embeddings
        )

        return fused, gate