import torch
from torch import nn


class GatedFusion(nn.Module):
    def __init__(self, hidden_size, init_bias=-6.0,uses_aux=True):
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
        x = torch.cat([token_embeddings, aux_embeddings, interaction], dim=-1) if self.uses_aux else token_embeddings
        gate = self.gate(x)
        fused = token_embeddings + gate * aux_embeddings if self.uses_aux else token_embeddings
        return fused, gate

class ElementwiseSigmoidGate(nn.Module):
    """Post-SDPA, element-wise, head-specific sigmoid gate.
    Optionally conditioned on an auxiliary encoded signal (e.g. abundance)
    in addition to the token's own hidden state.
    """
    def __init__(self, hidden_size, uses_aux=False):
        super().__init__()
        in_dim = hidden_size * 2 if uses_aux else hidden_size
        self.proj = nn.Linear(in_dim, hidden_size)
        self.uses_aux = uses_aux
        # Start near-open (gate ~1) so this doesn't suppress signal at init;
        # follows the paper's finding that gating helps most once learned,
        # not by starting closed. Adjust if you want a conservative start instead.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, token_embeddings, aux_embeddings=None):
        x = torch.cat([token_embeddings, aux_embeddings], dim=-1) if self.uses_aux else token_embeddings
        return torch.sigmoid(self.proj(x))   # (batch, seq, hidden_size)