import math
import torch
import torch.nn as nn

class AbundanceEncoder(nn.Module):
    """Continuous standardized-abundance value -> dense vector, via
    sinusoidal features + a zero-initialized MLP (see v1 docstring for
    rationale). Zero-init means a freshly wrapped, freshly loaded
    pretrained checkpoint is numerically IDENTICAL to plain MGM until
    training moves this pathway away from zero."""

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


class AbundanceEncoderWithFiLM(nn.Module):
    def __init__(self, hidden_size, num_freqs=16):
        super().__init__()
        freqs = torch.exp(
            torch.linspace(0, math.log(1000.0), num_freqs)
        )

        self.register_buffer("freqs", freqs)

        self.encoder = nn.Sequential(
            nn.Linear(2 * num_freqs, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )

        # Start as identity:
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    def forward(self, abundance):
        x = abundance.unsqueeze(-1) * self.freqs

        features = torch.cat(
            [torch.sin(x), torch.cos(x)],
            dim=-1
        )

        gamma_beta = self.encoder(features)

        gamma, beta = gamma_beta.chunk(2, dim=-1)

        # identity initialization
        gamma = 1.0 + gamma

        return gamma, beta
