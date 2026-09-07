"""
Abundance-aware microbial transformer.

Token composition follows the BiomeGPT-style additive fusion:
    T = LayerNorm(S) + LayerNorm(A)
where S is a fixed per-genus identity embedding and A is a continuous
embedding of that genus's relative abundance in the sample, produced by
a small MLP. A learned zero-abundance embedding A0 stands in for absent
taxa and doubles as the source-derived baseline for Integrated Gradients
attribution later (see attribution.py).

Continuous (not binned) abundance encoding is a deliberate choice: it
keeps the abundance -> embedding map differentiable end-to-end, which
matters for the "integrate through the encoder" variant of IG.
"""
import math
import torch
import torch.nn as nn


class TaxonAbundanceEmbedding(nn.Module):
    """Composes S (taxon identity) + A (abundance) into one token embedding."""

    def __init__(self, vocab_size: int, d_model: int, abundance_hidden: int = 64,
                 pad_idx: int = 0):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        # S: fixed per-genus identity embedding.
        self.taxon_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        # A: continuous abundance -> embedding MLP. Input is a single scalar
        # (we recommend CLR-transformed or log1p relative abundance upstream).
        self.abundance_encoder = nn.Sequential(
            nn.Linear(1, abundance_hidden),
            nn.GELU(),
            nn.Linear(abundance_hidden, d_model),
        )

        # A0: learned "zero-abundance" embedding. Used (a) whenever a taxon
        # slot is absent/padded, and (b) as the IG baseline embedding later.
        self.zero_abundance = nn.Parameter(torch.zeros(d_model))

        self.norm_s = nn.LayerNorm(d_model)
        self.norm_a = nn.LayerNorm(d_model)

    def encode_abundance(self, abundance: torch.Tensor) -> torch.Tensor:
        """abundance: (..., 1) float tensor -> (..., d_model) embedding."""
        return self.abundance_encoder(abundance)

    def forward(self, taxon_ids: torch.Tensor, abundance: torch.Tensor,
                abundance_embed_override: torch.Tensor = None) -> torch.Tensor:
        """
        taxon_ids: (batch, seq_len) long
        abundance: (batch, seq_len, 1) float, relative abundance
                   (pre-transformed by caller, e.g. log1p or CLR)
        abundance_embed_override: optional (batch, seq_len, d_model) — lets
            the attribution code inject an interpolated abundance embedding
            directly, bypassing the encoder, for embedding-space IG.
        """
        s = self.norm_s(self.taxon_embed(taxon_ids))

        if abundance_embed_override is not None:
            a = self.norm_a(abundance_embed_override)
        else:
            a_raw = self.encode_abundance(abundance)
            # padded positions get the learned zero-abundance embedding
            pad_mask = (taxon_ids == self.pad_idx).unsqueeze(-1)
            a_raw = torch.where(pad_mask, self.zero_abundance.expand_as(a_raw), a_raw)
            a = self.norm_a(a_raw)

        return s + a


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                              (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MicrobialTransformer(nn.Module):
    """
    Causal transformer over rank-ordered (genus, abundance) sequences.
    Two prediction heads at every position, mirroring MGM's "predict the
    next genus" objective but extended with an abundance-regression head
    since our tokenization carries abundance as a continuous signal.
    """

    def __init__(self, vocab_size: int, d_model: int = 384, n_heads: int = 6,
                 n_layers: int = 8, ffn_dim: int = 1536, max_len: int = 512,
                 dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.pad_idx = pad_idx
        self.token_embed = TaxonAbundanceEmbedding(vocab_size, d_model, pad_idx=pad_idx)
        self.pos_encode = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.genus_head = nn.Linear(d_model, vocab_size)
        self.abundance_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    @staticmethod
    def causal_mask(seq_len: int, device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, taxon_ids: torch.Tensor, abundance: torch.Tensor,
                abundance_embed_override: torch.Tensor = None,
                return_hidden: bool = False):
        """
        taxon_ids: (batch, seq_len)
        abundance: (batch, seq_len, 1)
        """
        batch, seq_len = taxon_ids.shape
        x = self.token_embed(taxon_ids, abundance, abundance_embed_override)
        x = self.dropout(self.pos_encode(x))

        pad_mask = taxon_ids == self.pad_idx  # (batch, seq_len), True = ignore
        attn_mask = self.causal_mask(seq_len, taxon_ids.device)

        hidden = self.encoder(x, mask=attn_mask, src_key_padding_mask=pad_mask)

        genus_logits = self.genus_head(hidden)
        abundance_pred = self.abundance_head(hidden)

        if return_hidden:
            return genus_logits, abundance_pred, hidden
        return genus_logits, abundance_pred