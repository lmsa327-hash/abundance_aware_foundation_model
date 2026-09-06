import math
import torch
import torch.nn as nn

from abundance_aware.src.common.GatedFusion import GatedFusion

# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
BIAS_REGISTRY = {}

def register_bias(name):
    """Decorator: register an attention-bias module under `name`."""
    def decorator(cls):
        if name in BIAS_REGISTRY:
            raise ValueError(f"Bias '{name}' is already registered")
        cls.bias_name = name
        BIAS_REGISTRY[name] = cls
        return cls
    return decorator


class AttentionBias(nn.Module):
    input_key: str = None

    def __init__(self, use_gated_fusion=False, hidden_size=None, fusion_init_bias=-6.0):
        super().__init__()
        self.fusion = None
        if use_gated_fusion:
            if hidden_size is None:
                raise ValueError(f"{type(self).__name__}: hidden_size required for gated fusion")
            self.fusion = GatedFusion(hidden_size, init_bias=fusion_init_bias)

    def encode(self, raw):
        return raw  # override if raw needs an encoder step

    def forward(self, embeddings):
        raise NotImplementedError  # additive attention-bias term

    def compute(self, raw):
        return self.forward(self.encode(raw))

    def compute_from_encoded(self, encoded):
        return self.forward(encoded)

    def fuse_from_encoded(self, token_embeddings, encoded):
        if self.fusion is None:
            return None, None
        return self.fusion(token_embeddings, encoded)

    def fuse(self, token_embeddings, raw):
        """(fused, gate) if this bias does gated fusion, else (None, None)."""
        if self.fusion is None:
            return None, None
        return self.fusion(token_embeddings, self.encode(raw))