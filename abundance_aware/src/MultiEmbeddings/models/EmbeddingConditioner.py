import torch.nn as nn

from abundance_aware.src.MultiBiasArchitecture.common.GatedFusion import GatedFusion
from abundance_aware.src.common.TaxonomyHierarchyEncoder import \
    TaxonomyHierarchyEncoder
from abundance_aware.src.MultiBiasArchitecture.HierarchyAttention.utils.Utils import load_taxonomy_by_token
from abundance_aware.src.common.AbundanceEncoder import AbundanceEncoder, AbundanceEncoderWithFiLM


CONDITIONER_REGISTRY = {}

def register_conditioner(name):
    def decorator(cls):
        if name in CONDITIONER_REGISTRY:
            raise ValueError(f"Conditioner '{name}' already registered")
        cls.conditioner_name = name
        CONDITIONER_REGISTRY[name] = cls
        return cls
    return decorator


class EmbeddingConditioner(nn.Module):

    input_key: str = None
    collate_strategy: str = None  # "sequence" (pad like input_ids), "fixed" (stack as-is), or None (derived, e.g. hierarchy from input_ids)

    def __init__(self, use_gated_fusion=False, hidden_size=None, fusion_init_bias=-6.0):
        super().__init__()
        self.fusion = None
        if use_gated_fusion:
            if hidden_size is None:
                raise ValueError(f"{type(self).__name__}: hidden_size required for gated fusion")
            self.fusion = GatedFusion(hidden_size, init_bias=fusion_init_bias)

    def encode(self, raw, input_ids=None):
        return raw

    def apply(self, embeddings, encoded):
        """Default: additive, optionally gated. Override for non-additive
        conditioning (FiLM's gamma*x + beta)."""
        if self.fusion is not None:
            embeddings, _gate = self.fusion(embeddings, encoded)
            return embeddings
        return embeddings + encoded

    def condition(self, embeddings, raw, input_ids=None):
        return self.apply(embeddings, self.encode(raw, input_ids=input_ids))

    def zero_init_on_load(self):
        """Override to re-zero whatever HF's from_pretrained silently
        re-initializes when this module is missing from a checkpoint."""
        pass

    def logging_metrics(self):
        """Scalar diagnostics to log each eval step. Default: how open the
        gate is, via the pre-sigmoid bias — directly analogous to the old
        single lambda_A scalar, generalized per-conditioner."""
        if self.fusion is not None:
            return {"gate_bias": self.fusion.gate[-2].bias.detach().mean().item()}
        return {}


# --------------------------------------------------------------------------
# Your two existing conditioners, ported in
# --------------------------------------------------------------------------
@register_conditioner("abundance")
class AbundanceConditioner(EmbeddingConditioner):
    input_key = "abundances"
    collate_strategy = "sequence"  # one value per token, pads like input_ids

    def __init__(self, hidden_size, use_gated_fusion=False, fusion_init_bias=-6.0):
        super().__init__(use_gated_fusion=use_gated_fusion, hidden_size=hidden_size,
                          fusion_init_bias=fusion_init_bias)
        self.encoder = AbundanceEncoder(hidden_size)

    def encode(self, raw, input_ids=None):
        return self.encoder(raw)

    def zero_init_on_load(self):
        nn.init.zeros_(self.encoder.mlp[-1].weight)
        nn.init.zeros_(self.encoder.mlp[-1].bias)


@register_conditioner("abundance_film")
class AbundanceFiLMConditioner(EmbeddingConditioner):
    """FiLM is multiplicative+additive, not additive-then-gated, so it
    overrides `apply` directly rather than using GatedFusion."""
    input_key = "abundances"
    collate_strategy = "sequence"

    def __init__(self, hidden_size):
        super().__init__(use_gated_fusion=False)
        self.encoder = AbundanceEncoderWithFiLM(hidden_size)

    def encode(self, raw, input_ids=None):
        return self.encoder(raw)  # -> (gamma, beta)

    def apply(self, embeddings, encoded):
        gamma, beta = encoded
        return gamma * embeddings + beta

    def zero_init_on_load(self):
        nn.init.zeros_(self.encoder.mlp[-1].weight)
        nn.init.zeros_(self.encoder.mlp[-1].bias)


@register_conditioner("hierarchy")
class HierarchyConditioner(EmbeddingConditioner):
    input_key = "input_ids"
    collate_strategy = None

    def __init__(self, hidden_size, use_gated_fusion=False, fusion_init_bias=-6.0):
        super().__init__(use_gated_fusion=use_gated_fusion, hidden_size=hidden_size,
                          fusion_init_bias=fusion_init_bias)
        self.encoder = TaxonomyHierarchyEncoder(hidden_size)
        self._tax_by_token = None

    def encode(self, raw, input_ids=None):
        if self._tax_by_token is None:
            self._tax_by_token = load_taxonomy_by_token()
        return self.encoder(self._tax_by_token[raw])

    def zero_init_on_load(self):
        nn.init.zeros_(self.encoder.mlp[-1].weight)  # adjust to actual module structure
        nn.init.zeros_(self.encoder.mlp[-1].bias)


# @register_conditioner("geochemistry")
# class GeochemistryConditioner(EmbeddingConditioner):
#     input_key = "geochem_values"
#     collate_strategy= "fixed"
#     def __init__(self, hidden_size, num_features, use_gated_fusion=True, fusion_init_bias=-6.0):
#         super().__init__(use_gated_fusion=use_gated_fusion, hidden_size=hidden_size,
#                           fusion_init_bias=fusion_init_bias)
#         self.tokenizer = GeochemFeatureTokenizer(num_features, hidden_size)
#     def encode(self, raw, input_ids=None):
#         tokens, _mask = self.tokenizer(raw)
#         return tokens.mean(dim=1)  # pool to one additive vector per sample for this entry point


# --------------------------------------------------------------------------
# Mixin: wiring any two HF base classes up to the registry, once
# --------------------------------------------------------------------------
class ConditionedEmbeddingsMixin:
    def _init_conditioners(self, conditioners, hidden_size):
        self.conditioners = nn.ModuleDict()
        self._conditioner_order = []
        for name, cfg in (conditioners or {}).items():
            if name not in CONDITIONER_REGISTRY:
                raise KeyError(f"No conditioner registered under '{name}'. "
                               f"Available: {list(CONDITIONER_REGISTRY)}")
            cfg = dict(cfg)
            cfg.setdefault("hidden_size", hidden_size)
            self.conditioners[name] = CONDITIONER_REGISTRY[name](**cfg)
            self._conditioner_order.append(name)

    def apply_conditioning(self, embeddings, input_ids, **kwargs):
        for name in self._conditioner_order:
            conditioner = self.conditioners[name]
            raw = input_ids if conditioner.input_key == "input_ids" else kwargs.get(conditioner.input_key)
            if raw is None:
                continue  # conditioner configured but its input wasn't supplied this call
            embeddings = conditioner.condition(embeddings, raw, input_ids=input_ids)
        return embeddings

    def zero_init_conditioners_on_load(self):
        for module in self.conditioners.values():
            module.zero_init_on_load()

    def conditioner_logging_metrics(self):
        return {
            f"conditioner/{name}/{metric}": value
            for name, module in self.conditioners.items()
            for metric, value in module.logging_metrics().items()
        }
