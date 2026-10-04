from abundance_aware.src.common import TaxonomyHierarchyEncoder, AbundanceEncoder
from abundance_aware.src.common.AbundanceEncoder import AbundanceEncoderWithFiLM

ENCODER_REGISTRY = {}

def register_encoder(name):
    def decorator(cls):
        if name in ENCODER_REGISTRY:
            raise ValueError(f"Encoder '{name}' already registered")
        ENCODER_REGISTRY[name] = cls
        return cls
    return decorator

register_encoder("abundance_mlp")(AbundanceEncoder)
register_encoder("abundance_film")(AbundanceEncoderWithFiLM)
register_encoder("taxonomy_hierarchy")(TaxonomyHierarchyEncoder)