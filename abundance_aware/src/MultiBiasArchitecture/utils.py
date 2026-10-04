import ast

from abundance_aware.src.MultiBiasArchitecture.common.RegisterEncoder import ENCODER_REGISTRY


def attach_bias_spec_to_config(config, cfg_ini_section_name="biases"):
    """Call once at model construction. Stores the serializable spec (encoder
    names, not instances) on config so from_pretrained can rebuild identical
    bias modules without re-reading the original .ini."""
    # bias_specs at this point has live encoder instances in it; store the
    # by-name form alongside for serialization
    config.bias_spec_source = cfg_ini_section_name  # or inline the serializable dict directly

def build_bias_specs_from_config(cfg, hidden_size, num_heads):
    """Reads e.g.:
        [biases]
        enabled = abundance, hierarchy

        [bias:abundance]
        rank = 16
        encoder = abundance_film
        use_gated_fusion = true
        init_scale = 0.0

        [bias:hierarchy]
        rank = 16
        encoder = taxonomy_hierarchy
        use_gated_fusion = true
        init_scale = 0.0

    Returns {name: kwargs_dict} with actual encoder INSTANCES resolved —
    one shared instance per encoder name, reused across every transformer
    layer's bias module (matches the original hand-wired pattern).
    """
    if "biases" not in cfg or not cfg.get("biases", "enabled", fallback="").strip():
        return {}

    names = [n.strip() for n in cfg.get("biases", "enabled").split(",") if n.strip()]
    encoder_cache = {}  # encoder name -> shared instance, so multiple biases can reuse one
    specs = {}

    for name in names:
        section = f"bias:{name}"
        if not cfg.has_section(section):
            raise KeyError(f"Bias '{name}' listed in [biases].enabled but has no [{section}] section")

        kwargs = {"hidden_size": hidden_size, "num_heads": num_heads}
        for key, raw_val in cfg.items(section):
            if key == "encoder":
                if raw_val not in ENCODER_REGISTRY:
                    raise KeyError(f"No encoder registered under '{raw_val}'. Available: {list(ENCODER_REGISTRY)}")
                if raw_val not in encoder_cache:
                    encoder_cache[raw_val] = ENCODER_REGISTRY[raw_val](hidden_size)
                kwargs["encoder"] = encoder_cache[raw_val]
            else:
                try:
                    kwargs[key] = ast.literal_eval(raw_val)
                except (ValueError, SyntaxError):
                    kwargs[key] = raw_val
        specs[name] = kwargs

    return specs