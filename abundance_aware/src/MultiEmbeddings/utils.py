import ast

def build_conditioners_from_config(cfg):
    """Reads e.g.:
        [conditioners]
        enabled = abundance, hierarchy

        [conditioner:abundance]
        use_gated_fusion = true
        fusion_init_bias = -6.0

        [conditioner:hierarchy]
        use_gated_fusion = true
    """
    if "conditioners" not in cfg or not cfg.get("conditioners", "enabled", fallback="").strip():
        return {}

    names = [n.strip() for n in cfg.get("conditioners", "enabled").split(",") if n.strip()]
    spec = {}
    for name in names:
        section = f"conditioner:{name}"
        kwargs = {}
        if cfg.has_section(section):
            for key, raw_val in cfg.items(section):
                try:
                    kwargs[key] = ast.literal_eval(raw_val)
                except (ValueError, SyntaxError):
                    kwargs[key] = raw_val
        spec[name] = kwargs
    return spec