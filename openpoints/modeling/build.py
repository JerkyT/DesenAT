from openpoints.utils import registry
Modeling = registry.Registry('modeling')

def build_modeling_from_cfg(cfg, **kwargs):
    """
    Build a model, defined by `NAME`.
    Args:
        cfg (eDICT): 
    Returns:
        Model: a constructed model specified by NAME.
    """
    if cfg:
        return Modeling.build(cfg, **kwargs)
    else:
        return False