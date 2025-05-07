from .custom_optimizers import *
from hydra.utils import instantiate


def build_optimizer(cfg, **kwargs):
    """Return optimizer

    Args:
        cfg (_type_): optimizer configuration

    Returns:
        
    """
    optimizer = instantiate(cfg, **kwargs)
    return optimizer