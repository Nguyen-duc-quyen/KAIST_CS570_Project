from .custom_optimizers import *
from hydra.utils import instantiate


def build_optimizer(cfg, params):
    """Return optimizer

    Args:
        cfg (_type_): optimizer configuration

    Returns:
        
    """
    optimizer = instantiate(cfg, params=params)
    return optimizer