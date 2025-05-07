from .custom_lr_schedulers import *
from hydra.utils import instantiate
from torch.optim import Optimizer


def build_lr_scheduler(cfg, **kwargs):
    """Return optimizer

    Args:
        cfg (_type_): optimizer configuration

    Returns:
        
    """
    if cfg == "None" or cfg == None:
        return None
    else:
        lr_scheduler = instantiate(cfg, **kwargs)
        return lr_scheduler