from .custom_losses import *
from hydra.utils import instantiate


def build_loss_fc(cfg):
    """Return list of torcheval and custom metrics from hydra config

    Args:
        cfg (_type_): metrics configuration

    Returns:
        List[torcheval.Metrics]: list of use metrics
    """
    loss_fc = instantiate(cfg)
    return loss_fc