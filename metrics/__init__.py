import torcheval.metrics as metrics
from hydra.utils import instantiate


def build_metrics(cfg):
    """Return list of torcheval and custom metrics from hydra config

    Args:
        cfg (_type_): metrics configuration

    Returns:
        List[torcheval.Metrics]: list of use metrics
    """
    metrics = [instantiate(metric) for metric in cfg]
    return metrics