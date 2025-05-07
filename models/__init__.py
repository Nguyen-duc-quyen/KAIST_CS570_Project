import torch
from hydra.utils import instantiate


def build_model(cfg, **kwargs):
    model = instantiate(cfg, **kwargs)
    return model