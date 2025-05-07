from .samplers import *
from hydra.utils import instantiate


def build_timestep_scheduler(cfg, **kwargs):
    timestep_scheduler = instantiate(cfg, **kwargs)
    return timestep_scheduler