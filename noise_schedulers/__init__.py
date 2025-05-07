from .karras_schedulers import *
from hydra.utils import instantiate
import diffusers


def build_noise_scheduler(cfg, **kwargs):
    """
        Build diffusion scheduler from configs
    """
    scheduler = instantiate(cfg, **kwargs)
    return scheduler