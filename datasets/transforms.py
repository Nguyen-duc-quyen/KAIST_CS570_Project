import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
    
    
class ScaleLatent(A.ImageOnlyTransform):
    def __init__(self, scale=1.0, always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.scale = scale

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        # img could be any numpy array (e.g. latent tensor on CPU)
        return img * self.scale

    def get_transform_init_args_names(self):
        # ensures `self.scale` is serialized in .to_dict()
        return ("scale",)