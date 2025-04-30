import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupNorm32(nn.GroupNorm):
    """
    Force x to be float32 before applying GroupNorm.
    Normalization layer can become unstable when using float16.
    This is a workaround for that.
    """
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
    

def normalization(channels: int):
    """
    Create a normalization layer.
    """
    return GroupNorm32(32, channels)