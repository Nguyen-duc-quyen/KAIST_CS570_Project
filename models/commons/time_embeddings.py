import numpy as np
import torch
import torch.nn as nn
import math
from abc import abstractmethod


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Compute sinusoidal timestep embeddings.
    Args:
        timesteps (torch.Tensor): The input timesteps.
        dim (int): The dimension of the output embeddings.
    Returns:
        torch.Tensor: The sinusoidal timestep embeddings.
    """
    half = dim // 2
    max_period = torch.tensor(max_period, dtype=timesteps.dtype, device=timesteps.device)
    freqs = torch.exp(
        -torch.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x