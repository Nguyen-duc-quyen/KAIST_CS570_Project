from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import torch.distributed as dist


class ScheduleSampler(ABC):
    """
    Abstract base class for schedule samplers.

    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights don't need to be normalized, however they should be positive.
        """
        return NotImplemented
    

    def sample(self, batch_size, device):
        """
        Sample timeseteps for a batch
        Args:
            batch_size (int): The number of timesteps to draw.
            device (torch.device): The device to sample on.
        Returns:
            timesteps: a tensor of timesteps indicies (batch_size,)
            weights: a tensor of weights to scale the resulting losses(batch_size,)
        """
        w = self.weights()
        p = w / np.sum(w) # Normalize weights
        indices_np = np.random.choice(len(p), size=batch_size, p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1/(len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights
    

class UniformSampler(ScheduleSampler):
    """
    Uniformly sample timesteps.
    """

    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_train_timesteps])

    def weights(self):
        return self._weights


class LossAwareSampler(ScheduleSampler):
    pass