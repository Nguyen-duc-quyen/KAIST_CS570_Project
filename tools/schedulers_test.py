# Import the DDIMScheduler class from the diffusers library
from diffusers import CMStochasticIterativeScheduler
import torch
import numpy as np

def get_karras_sigmas(
    num_discretization_steps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    dtype=torch.float32,
):
    """
    Calculates the Karras sigmas timestep discretization of [sigma_min, sigma_max].
    """
    ramp = np.linspace(0, 1, num_discretization_steps)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    # Make sure sigmas are in increasing rather than decreasing order (see section 2 of the iCT paper)
    sigmas = sigmas[::-1].copy()
    sigmas = torch.from_numpy(sigmas).to(dtype=dtype)
    return sigmas

def add_noise(original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor):
    # Make sure timesteps (Karras sigmas) have the same device and dtype as original_samples
    sigmas = timesteps.to(device=original_samples.device, dtype=original_samples.dtype)
    while len(sigmas.shape) < len(original_samples.shape):
        sigmas = sigmas.unsqueeze(-1)

    noisy_samples = original_samples + noise * sigmas

    return noisy_samples




scheduler = CMStochasticIterativeScheduler(
    sigma_min=0.002,
    sigma_max=80,
    sigma_data=0.5,
    num_train_timesteps=10,
    rho=7.0
)

print(len(scheduler.sigmas))
print(scheduler.sigmas)

current_timesteps = get_karras_sigmas(
    num_discretization_steps=10,
    sigma_min=0.002,
    sigma_max=80.0,
    rho=7.0,
    dtype=torch.float32,
)
skip_steps = 1

valid_teacher_timesteps_plus_one = current_timesteps[: len(current_timesteps) - skip_steps + 1]
print(valid_teacher_timesteps_plus_one)

