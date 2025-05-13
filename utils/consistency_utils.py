import torch
import math


def get_noise_preconditioning(sigmas, noise_precond_type: str = "cm"):
    """
    Calculates the noise preconditioning function c_noise, which is used to transform the raw Karras sigmas into the
    timestep input for the U-Net.
    """
    if noise_precond_type == "none":
        return sigmas
    elif noise_precond_type == "edm":
        return 0.25 * torch.log(sigmas)
    elif noise_precond_type == "cm":
        return 1000 * 0.25 * torch.log(sigmas + 1e-44)
    else:
        raise ValueError(
            f"Noise preconditioning type {noise_precond_type} is not current supported. Currently supported noise"
            f" preconditioning types are `none` (which uses the sigmas as is), `edm`, and `cm`."
        )


def get_input_preconditioning(sigmas, sigma_data=0.5, input_precond_type: str = "cm"):
    """
    Calculates the input preconditioning factor c_in, which is used to scale the U-Net image input.
    """
    if input_precond_type == "none":
        return torch.ones_like(sigmas)
    elif input_precond_type == "cm":
        return 1.0 / (sigmas**2 + sigma_data**2)
    else:
        raise ValueError(
            f"Input preconditioning type {input_precond_type} is not current supported. Currently supported input"
            f" preconditioning types are `none` (which uses a scaling factor of 1.0) and `cm`."
        )
        
        
def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=1.0):
    scaled_timestep = timestep_scaling * timestep
    c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
    c_out = scaled_timestep / (scaled_timestep**2 + sigma_data**2) ** 0.5
    return c_skip, c_out


def add_noise(original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor):
    # Make sure timesteps (Karras sigmas) have the same device and dtype as original_samples
    sigmas = timesteps.to(device=original_samples.device, dtype=original_samples.dtype)
    while len(sigmas.shape) < len(original_samples.shape):
        sigmas = sigmas.unsqueeze(-1)

    noisy_samples = original_samples + noise * sigmas

    return noisy_samples


def get_discretized_lognormal_weights(noise_levels: torch.Tensor, p_mean: float = -1.1, p_std: float = 2.0):
    """
    Calculates the unnormalized weights for a 1D array of noise level sigma_i based on the discretized lognormal"
    " distribution used in the iCT paper (given in Equation 10).
    """
    upper_prob = torch.special.erf((torch.log(noise_levels[1:]) - p_mean) / (math.sqrt(2) * p_std))
    lower_prob = torch.special.erf((torch.log(noise_levels[:-1]) - p_mean) / (math.sqrt(2) * p_std))
    weights = upper_prob - lower_prob
    return weights


def get_loss_weighting_schedule(noise_levels: torch.Tensor):
    """
    Calculates the loss weighting schedule lambda given a set of noise levels.
    """
    return 1.0 / (noise_levels[1:] - noise_levels[:-1])


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def get_discretization_steps(global_step: int, max_train_steps: int, s_0: int = 10, s_1: int = 1280, constant=False):
    """
    Gradully increases the number of discretization steps from s_0 to s_1 over the course of training.
    Calculates the current discretization steps at global step k using the discretization curriculum N(k).
    """
    if constant:
        return s_0 + 1

    k_prime = math.floor(max_train_steps / (math.log2(math.floor(s_1 / s_0)) + 1))
    num_discretization_steps = min(s_0 * 2 ** math.floor(global_step / k_prime), s_1) + 1

    return num_discretization_steps


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def add_noise(original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor):
    # Make sure timesteps (Karras sigmas) have the same device and dtype as original_samples
    sigmas = timesteps.to(device=original_samples.device, dtype=original_samples.dtype)
    while len(sigmas.shape) < len(original_samples.shape):
        sigmas = sigmas.unsqueeze(-1)

    noisy_samples = original_samples + noise * sigmas

    return noisy_samples