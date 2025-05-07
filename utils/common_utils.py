import torch


def unnormalize(tensor: torch.Tensor, mean: list, std: list):
    """
        Unnormalize the tensor given the mean and std
        This is commonly use during the inference process
    """
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
    return tensor*std + mean