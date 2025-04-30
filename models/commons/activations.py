import torch
import torch.nn as nn
import torch.nn.functional as F


class SiLU(nn.Module):
    """
    Swish activation function.
    """
    def forward(self, x):
        return x * torch.sigmoid(x)
    

class Mish(nn.Module):
    """
    Mish activation function.
    """
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))