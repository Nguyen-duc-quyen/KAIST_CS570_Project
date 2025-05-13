import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
        
        
class PseudoHuberLoss(nn.Module):
    """The Pseudo-Huber loss."""

    reductions = {'mean': torch.mean, 'sum': torch.sum, 'none': lambda x: x}
    
    def __init__(self, beta=1, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def extra_repr(self):
        return f'beta={self.beta:g}, reduction={self.reduction!r}'

    def forward(self, input, target):
        output = self.beta**2 * input.sub(target).div(self.beta).pow(2).add(1).sqrt().sub(1)
        return self.reductions[self.reduction](output)
    
    
class CustomPseudoHuberLoss(nn.Module):
    """
        The Pseudo-Huber Custom loss.
        This follows the implementation in the Consistency Model Training code
    """

    reductions = {'mean': torch.mean, 'sum': torch.sum, 'none': lambda x: x}
    
    def __init__(self, huber_c=1, reduction='mean'):
        super().__init__()
        self.huber_c = huber_c
        self.reduction = reduction

    def extra_repr(self):
        return f'beta={self.beta:g}, reduction={self.reduction!r}'

    def forward(self, input, target, weights=1.0):
        output = torch.sqrt((input-target)**2 + self.huber_c**2) - self.huber_c
        output = output * weights
        return self.reductions[self.reduction](output)