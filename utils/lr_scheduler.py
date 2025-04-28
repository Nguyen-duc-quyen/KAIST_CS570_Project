"""Learning rate scheduler"""
from torch.optim.lr_scheduler import _LRScheduler


class MultiStepLRWithWarmUp(_LRScheduler):
    def __init__(self, optimizer, base_lr, num_warmup_steps, decay_steps, gamma=0.1, last_epoch=-1): 
        self.base_lr = base_lr
        self.num_warmup_steps = num_warmup_steps
        self.decay_steps = decay_steps
        self.gamma = gamma
        self.alpha = 1.
        super(MultiStepLRWithWarmUp, self).__init__(optimizer, -1, False)
        self.last_epoch = last_epoch


    def get_lr(self):
        if self.last_epoch < self.num_warmup_steps: 
            self.alpha = self.gamma ** (self.num_warmup_steps - self.last_epoch - 1)
        elif self.last_epoch + 1 in self.decay_steps:
            self.alpha = self.alpha * self.gamma 
        return [self.base_lr * self.alpha for _ in self.optimizer.param_groups]