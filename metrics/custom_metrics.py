from torcheval.metrics import Metric
import torch


class MeanSquaredError(Metric):
    """
        An Example of how to create a customized torcheval metric
    """
    def __init__(self):
        super().__init__()
        self._add_state("sum_squared_error", torch.tensor(0.0), dist_reduce_fx="sum")
        self._add_state("count", torch.tensor(0.0), dist_reduce_fx="sum")
        
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update the metric states with new data.

        Args:
            predictions (torch.Tensor): The predicted values
            targets (torch.Tensor): The true values
        """
        squared_error = (predictions - targets)**2
        self.sum_squared_error += squared_error.sum()
        self.count += targets.numel()
        
    
    def compute(self) -> torch.Tensor:
        """Compute the final metric value

        Returns:
            torch.Tensor: the final value of the metric
        """
        return self.sum_squared_error/self.count
    

class BinarySensitivity(Metric):
    def __init__(self):
        super().__init__()
        self._add_state("true_positive", torch.tensor(0.0))
        self._add_state("false_negative", torch.tensor(0.0))

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update the states with predictions and targets.
        Args:
            preds (torch.Tensor): Binary predictions (0 or 1).
            targets (torch.Tensor): Ground-truth labels (0 or 1).
        """
        preds = preds.to(torch.int)
        targets = targets.to(torch.int)
        if self.true_positive.device != preds.device:
            self.true_positive = self.true_positive.to(preds.device)
                
        if self.false_negative.device != preds.device:
            self.false_negative = self.false_negative.to(preds.device)
        
        self.true_positive += torch.sum((preds == 1) & (targets == 1)).float()
        self.false_negative += torch.sum((preds == 0) & (targets == 1)).float()

    def compute(self) -> torch.Tensor:
        """
        Compute the sensitivity.
        """
        return self.true_positive / (self.true_positive + self.false_negative + 1e-10)

    def merge_state(self, metrics):
        for metric in metrics:
            self.true_positive += metric.true_positive
            self.false_negative += metric.false_negative
        return self


class BinarySpecificity(Metric):
    def __init__(self):
        super().__init__()
        self._add_state("true_negative", torch.tensor(0.0))
        self._add_state("false_positive", torch.tensor(0.0))

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update the states with predictions and targets.
        Args:
            preds (torch.Tensor): Binary predictions (0 or 1).
            targets (torch.Tensor): Ground-truth labels (0 or 1).
        """
        preds = preds.to(torch.int)
        targets = targets.to(torch.int)
        
        if self.true_negative.device != preds.device:
            self.true_negative = self.true_negative.to(preds.device)
                
        if self.false_positive.device != preds.device:
            self.false_positive = self.false_positive.to(preds.device)
        
        self.true_negative += torch.sum((preds == 0) & (targets == 0))
        self.false_positive += torch.sum((preds == 1) & (targets == 0))

    def compute(self) -> torch.Tensor:
        """
        Compute the specificity.
        """
        return self.true_negative / (self.true_negative + self.false_positive + 1e-10)
    
    def merge_state(self, metrics):
        for metric in metrics:
            self.true_negative += metric.true_negative
            self.false_positive += metric.false_positive
        return self