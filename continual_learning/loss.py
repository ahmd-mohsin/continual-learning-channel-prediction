import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.loss import _Loss

class NMSELoss(_Loss):
    """
    Normalized Mean Squared Error Loss.

    NMSE = (pred - target)^2 / ( E[target^2] + eps )

    Args:
        reduction (str): 'none' | 'mean' | 'sum' (default: 'mean')
        eps (float): small constant to avoid divide-by-zero (default: 1e-8)

    Shape:
        - Input: (N, *) where * means any number of additional dimensions
        - Target: same shape as input
    """
    def __init__(self, reduction: str = 'mean', eps: float = 1e-8):
        super().__init__(reduction=reduction)
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got {reduction!r}")
        self.eps = eps

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # element-wise squared error
        sq_err = (pred - target).pow(2)
        # scalar: average power of target over all elements
        power = target.pow(2).mean()
        # normalized error
        nmse = sq_err / (power + self.eps)

        if self.reduction == 'none':
            return nmse
        elif self.reduction == 'sum':
            return nmse.sum()
        else:  # 'mean'
            return nmse.mean()
