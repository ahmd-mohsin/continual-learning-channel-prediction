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





def masked_nmse_per_sample(pred_mag, true_mag, true_mask, eps=1e-8):
    """
    Compute NMSE **per sample** in the batch.
    
    Args:
      pred_mag   (B,R,S,T): predicted magnitudes
      true_mag   (B,R,S,T): ground‐truth magnitudes
      true_mask  (B,R,S,T): binary mask (1=valid)
      eps: small constant to avoid division by zero
    
    Returns:
      nmse       (B,)      : normalized MSE for each sample
    """
    # squared error, masked
    se = (pred_mag - true_mag).pow(2) * true_mask                # (B,R,S,T)
    # sum over R,S,T for each batch element
    se_sum    = se.sum(dim=(1,2,3))                             # (B,)
    mask_sum  = true_mask.sum(dim=(1,2,3)) + eps                # (B,)
    
    # mean‐squared‐error per sample
    mse       = se_sum / mask_sum                               # (B,)
    
    # signal power per sample
    power_num = (true_mag.pow(2) * true_mask).sum(dim=(1,2,3))   # (B,)
    power     = power_num / mask_sum                            # (B,)
    
    # normalized MSE
    nmse      = mse / (power + eps)                             # (B,)
    return nmse



def masked_nmse(pred_mag, true_mag, true_mask, eps=1e-8):
    """
    NMSE computed **only** where mask==1.
    pred_mag, true_mag, true_mask : (B,R,S,T)
    """
    se     = (pred_mag - true_mag).pow(2) * true_mask
    mse    = se.sum()    / (true_mask.sum() + eps)
    power  = (true_mag.pow(2) * true_mask).sum() / (true_mask.sum() + eps)
    return mse / (power + eps)