import torch
import torch.nn as nn

class NMSELoss(nn.Module):
    """
    Normalized MSE loss.
    - reduction='none': returns a tensor of shape (batch, *spatial_dims)
      containing the squared error normalized by each sample's power.
    - reduction='mean': returns a single scalar = mean(sample_nmse).
    - reduction='sum' : returns a single scalar = sum(sample_nmse).
    """
    def __init__(self, eps: float = 1e-8, reduction: str = 'mean'):
        super().__init__()
        self.eps       = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # element‐wise squared error
        se = (pred - target).pow(2)                                       # (B, …)
        # element‐wise target power
        tp = target.pow(2)                                                # (B, …)

        # compute per‐sample MSE and power
        B = pred.size(0)
        se_flat = se.view(B, -1)
        tp_flat = tp.view(B, -1)
        mse_per_sample   = se_flat.mean(1)                                # (B,)
        power_per_sample = tp_flat.mean(1).add(self.eps)                  # (B,)
        nmse_per_sample  = mse_per_sample / power_per_sample              # (B,)

        if self.reduction == 'none':
            # we need to return element‐wise normalized error so that
            # your existing `.view(B, -1).mean(1)` still works:
            #   (se / power) has shape (B, …)
            power_reshaped = power_per_sample.view([B] + [1]*(se.dim()-1))
            return se.div(power_reshaped)                                 # (B, …)

        if self.reduction == 'sum':
            return nmse_per_sample.sum()                                  # scalar

        # default: 'mean'
        return nmse_per_sample.mean()                                     # scalar
