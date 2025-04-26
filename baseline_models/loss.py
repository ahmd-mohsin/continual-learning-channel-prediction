import torch
import torch.nn as nn

class NMSELoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse   = torch.mean((pred - target) ** 2)
        power = torch.mean(target ** 2)
        return mse / (power + self.eps)

criterion = NMSELoss()