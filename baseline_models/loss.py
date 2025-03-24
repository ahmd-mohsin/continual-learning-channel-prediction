import torch
import torch.nn as nn

class NormalizedMSELoss(nn.Module):
    def __init__(self, eps=1e-12):
        super(NormalizedMSELoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        numerator = torch.sum((y_true - y_pred) ** 2)
        denominator = torch.sum(y_true ** 2) + self.eps
        return numerator / denominator



class ScaledMSELoss(nn.Module):
    def __init__(self, min_value=-1e-5, max_value=1e-5, scale_factor=10.0, l1_weight=0.5):
        super(ScaledMSELoss, self).__init__()
        self.min_value = min_value  # Minimum value for normalization (adjust if needed)
        self.max_value = max_value  # Maximum value for normalization (adjust if needed)
        self.scale_factor = scale_factor  # Factor to scale loss to the range 0-10
        self.l1_weight = l1_weight  # Weight for the L1 loss component

    def forward(self, predictions, targets):
        # Compute the Mean Squared Error (MSE)
        mse_loss = torch.mean((predictions - targets) ** 2)

        # Compute the L1 loss
        l1_loss = torch.mean(torch.abs(predictions - targets))

        # Normalize the MSE loss
        mse_loss_normalized = (mse_loss - self.min_value) / (self.max_value - self.min_value)

        # Scale the normalized MSE loss to the range [0, 10]
        scaled_mse_loss = mse_loss_normalized * self.scale_factor

        # Combine the MSE and L1 losses
        total_loss = scaled_mse_loss + self.l1_weight * l1_loss

        return total_loss
    


class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, predictions, targets):
        # Calculate the difference between predictions and targets
        diff = predictions - targets
        
        # Calculate log(cosh(diff))
        log_cosh = torch.log(torch.cosh(diff))
        
        # Return the mean of the log_cosh values
        return torch.mean(log_cosh)



class SigmoidLoss(nn.Module):
    def __init__(self, scale_factor=10.0):
        super(SigmoidLoss, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, predictions, targets):
        # Calculate the absolute error between predictions and targets
        abs_error = torch.abs(predictions - targets)
        # Apply a sigmoid function to the error
        sigmoid_loss = torch.sigmoid(abs_error)
        # Scale the loss to oscillate between 0 and 10
        scaled_loss = sigmoid_loss * self.scale_factor
        return scaled_loss.mean()  # Return the average loss
