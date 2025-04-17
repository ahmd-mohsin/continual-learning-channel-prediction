import torch
import torch.nn as nn

import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        Custom Loss combining MSE and F1 score loss
        alpha: The weight given to MSE loss (0 <= alpha <= 1)
        """
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()

    def f1_loss(self, y_pred, y_true, epsilon=1e-6):
        """
        Calculate F1 loss (negative F1 score)
        """
        # Flatten the tensors for simplicity
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        # Calculate Precision and Recall
        true_positive = torch.sum(y_true * y_pred)
        predicted_positive = torch.sum(y_pred)
        actual_positive = torch.sum(y_true)

        precision = true_positive / (predicted_positive + epsilon)
        recall = true_positive / (actual_positive + epsilon)

        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)

        # F1 loss is the negative F1 score, since we want to minimize loss
        return 1 - f1

    def forward(self, y_pred, y_true):
        """
        Combine MSE loss and F1 loss
        """
        mse = self.mse_loss(y_pred, y_true)
        f1 = self.f1_loss(y_pred, y_true)
        
        # Combine both losses
        return self.alpha * mse + (1 - self.alpha) * f1
