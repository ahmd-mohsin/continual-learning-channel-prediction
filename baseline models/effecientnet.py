import torch
import torch.nn as nn
from torchvision import models

class EfficientLSTMChannelPredictor(nn.Module):
    """
    EfficientNet-b0 + LSTM model for channel prediction.
    
    Input:
      - x: shape (batch, time_steps, 4, 18, 8)
             where 4 channels represent [2 (Rx antennas) x (real & imag)]
      
    Output:
      - prediction: shape (batch, 4, 18, 8)
             (can be later split into real and imaginary parts if desired)
    """
    def __init__(self, in_channels=4, out_channels=4, hidden_dim=512, num_layers=2, pretrained=False):
        super(EfficientLSTMChannelPredictor, self).__init__()

        efficient_net = models.efficientnet_b0(pretrained=pretrained)
        self.feature_extractor = efficient_net.features

        first_conv = self.feature_extractor[0][0]
        self.feature_extractor[0][0] = nn.Conv2d(
            in_channels,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False
        )
        feature_dim = 320
        
        # LSTM for temporal modeling over the full sequence.
        self.lstm = nn.LSTM(input_size=feature_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_channels * 18 * 8)
    
    def forward(self, x):
        """
        x shape: (batch, time_steps, 4, 18, 8)
        Returns:
          - prediction of shape: (batch, 4, 18, 8)
        """
        batch, time_steps, C, H, W = x.shape
        
        x = x.view(batch * time_steps, C, H, W)  # shape: (batch*time_steps, 4, 18, 8)
        features = self.feature_extractor(x)
        features = features.view(batch, time_steps, -1)  # shape: (batch, time_steps, feature_dim)
        
        lstm_out, _ = self.lstm(features)  # lstm_out: (batch, time_steps, hidden_dim)
        final_output = lstm_out[:, -1, :]  # shape: (batch, hidden_dim)
        prediction = self.fc(final_output)  # shape: (batch, out_channels * 18 * 8)
        prediction = prediction.view(batch, 4, 18, 8)  # Reshape to (batch, 4, 18, 8)
        return prediction