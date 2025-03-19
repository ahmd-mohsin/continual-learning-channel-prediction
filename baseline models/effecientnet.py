import torch
import torch.nn as nn
from torchvision import models

class EfficientLSTMChannelPredictor(nn.Module):
    """
    EfficientNet-b0 + LSTM model for channel prediction.
    
    Input:
      - x: shape (batch, time_steps, 4, 18, 8)
             4 channels represent [2 (Rx antennas) x (real & imag)]
      
    Output:
      - prediction: shape (batch, 4, 18, 8)
             This can be later split into real and imaginary parts if desired.
    """
    def __init__(self, in_channels=4, out_channels=4, hidden_dim=512, num_layers=2, pretrained=False):
        super(EfficientLSTMChannelPredictor, self).__init__()
        
        # Load EfficientNet-b0 backbone and modify its first conv layer
        efficient_net = models.efficientnet_b0(pretrained=pretrained)
        self.feature_extractor = efficient_net.features
        
        # Modify the first convolution layer to accept 'in_channels' instead of 3.
        # In EfficientNet-b0, the first block is usually accessed as self.feature_extractor[0],
        # and its first layer is the convolution.
        first_conv = self.feature_extractor[0][0]
        self.feature_extractor[0][0] = nn.Conv2d(
            in_channels,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False
        )
        
        # Determine the dimension of the features by forwarding a dummy input.
        dummy_input = torch.randn(1, in_channels, 18, 8)
        with torch.no_grad():
            dummy_features = self.feature_extractor(dummy_input)
        feature_dim = dummy_features.view(1, -1).shape[-1]
        
        # LSTM to capture temporal dependencies.
        self.lstm = nn.LSTM(input_size=feature_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)
        
        # Fully connected layer maps the LSTM hidden state to the predicted channel matrix.
        self.fc = nn.Linear(hidden_dim, out_channels * 18 * 8)
    
    def forward(self, x):
        """
        x shape: (batch, time_steps, in_channels, 18, 8)
        returns shape: (batch, 4, 18, 8)
        """
        batch, time_steps, C, H, W = x.shape
        # Combine batch and time dimensions to process each time step through the CNN.
        x = x.view(batch * time_steps, C, H, W)
        features = self.feature_extractor(x)
        features = features.view(batch, time_steps, -1)  # shape: (batch, time_steps, feature_dim)
        
        # Feed the sequence of features into the LSTM.
        lstm_out, _ = self.lstm(features)  # lstm_out: (batch, time_steps, hidden_dim)
        
        # Use the last time step's output for prediction.
        final_output = lstm_out[:, -1, :]  # shape: (batch, hidden_dim)
        prediction = self.fc(final_output)  # shape: (batch, out_channels * 18 * 8)
        prediction = prediction.view(batch, 4, 18, 8)
        return prediction
