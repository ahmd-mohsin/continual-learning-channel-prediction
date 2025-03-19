import torch
import torch.nn as nn
from torchvision import models

class MobileNetChannelPredictor(nn.Module):
    """
    MobileNetV2-based model for channel prediction.
    - Input:  (batch, 4, 18, 8) => 4 channels (2 rx antennas * real/imag).
    - Output: (batch, 576) => can reshape to (batch, 4, 18, 8).
    """
    def __init__(self, in_channels=4, out_features=576, pretrained=False):
        super(MobileNetChannelPredictor, self).__init__()
        
        # Load MobileNetV2 architecture
        self.model = models.mobilenet_v2(pretrained=pretrained)
        
        # Modify the first convolution layer to accept 'in_channels' instead of 3
        first_conv = self.model.features[0][0]  # initial Conv2d layer in 'ConvBNReLU'
        new_first_conv = nn.Conv2d(
            in_channels,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=(first_conv.bias is not None)
        )
        self.model.features[0][0] = new_first_conv
        
        # Modify the final classifier layer to have 'out_features' outputs
        old_linear = self.model.classifier[1]
        new_linear = nn.Linear(old_linear.in_features, out_features)
        self.model.classifier[1] = new_linear

    def forward(self, x):
        """
        x shape: (batch_size, in_channels=4, 18, 8)
        returns shape: (batch_size, out_features=576)
        """
        return self.model(x)
