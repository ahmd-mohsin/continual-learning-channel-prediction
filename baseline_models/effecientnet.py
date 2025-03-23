import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class LSTMChannelPredictor(nn.Module):
#     """
#     LSTM-based model for wireless channel prediction.
    
#     This model uses a combination of convolutional layers to extract spatial features
#     from the channel matrix, followed by LSTM layers to capture temporal dynamics.
    
#     Input:
#       - x: shape (batch, 4, 18, 8) representing a single channel state
#              where 4 channels represent [2 (Rx antennas) x (real & imag)]
      
#     Output:
#       - prediction: shape (batch, 4, 18, 8)
#              representing the predicted next channel state
#     """
#     def __init__(self, in_channels=4, out_channels=4, hidden_dim=256, num_layers=2):
#         super(LSTMChannelPredictor, self).__init__()
        
#         # CNN for spatial feature extraction from channel matrices
#         self.cnn_encoder = nn.Sequential(
#             nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.BatchNorm2d(64),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.BatchNorm2d(128),
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.BatchNorm2d(256)
#         )
        
#         self.feature_dim = 256 * 18 * 8
        
#         # LSTM for temporal modeling
#         self.lstm = nn.LSTM(
#             input_size=self.feature_dim,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             batch_first=True
#         )
        
    
#         self.decoder = nn.Sequential(
#             nn.Linear(hidden_dim, self.feature_dim),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.2)
#         )
        
       
#         self.cnn_decoder = nn.Sequential(
#             nn.Conv2d(256, 128, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.BatchNorm2d(128),
#             nn.Conv2d(128, 64, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.BatchNorm2d(64),
#             nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
#         )
    
#     def forward(self, x, hidden=None):
#         """
#         Forward pass for a single time step.
        
#         Args:
#             x: Input tensor of shape (batch, 4, 18, 8)
#             hidden: Optional hidden state for LSTM
            
#         Returns:
#             prediction: Output tensor of shape (batch, 4, 18, 8)
#             hidden: Updated hidden state for LSTM
#         """
#         batch_size = x.size(0)
        
        
#         features = self.cnn_encoder(x)
        
        
#         features = features.view(batch_size, 1, -1)
        
#         if hidden is None:
#             lstm_out, hidden = self.lstm(features)
#         else:
#             lstm_out, hidden = self.lstm(features, hidden)
        
#         # Decode LSTM output
#         decoded = self.decoder(lstm_out.reshape(batch_size, -1))
        
#         # Reshape back to channel dimensions
#         decoded = decoded.view(batch_size, 256, 18, 8)
        
#         # Final reconstruction
#         prediction = self.cnn_decoder(decoded)
        
#         return prediction, hidden

# Define LSTM Model
class CustomLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CustomLSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
        self.fc = nn.Linear(hidden_size, output_size).to(device)
    
    def forward(self, x):
        batch_size, channels, height, width, time_steps = x.shape  # (batch, 2, 18, 8, 500)
        
        # Move input tensor to device
        x = x.to(device)
        
        # Reshape to feed into LSTM
        x = x.view(batch_size * channels * height * width, time_steps, -1)  # (batch * 2 * 18 * 8, 3000, feature_dim)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Get the last time step output
        lstm_out = lstm_out[:, -1, :]  # (batch * 2 * 18 * 8, hidden_size)
        
        # Fully connected layer to match output shape
        out = self.fc(lstm_out)  # (batch * 2 * 18 * 8, output_size)
        
        # Reshape to match desired output
        out = out.view(batch_size, channels, width, height)  # (batch, 2, 8, 18)
        
        return out