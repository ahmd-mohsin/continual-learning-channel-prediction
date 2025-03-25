import torch
import torch.nn as nn

class CustomLSTMModel(nn.Module):
    def __init__(self, input_size = 1, hidden_size = 32, num_layers = 3, output_size = 1):
        super(CustomLSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        batch_size, channels, height, width, time_steps = x.shape  # (batch, 2, 18, 8, 3000)
        
        # Reshape to feed into LSTM
        x = x.view(batch_size * channels * height * width, time_steps, -1)  # (batch * 2 * 18 * 8, 3000, feature_dim)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Get the last time step output
        lstm_out = lstm_out[:, -1, :]  # (batch * 2 * 18 * 8, hidden_size)
        
        # Fully connected layer to match output shape
        out = self.fc(lstm_out)  # (batch * 2 * 18 * 8, output_size)
        
        # Reshape to match desired output
        out = out.view(batch_size, channels, height, width)  # (batch, 2, 18, 8)
        
        return out

def load_model(model_path, device):
    model = CustomLSTMModel().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded trained model from {model_path}")
    return model
