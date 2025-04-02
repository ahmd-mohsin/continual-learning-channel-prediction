# import torch
# import torch.nn as nn

# class CustomLSTMModel(nn.Module):
#     def __init__(self, input_size = 1, hidden_size = 32, num_layers = 3, output_size = 1):
#         super(CustomLSTMModel, self).__init__()
        
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
    
#     def forward(self, x):
#         batch_size, channels, height, width, time_steps = x.shape  # (batch, 2, 18, 8, 3000)
        
#         # Reshape to feed into LSTM
#         x = x.view(batch_size * channels * height * width, time_steps, -1)  # (batch * 2 * 18 * 8, 3000, feature_dim)
#         lstm_out, _ = self.lstm(x)
#         lstm_out = lstm_out[:, -1, :]  # (batch * 2 * 18 * 8, hidden_size)
#         out = self.fc(lstm_out)  # (batch * 2 * 18 * 8, output_size)
#         out = out.view(batch_size, channels, height, width)  # (batch, 2, 18, 8) 
#         return out

# def load_model(model_path, device):
#     model = CustomLSTMModel().to(device)
#     checkpoint = torch.load(model_path, map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
#     print(f"Loaded trained model from {model_path}")
#     return model


import torch
import torch.nn as nn
import math

###############################################################################
# Helper function for Transformer masks (like in the notebook)
###############################################################################
def generate_square_subsequent_mask(dim1, dim2):
    """
    Generate a square mask for the sequence so that the model 
    does not look ahead.
    """
    mask = (torch.triu(torch.ones(dim1, dim2)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 1, float('-inf'))
    return mask

###############################################################################
# 1) MLP Model
###############################################################################
class MLPModel(nn.Module):
    """
    Predict the entire channel matrix after flattening.
    - Input shape: (batch, 2, H, W, seq_len)
    - Output shape: (batch, 2, H, W)
    """
    def __init__(self, input_dim=16, hidden_dim=128, H=18, W=8):
        """
        Args:
            input_dim: how many features per time-step 
                       (we'll often flatten (2, H, W) => 2*H*W if we want each time step as a single vector).
            hidden_dim: size of hidden layers in MLP
            H, W: height and width of your channel matrix
        """
        super(MLPModel, self).__init__()
        self.H = H
        self.W = W
        # We'll define a simple 2-layer MLP to illustrate. 
        # Adjust depth/width as desired.
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2 * H * W)  # final output is 2×H×W (real + imag)

        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch, 2, H, W, seq_len)
        batch_size, two, H, W, seq_len = x.shape

        # Flatten all but the time dimension => treat each time step as a single scalar?
        # Alternatively, you might want (2*H*W) features per time step, then sum over seq_len.
        # For a basic approach, let's flatten the entire 5D input:
        #   => (batch_size, 2*H*W*seq_len)
        x = x.view(batch_size, -1)  # (batch, 2*H*W*seq_len)

        # MLP forward
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # (batch, 2*H*W)

        # Reshape to (batch, 2, H, W)
        x = x.view(batch_size, 2, H, W)
        return x


###############################################################################
# 2) CNN Model
###############################################################################
class CNNModel(nn.Module):
    """
    Simple CNN-based approach for channel prediction.
    - We can treat seq_len as a 'temporal' dimension, so we do a 1D convolution across time
      or do 2D conv across (H, W) for each time slice. 
    - Output shape: (batch, 2, H, W)
    """
    def __init__(self, in_channels=2, H=18, W=8, seq_len=16, hidden_channels=32):
        super(CNNModel, self).__init__()
        self.H = H
        self.W = W
        self.seq_len = seq_len

        # Example: Conv1D that slides over the time dimension, 
        # or we can do a 3D conv over (time, H, W). 
        # For simplicity, let's do a 2D conv over H,W for each time, then pool across time.

        # input shape we might re-map to: (batch, seq_len*2, H, W) 
        # or (batch, 2, seq_len, H, W).

        # Example: treat (2, seq_len) as channels:
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2 * seq_len, out_channels=hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Linear(hidden_channels * H * W, 2 * H * W)

    def forward(self, x):
        # x shape: (batch, 2, H, W, seq_len)
        batch_size, two, H, W, seq_len = x.shape

        # Rearrange to (batch, 2*seq_len, H, W)
        x = x.permute(0, 1, 4, 2, 3)  # (batch, 2, seq_len, H, W)
        x = x.reshape(batch_size, 2 * seq_len, H, W)  # combine 2 & seq_len

        x = self.conv(x)  # (batch, hidden_channels, H, W)
        x = x.view(batch_size, -1)  # flatten for fully connected
        x = self.fc(x)  # => (batch, 2*H*W)
        x = x.view(batch_size, 2, H, W)
        return x


###############################################################################
# 3) GRU Model (similar to code in your notebook)
###############################################################################
class GRUModel(nn.Module):
    """
    GRU model for channel prediction.
    - Input shape: (batch, 2, H, W, seq_len)
    - We flatten (2, H, W) as 'spatial channels' so that each time-step is a big vector
      dimension = 2*H*W
    - Output shape: (batch, 2, H, W)
    """
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1, n_layers=3, H=18, W=8):
        super(GRUModel, self).__init__()
        self.H = H
        self.W = W
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # GRU takes (batch, seq_len, input_size)
        # We want input_size = 2*H*W if we want the entire channel matrix as one vector
        self.input_size = 2 * H * W  
        self.gru = nn.GRU(self.input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2 * H * W)

    def forward(self, x):
        # x shape: (batch, 2, H, W, seq_len)
        batch_size, two, H, W, seq_len = x.shape

        # Flatten the matrix for each time step:
        # After flattening, each time step is a vector of size (2*H*W).
        # So we reshape into (batch, seq_len, 2*H*W).
        x = x.permute(0, 4, 1, 2, 3)  # => (batch, seq_len, 2, H, W)
        x = x.reshape(batch_size, seq_len, 2 * H * W)

        # Pass through GRU
        out, _ = self.gru(x)  # out shape: (batch, seq_len, hidden_dim)
        # Take the last time-step
        out = out[:, -1, :]  # (batch, hidden_dim)

        # Map to (2*H*W)
        out = self.fc(out)  # => (batch, 2*H*W)
        out = out.view(batch_size, 2, H, W)  # => (batch, 2, H, W)
        return out


###############################################################################
# 4) LSTM Model (similar to your existing code, but more explicit)
###############################################################################
class LSTMModel(nn.Module):
    """
    LSTM-based channel prediction.
    - Very similar to GRUModel, but we use LSTM instead of GRU.
    """
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1, n_layers=3, H=18, W=8):
        super(LSTMModel, self).__init__()
        self.H = H
        self.W = W
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.input_size = 2 * H * W
        self.lstm = nn.LSTM(self.input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2 * H * W)

    def forward(self, x):
        # x shape: (batch, 2, H, W, seq_len)
        batch_size, two, H, W, seq_len = x.shape

        # Flatten the matrix for each time step
        x = x.permute(0, 4, 1, 2, 3)  # => (batch, seq_len, 2, H, W)
        x = x.reshape(batch_size, seq_len, 2 * H * W)

        # Pass through LSTM
        out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        out = out[:, -1, :]    # (batch, hidden_dim)
        out = self.fc(out)     # (batch, 2*H*W)
        out = out.view(batch_size, 2, H, W)
        return out


###############################################################################
# 5) Transformer Model
###############################################################################
class TransformerModel(nn.Module):
    """
    A Transformer for time-series channel prediction.
    
    Expected input:
      x of shape (batch, out_channels, H, W, seq_len)
      
    This model:
      1. Flattens the (out_channels, H, W) dimensions into a vector of size:
             input_size = out_channels * H * W
      2. Projects each time step from input_size -> dim_val,
         and adds positional encoding.
      3. Uses the Transformer to decode a single time step.
      4. Projects back to input_size and reshapes to
             (batch, out_channels, H, W)
    
    Adjust out_channels, H, and W so that they match your target shape.
    In our example, the dataloader returns targets with shape (batch, 4, 18, 2)
    (i.e. 4 x 18 x 2 = 144 features).
    """
    def __init__(
        self, 
        dim_val=128,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        out_channels=2,   # default; for your data, use 4
        H=18,
        W=8,
        seq_len=16
    ):
        super(TransformerModel, self).__init__()
        self.out_channels = out_channels
        self.H = H
        self.W = W
        self.seq_len = seq_len

        # Derive flattened input size from out_channels, H, and W.
        self.input_size = out_channels * H * W  
        self.dim_val = dim_val

        # 1) Project input vector (size=input_size) to dimension dim_val.
        self.input_projection = nn.Linear(self.input_size, dim_val)
        # 2) Positional encoding.
        self.pos_encoder = PositionalEncoding(dim_val)
        # 3) Transformer block (with batch_first=True so that shape is (batch, seq_len, d_model)).
        self.transformer = nn.Transformer(
            d_model=dim_val,
            nhead=n_heads,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            batch_first=True
        )
        # 4) Map from dim_val back to the flattened vector of size input_size.
        self.fc_out = nn.Linear(dim_val, self.input_size)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, out_channels, H, W, seq_len)
        Returns:
            Tensor of shape (batch, out_channels, H, W)
        """
        batch_size = x.size(0)
        # Flatten the (out_channels, H, W) dimensions for each time step.
        # New shape: (batch, seq_len, out_channels * H * W)
        x = x.permute(0, 4, 1, 2, 3).reshape(batch_size, self.seq_len, -1)
        
        # Project each time step to dim_val and add positional encoding.
        src = self.input_projection(x)       # Shape: (batch, seq_len, dim_val)
        src = self.pos_encoder(src)

        # Create a target sequence of length 1 (for single-step prediction).
        tgt = torch.zeros(batch_size, 1, self.dim_val, device=x.device)
        tgt = self.pos_encoder(tgt)

        # Create masks: source mask for the entire sequence, target mask for a single token.
        src_mask = generate_square_subsequent_mask(self.seq_len, self.seq_len).to(x.device)
        tgt_mask = generate_square_subsequent_mask(1, 1).to(x.device)

        # Run the Transformer decoder.
        out = self.transformer(
            src=src,
            tgt=tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )  # Shape: (batch, 1, dim_val)

        # Map back to the flattened channel vector.
        out = self.fc_out(out).squeeze(1)  # Shape: (batch, input_size)
        # Reshape to (batch, out_channels, H, W)
        out = out.view(batch_size, self.out_channels, self.H, self.W)
        return out


###############################################################################
# Positional Encoding (commonly used in Transformers)
###############################################################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        Returns:
            Tensor with positional encodings added.
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]