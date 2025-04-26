import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from positionalembedder import get_embedder

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
# 3) GRU Model (with ReLU + Dropout)
###############################################################################
class GRUModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1,
                 n_layers=3, H=18, W=8, dropout=0.5):
        super().__init__()
        self.H, self.W, self.n_layers, self.hidden_dim = H, W, n_layers, hidden_dim
        self.input_size = 2 * H * W  
        # apply dropout between GRU layers
        self.gru = nn.GRU(self.input_size, hidden_dim, n_layers,
                          batch_first=True, dropout=dropout)
        self.fc    = nn.Linear(hidden_dim, self.input_size)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        b, r, H, W, R, seq = x.shape
        x = x.permute(0,5,1,2,3,4).reshape(b, seq, -1)       # (b, seq, input_size)
        out, _ = self.gru(x)                                 # (b, seq, hidden_dim)
        out     = out[:, -1, :]                              # (b, hidden_dim)
        out     = self.fc(out)                               # (b, input_size)
        out     = self.relu(out)
        out     = self.drop(out)
        out     = out.view(b, r, H, W, R)                    # (b, 2, H, W, R)
        return out

###############################################################################
# 4) LSTM Model (with ReLU + Dropout)
###############################################################################
class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1,
                 n_layers=3, H=18, W=8, dropout=0.5):
        super().__init__()
        self.H, self.W, self.n_layers, self.hidden_dim = H, W, n_layers, hidden_dim
        self.input_size = 2 * H * W
        self.lstm  = nn.LSTM(self.input_size, hidden_dim, n_layers,
                             batch_first=True, dropout=dropout)
        self.fc    = nn.Linear(hidden_dim, self.input_size)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        b, r, H, W, R, seq = x.shape
        x = x.permute(0,5,1,2,3,4).reshape(b, seq, -1)
        out, _ = self.lstm(x)
        out     = out[:, -1, :]
        out     = self.fc(out)
        out     = self.relu(out)
        out     = self.drop(out)
        out     = out.view(b, r, H, W, R)
        return out

###############################################################################
# 5) Transformer Model (with ReLU + Dropout)
###############################################################################
class TransformerModel(nn.Module):
    def __init__(self, dim_val=128, n_heads=4,
                 n_encoder_layers=1, n_decoder_layers=1,
                 out_channels=4, H=18, W=16,
                 multires=6, dropout=0.1):
        super().__init__()
        self.H, self.W, self.out_ch = H, W, out_channels
        self.input_size = out_channels * H * W
        self.dim_val    = dim_val

        self.input_projection = nn.Linear(self.input_size, dim_val)
        self.pos_encoder      = PositionalEncoding(d_model=dim_val, multires=multires)
        self.transformer      = Transformer(
            d_model=dim_val, nhead=n_heads,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dropout=dropout, batch_first=True
        )
        self.fc_out = nn.Linear(dim_val, self.input_size)
        self.relu   = nn.ReLU()
        self.drop   = nn.Dropout(dropout)

    def forward(self, x):
        b, r, H, W, R, seq = x.shape
        # flatten spatial & channels
        src = x.permute(0,5,1,2,3,4).reshape(b, seq, -1)
        src = self.input_projection(src)
        src = self.pos_encoder(src)

        # prepare one-step target
        tgt = torch.zeros(b, 1, self.dim_val, device=src.device)
        tgt = self.pos_encoder(tgt)

        src_mask = generate_square_subsequent_mask(seq, seq).to(src.device)
        tgt_mask = generate_square_subsequent_mask(1, 1).to(src.device)

        out = self.transformer(src=src, tgt=tgt,
                               src_mask=src_mask,
                               tgt_mask=tgt_mask)      # (b,1,dim_val)
        out = out.squeeze(1)
        out = self.fc_out(out)
        out = self.relu(out)
        out = self.drop(out)
        out = out.view(b, r, H, W, R)
        return out

###############################################################################
# Positional Encoding (commonly used in Transformers)
###############################################################################
class PositionalEncoding(nn.Module):
    """
    Wraps the frequency-based embedder (from positionalembber.py)
    so that it can replace the old sinusoidal PositionalEncoding.
    """

    def __init__(self, d_model, multires=6):
        """
        Args:
            d_model (int): The 'model dimension' that the Transformer uses.
            multires (int): Number of frequency bands (L) for the embedder.
                            Increase/decrease as you like.
        """
        super(PositionalEncoding, self).__init__()

        # Build the embedder configured to take in dimension = d_model
        self.embedder, embed_dim = get_embedder(
            multires,           # e.g. 6 or 10
            input_dims=d_model, # we treat each 'd_model' channel as an input
            include_input=True
        )

        # If embedder's output dimension differs from d_model,
        # define a linear layer to project it back to d_model
        self.need_projection = (embed_dim != d_model)
        if self.need_projection:
            self.proj = nn.Linear(embed_dim, d_model)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model),
            with the frequency-based positional encodings.
        """
        # Apply the multi-frequency embedder
        # shape after embedder: (batch, seq_len, embed_dim)
        encoded = self.embedder(x)

        # If embed_dim != d_model, project back down
        if self.need_projection:
            encoded = self.proj(encoded)

        return encoded

