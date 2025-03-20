import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

class ChannelSequenceDataset(Dataset):
    def __init__(self, channel_data):
        """
        channel_data: a NumPy array or Torch tensor of shape (256, 2, 18, 8, 3000),
                      representing [user, rx_ant, RB, tx_ant, time].
                      
        For each user, we use the sequence from t=0 to t=2998 as input (after concatenating 
        real and imaginary parts along the channel dimension) and the channel at t=2999 as the target.
        """
        if not torch.is_tensor(channel_data):
            channel_data = torch.tensor(channel_data, dtype=torch.cdouble)
        self.data = channel_data  # shape: (num_users, 2, 18, 8, time_length)
        self.real = self.data.real.float()  # shape: (num_users, 2, 18, 8, time_length)
        self.imag = self.data.imag.float()  # shape: (num_users, 2, 18, 8, time_length)
        
        self.num_users = self.data.shape[0]
        self.time_length = self.data.shape[-1]
    
    def __len__(self):
        return self.num_users
    
    def __getitem__(self, idx):
        # Create pairs of consecutive samples for prediction
        # This sets up the (t-1) → t prediction task
        
        # We'll return multiple pairs across the time sequence
        time_pairs = self.time_length - 1
        X = torch.zeros(time_pairs, 4, 18, 8)
        Y = torch.zeros(time_pairs, 4, 18, 8)
        
        for t in range(time_pairs):
            # Input: channel at time t
            X_real_t = self.real[idx, :, :, :, t]  # (2, 18, 8)
            X_imag_t = self.imag[idx, :, :, :, t]  # (2, 18, 8)
            X[t] = torch.cat([X_real_t, X_imag_t], dim=0)  # (4, 18, 8)
            
            # Target: channel at time t+1
            Y_real_t = self.real[idx, :, :, :, t+1]  # (2, 18, 8)
            Y_imag_t = self.imag[idx, :, :, :, t+1]  # (2, 18, 8)
            Y[t] = torch.cat([Y_real_t, Y_imag_t], dim=0)  # (4, 18, 8)
        
        return X, Y