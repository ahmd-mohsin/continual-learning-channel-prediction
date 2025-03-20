import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

class ChannelSequenceDataset(Dataset):
    def __init__(self, channel_data, device = "cpu"):
        if not torch.is_tensor(channel_data):
            # channel_data = torch.tensor(channel_data, dtype=torch.cdouble)
            channel_data = torch.tensor(channel_data)
        self.data = channel_data.to(device)  # Move to CUDA if available
        self.real = self.data.real.float()
        self.imag = self.data.imag.float()
        self.num_users = self.data.shape[0]
        self.time_length = self.data.shape[-1]
    
    def __len__(self):
        return self.num_users
    
    def __getitem__(self, idx):
        time_pairs = self.time_length - 1
        X = torch.zeros(time_pairs, 4, 18, 8, device=self.data.device)
        Y = torch.zeros(time_pairs, 4, 18, 8, device=self.data.device)
        
        for t in range(time_pairs):
            X_real_t = self.real[idx, :, :, :, t]
            X_imag_t = self.imag[idx, :, :, :, t]
            X[t] = torch.cat([X_real_t, X_imag_t], dim=0)
            
            Y_real_t = self.real[idx, :, :, :, t+1]
            Y_imag_t = self.imag[idx, :, :, :, t+1]
            Y[t] = torch.cat([Y_real_t, Y_imag_t], dim=0)
        
        return X, Y