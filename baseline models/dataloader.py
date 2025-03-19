import torch
from torch.utils.data import Dataset, DataLoader

class ChannelDataset(Dataset):
    def __init__(self, channel_data):
        """
        channel_data: a NumPy array or Torch tensor of shape (256, 2, 18, 8, 3000),
                      representing [user, rx_ant, RB, tx_ant, time].
        """
        if isinstance(channel_data, torch.Tensor):
            self.data = channel_data  
        else:
            self.data = torch.tensor(channel_data, dtype=torch.cdouble) 
        
        self.real = self.data.real.float()
        self.imag = self.data.imag.float()
        
        self.num_users = self.data.shape[0]    
        self.time_length = self.data.shape[-1] 
        self.total_samples = self.num_users * (self.time_length - 1)
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        user_idx = idx // (self.time_length - 1)
        t = idx % (self.time_length - 1)
        X_real = self.real[user_idx, :, :, :, t]   # shape (2, 18, 8)
        X_imag = self.imag[user_idx, :, :, :, t]   # shape (2, 18, 8)
        Y_real = self.real[user_idx, :, :, :, t+1] # shape (2, 18, 8)
        Y_imag = self.imag[user_idx, :, :, :, t+1] # shape (2, 18, 8)
        X = torch.cat([X_real, X_imag], dim=0)
        Y = torch.cat([Y_real, Y_imag], dim=0)
        return X, Y

