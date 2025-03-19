import torch
from torch.utils.data import Dataset, DataLoader

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
        
        X_real = self.real[idx, :, :, :, :-1]  
        X_imag = self.imag[idx, :, :, :, :-1]  
        X_real = X_real.permute(3, 0, 1, 2)
        X_imag = X_imag.permute(3, 0, 1, 2)
        X = torch.cat([X_real, X_imag], dim=1)
        s
        # Target: channel state at the final time sample t=T-1, shape (2, 18, 8) each for real and imag.
        Y_real = self.real[idx, :, :, :, -1]  # shape: (2, 18, 8)
        Y_imag = self.imag[idx, :, :, :, -1]  # shape: (2, 18, 8)
        Y = torch.cat([Y_real, Y_imag], dim=0)  # shape: (4, 18, 8)
        
        return X, Y

