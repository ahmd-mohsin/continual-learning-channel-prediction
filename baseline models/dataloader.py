import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ChannelDataset(Dataset):
    """
    A dataset that provides pairs (H(t), H(t+1)) for each user and time t.
    channel_matrix shape is assumed to be (num_users, rx_ant, nRB, tx_ant, time).
    """
    def __init__(self, channel_matrix):
        super().__init__()
        
        # channel_matrix: shape (256, 2, 18, 8, 3000), complex128
        real_data = channel_matrix.real  # shape (256, 2, 18, 8, 3000)   
        imag_data = channel_matrix.imag  # shape (256, 2, 18, 8, 3000)
        # user gives in batch
        # input is (batch, 2,18,8,3000), output is (batch, 2,8,18)
        # Stack real and imaginary along a new last dimension = 2
        # shape => (256, 2, 18, 8, 3000, 2)
        self.data = torch.from_numpy(
            np.stack([real_data, imag_data], axis=-1)
        ).float()
        self.num_users = self.data.shape[0]  # 256
        self.num_times = self.data.shape[4]  # 3000
        
        self.indices = []
        for u in range(self.num_users):             
            for t in range(self.num_times - 1):     
                self.indices.append((u, t))

    def __len__(self):
        # Total number of (user, time) pairs
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Returns a tuple: (input_tensor, target_tensor)

        input_tensor = H(t)   -> shape (2, 18, 8, 2)
        target_tensor = H(t+1)-> shape (2, 18, 8, 2)
        """
        u, t = self.indices[idx]
        h_t = self.data[u, :, :, :, t, :]   
        h_next = self.data[u, :, :, :, t+1, :] 
        h_t = h_t.permute(0, 3, 1, 2)  # => (2, 2, 18, 8)
        h_t = h_t.reshape(4, 18, 8)   # => (4, 18, 8)

        h_next = h_next.permute(0, 3, 1, 2)  # => (2, 2, 18, 8)
        h_next = h_next.reshape(4, 18, 8)   # => (4, 18, 8)

        return h_t, h_next