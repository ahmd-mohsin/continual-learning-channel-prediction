import torch
from torch.utils.data import Dataset, DataLoader

class ChannelDataset(Dataset):
    """
    A dataset that provides pairs (H(t), H(t+1)) for each user and time t.
    channel_matrix shape is assumed to be (num_users, rx_ant, nRB, tx_ant, time).
    """
    def __init__(self, channel_matrix):
        super().__init__()
        
        # channel_matrix: shape (256, 2, 18, 8, 3000), complex128
        # We'll separate real & imag. 
        # For convenience, store them in a single float tensor of shape (256, 2, 18, 8, 3000, 2).
        
        real_data = channel_matrix.real  # shape (256, 2, 18, 8, 3000)
        imag_data = channel_matrix.imag  # shape (256, 2, 18, 8, 3000)
        
        # Stack real and imaginary along a new last dimension = 2
        # shape => (256, 2, 18, 8, 3000, 2)
        self.data = torch.from_numpy(
            np.stack([real_data, imag_data], axis=-1)
        ).float()
        
        self.num_users = self.data.shape[0]  # 256
        self.num_times = self.data.shape[4]  # 3000
        
        # We'll create an index list of all valid (user, time) pairs
        # For t in [0..(3000-2)] if you want (t -> t+1).
        # Or [0..(3000-2)] if you prefer not to exceed bounds. 
        # We'll do [0..(3000-2)] so that H(t+1) is valid for up to t=2998.
        
        self.indices = []
        for u in range(self.num_users):             # 0..255
            for t in range(self.num_times - 1):     # 0..2998
                self.indices.append((u, t))
                
        # If you want to shuffle differently, you could do that in the DataLoader's sampler.

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
        
        # Input is channel at time t
        # shape: (rx_ant=2, nRB=18, tx_ant=8, 2 for real/imag)
        h_t = self.data[u, :, :, :, t, :]   # => (2, 18, 8, 2)
        
        # Target is channel at time t+1
        h_next = self.data[u, :, :, :, t+1, :]  # => (2, 18, 8, 2)

        # Option 1: Return them "as is." The shape is (2, 18, 8, 2).
        # Option 2: If you're using a 2D CNN (e.g., MobileNet), you might reorder to (C,H,W).
        # Example: combine [rx_ant=2, real/imag=2] into a single channel dimension => 4 channels.
        # Then H=18, W=8. 
        # Let's do that reordering for a typical CNN input: (channels=4, height=18, width=8).
        
        # (2, 18, 8, 2) -> (2, 2, 18, 8) if we swap last two dims => (rx_ant, real/imag, nRB, tx_ant)
        # Then reshape => (rx_ant*realimag, nRB, tx_ant) => (4, 18, 8).
        h_t = h_t.permute(0, 3, 1, 2)  # => (2, 2, 18, 8)
        h_t = h_t.reshape(4, 18, 8)   # => (4, 18, 8)

        h_next = h_next.permute(0, 3, 1, 2)  # => (2, 2, 18, 8)
        h_next = h_next.reshape(4, 18, 8)   # => (4, 18, 8)

        return h_t, h_next