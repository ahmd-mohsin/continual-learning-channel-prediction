import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

class ChannelSequenceDataset(Dataset):
    def __init__(self, file_path, file_extension, device):
        self.file_path = file_path + file_extension
        self.file_extension = file_extension
        self.device = device
        
        if self.file_extension == "npy":
            self.data = np.load(self.file_path, mmap_mode='r')
            self.num_users = self.data.shape[0]
            self.time_length = self.data.shape[-1]
        elif self.file_extension == "mat":
            with h5py.File(self.file_path, "r") as f:
                self.num_users = f["channel_matrix"].shape[0]
                self.time_length = f["channel_matrix"].shape[-1]
        else:
            raise ValueError("Unsupported file format. Please use npy or mat.")
    
        self.overlapping_index = 16
    
    def __len__(self):
        return self.num_users * (self.time_length - (self.overlapping_index + 1))

    def __getitem__(self, idx):
        """
        Returns:
            input  -> (4, 18, 8, self.overlapping_index)  (self.overlapping_index time steps with real and imaginary parts concatenated)
            output -> (4, 18, 8)(next time step with real and imaginary parts concatenated)
        """
        sample_idx = idx // (self.time_length - (self.overlapping_index + 1))  
        time_idx = idx % (self.time_length - (self.overlapping_index + 1))  

        real_input = self.data.real[sample_idx, :, :, :, time_idx:time_idx+self.overlapping_index]
        imag_input = self.data.imag[sample_idx, :, :, :, time_idx:time_idx+self.overlapping_index]
        
        real_output = self.data.real[sample_idx, :, :, :, time_idx+self.overlapping_index]
        imag_output = self.data.imag[sample_idx, :, :, :, time_idx+self.overlapping_index]

        real_input = torch.tensor(real_input, dtype=torch.float32, device=self.device)
        imag_input = torch.tensor(imag_input, dtype=torch.float32, device=self.device)
        real_output = torch.tensor(real_output, dtype=torch.float32, device=self.device)
        imag_output = torch.tensor(imag_output, dtype=torch.float32, device=self.device)

       
        input_data = torch.cat([real_input, imag_input], dim=0)  
        output_data = torch.cat([real_output, imag_output], dim=0)  

        return input_data, output_data