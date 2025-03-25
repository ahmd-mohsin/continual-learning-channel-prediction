import torch
from torch.utils.data import Dataset
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
                channel_group = f["channel_matrix"]
                real = np.array(channel_group["real"])
                imag = np.array(channel_group["imag"])
                self.data = real + 1j * imag
        else:
            raise ValueError("Unsupported file format. Please use npy or mat.")
    
        self.overlapping_index = 16

        # Find the max values for the entire dataset
        self.max_value = np.max(np.abs(self.data))  # The absolute max value across both real and imaginary parts

    def __len__(self):
        return self.num_users * (self.time_length - (self.overlapping_index + 1))

    def __getitem__(self, idx):
        sample_idx = idx // (self.time_length - (self.overlapping_index + 1))
        time_idx = idx % (self.time_length - (self.overlapping_index + 1))

        real_input = self.data.real[sample_idx, :, :, :, time_idx:time_idx+self.overlapping_index]
        imag_input = self.data.imag[sample_idx, :, :, :, time_idx:time_idx+self.overlapping_index]
        
        real_output = self.data.real[sample_idx, :, :, :, time_idx+self.overlapping_index]
        imag_output = self.data.imag[sample_idx, :, :, :, time_idx+self.overlapping_index]

        # Normalize the input and output values using the max values
        # real_input = real_input 
        # imag_input = imag_input 
        # real_output = real_output 
        # imag_output = imag_output 
        # -----------------------------
        real_input = real_input / self.max_value
        imag_input = imag_input / self.max_value
        real_output = real_output / self.max_value
        imag_output = imag_output / self.max_value

        real_input = torch.tensor(real_input, dtype=torch.float32, device=self.device)
        imag_input = torch.tensor(imag_input, dtype=torch.float32, device=self.device)
        real_output = torch.tensor(real_output, dtype=torch.float32, device=self.device)
        imag_output = torch.tensor(imag_output, dtype=torch.float32, device=self.device)

        input_data = torch.cat([real_input, imag_input], dim=0)
        output_data = torch.cat([real_output, imag_output], dim=0)

        return input_data, output_data

    def get_max_values(self):
        return {
            'overall_max': self.max_value,
        }