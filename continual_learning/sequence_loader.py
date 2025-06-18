import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class ChannelSequenceDataset(Dataset):
    def __init__(self, file_path, file_extension, device):
        self.file_path = file_path + file_extension
        self.device = device

        # --- load complex data into a numpy array `raw` ---
        if file_extension == "npy":
            raw = np.load(self.file_path, mmap_mode='r')            # shape: (U, …, T)
        elif file_extension == "mat":
            with h5py.File(self.file_path, "r") as f:
                grp = f["channel_matrix"]
                real = np.array(grp["real"])
                imag = np.array(grp["imag"])
            raw = real + 1j*imag
        else:
            raise ValueError("Unsupported file format. Use .npy or .mat")

        self.num_users = raw.shape[0]
        self.time_length = raw.shape[-1]
        self.overlapping_index = 16

        # --- compute magnitude and normalize globally ---
        mag = np.abs(raw)                                           # now real‐valued
        mag_min, mag_max = mag.min(), mag.max()
        self.data = (mag - mag_min) / (mag_max - mag_min)           # in [0,1]
        # self.data = mag
        self.max_value = mag_max

        print("Loaded data with shape:", self.data.shape)

    def __len__(self):
        return self.num_users * (self.time_length - (self.overlapping_index + 1))

    def __getitem__(self, idx):
        user_idx = idx // (self.time_length - (self.overlapping_index + 1))
        t0       = idx %  (self.time_length - (self.overlapping_index + 1))

        # slice out window of magnitudes
        inp = self.data[user_idx,  :, :, :, t0 : t0 + self.overlapping_index]
        out = self.data[user_idx,  :, :, :, t0 + self.overlapping_index]
        inp = torch.tensor(inp, dtype=torch.float32, device=self.device)
        out = torch.tensor(out, dtype=torch.float32, device=self.device)

        return inp, out

    def get_max_values(self):
        return {'overall_max': self.max_value}
