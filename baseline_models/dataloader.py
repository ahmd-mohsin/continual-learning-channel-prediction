import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import torch.nn.functional as F



import os
import copy
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset, TensorDataset


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

        # to torch tensors, add channel dim =1
        # inp = torch.tensor(inp, dtype=torch.float32, device=self.device).unsqueeze(0)
        # out = torch.tensor(out, dtype=torch.float32, device=self.device).unsqueeze(0)
        inp = torch.tensor(inp, dtype=torch.float32, device=self.device)
        out = torch.tensor(out, dtype=torch.float32, device=self.device)

        return inp, out

    def get_max_values(self):
        return {'overall_max': self.max_value}





# Set device for training (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to datasets (without extension)

# Split each dataset into training and testing sets (80/20 split)
def split_dataset(dataset, split_ratio=0.8):
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])


def get_all_datasets(data_dir, batch_size = 16, dataset_id = 1):
    
    def load_dataset(path_prefix):
        dataset = ChannelSequenceDataset(path_prefix, "mat", device=device)
        train, test = split_dataset(dataset, split_ratio=0.8)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=False)
        return train, test, train_loader, test_loader

    # Initialize everything as None
    train_S1 = test_S1 = train_loader_S1 = test_loader_S1 = None
    train_S2 = test_S2 = train_loader_S2 = test_loader_S2 = None
    train_S3 = test_S3 = train_loader_S3 = test_loader_S3 = None

    if dataset_id == 1 or dataset_id == 'all':
        s1_path = os.path.join(data_dir, "umi_compact_8Tx_2Rx.")
        train_S1, test_S1, train_loader_S1, test_loader_S1 = load_dataset(s1_path)
        print(f"Loaded S1: {len(train_S1)} train samples, {len(test_S1)} test samples")

    if dataset_id == 2 or dataset_id == 'all':
        s2_path = os.path.join(data_dir, "umi_dense_8Tx_2Rx.")
        train_S2, test_S2, train_loader_S2, test_loader_S2 = load_dataset(s2_path)
        print(f"Loaded S2: {len(train_S2)} train samples, {len(test_S2)} test samples")

    if dataset_id == 3 or dataset_id == 'all':
        s3_path = os.path.join(data_dir, "umi_standard_8Tx_2Rx.")
        train_S3, test_S3, train_loader_S3, test_loader_S3 = load_dataset(s3_path)
        print(f"Loaded S3: {len(train_S3)} train samples, {len(test_S3)} test samples")

    return (
        train_S1, test_S1, train_S2, test_S2, train_S3, test_S3,
        train_loader_S1, test_loader_S1, train_loader_S2, test_loader_S2, train_loader_S3, test_loader_S3
    )