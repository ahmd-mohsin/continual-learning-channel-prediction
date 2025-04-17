import os
import copy
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset, TensorDataset

# Import provided modules (assumed to be in the same repository)
from sequence_loader import ChannelSequenceDataset
from model import LSTMModel
from nmse import evaluate_nmse_vs_snr  # NMSE vs SNR evaluation function

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
        s1_path = os.path.join(data_dir, "umi_compact_conf_2tx_2rx.")
        train_S1, test_S1, train_loader_S1, test_loader_S1 = load_dataset(s1_path)
        print(f"Loaded S1: {len(train_S1)} train samples, {len(test_S1)} test samples")

    if dataset_id == 2 or dataset_id == 'all':
        s2_path = os.path.join(data_dir, "umi_dense_conf_8tx_2rx.")
        train_S2, test_S2, train_loader_S2, test_loader_S2 = load_dataset(s2_path)
        print(f"Loaded S2: {len(train_S2)} train samples, {len(test_S2)} test samples")

    if dataset_id == 3 or dataset_id == 'all':
        s3_path = os.path.join(data_dir, "umi_standard_conf_16tx_2rx.")
        train_S3, test_S3, train_loader_S3, test_loader_S3 = load_dataset(s3_path)
        print(f"Loaded S3: {len(train_S3)} train samples, {len(test_S3)} test samples")

    return (
        train_S1, test_S1, train_S2, test_S2, train_S3, test_S3,
        train_loader_S1, test_loader_S1, train_loader_S2, test_loader_S2, train_loader_S3, test_loader_S3
    )