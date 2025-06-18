import time
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import csv
from loss import *  # If you need CustomLoss from here

def compute_device():
    """
    Determines the best available computing device (CUDA, MPS, or CPU).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA for computation")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) for computation")
    else:
        device = torch.device("cpu")
        print("Using CPU for computation")
    return device



@torch.no_grad()
def evaluate_nmse_vs_snr_masked(model, dataloader, device, snr_list):
    """
    Adds AWGN of given SNR (dB) and computes masked-NMSE for each SNR.
    Returns {snr : nmse}.
    """
    model.eval()
    results = {}

    # Pre-load the whole set into RAM to avoid re-reading per SNR
    full_X, full_Y = [], []
    for X, Y in dataloader:
        full_X.append(X); full_Y.append(Y)
    full_X = torch.cat(full_X).to(device)
    full_Y = torch.cat(full_Y).to(device)

    mag_true  = full_Y[:, 0]
    mask_true = full_Y[:, 1]

    signal_power = (mag_true.pow(2) * mask_true).sum() / mask_true.sum()

    for snr in snr_list:
        snr_lin  = 10 ** (snr / 10)
        noise_var = signal_power / snr_lin

        noise = torch.randn_like(full_X[:, 0]) * noise_var.sqrt()
        noisy_mag = full_X.clone()
        noisy_mag[:, 0] += noise          # ← unsqueeze no longer needed


        mag_pred, _ = model(noisy_mag)
        nmse = masked_nmse(mag_pred, mag_true, mask_true).item()
        results[snr] = nmse
        print(f"NMSE for SNR {snr} dB: {nmse}")

    return results