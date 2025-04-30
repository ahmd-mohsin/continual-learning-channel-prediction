# utils.py  – mask-aware utilities
# --------------------------------
import torch
from tqdm import tqdm
from loss import masked_nmse          # <- function from loss.py
import torch.nn.functional as F


# ------------------------------------------------------------------
# device picker (unchanged)
# ------------------------------------------------------------------
def compute_device():
    if torch.cuda.is_available():
        print("Using CUDA for computation")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using MPS for computation")
        return torch.device("mps")
    print("Using CPU for computation")
    return torch.device("cpu")


# ------------------------------------------------------------------
# evaluation on a single dataloader
# ------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """
    Returns average *masked* NMSE over the dataloader.
    Assumes each sample is
        X : (B, 2, R, S, T, L)
        Y : (B, 2, R, S, T)   where [0]=magnitude, [1]=mask
    """
    model.eval()
    total_loss, n_batches = 0.0, 0

    for X, Y in tqdm(dataloader, desc="Evaluating"):
        X, Y = X.to(device), Y.to(device)
        mag_t, mask_t = Y[:, 0], Y[:, 1]          # targets

        mag_p, _ = model(X)
        total_loss += masked_nmse(mag_p, mag_t, mask_t).item()
        n_batches += 1

    avg = total_loss / max(n_batches, 1)
    print(f"Evaluation completed – masked NMSE: {avg:.4e}")
    return avg


# ------------------------------------------------------------------
# evaluate NMSE vs. SNR curve (mask-aware)
# ------------------------------------------------------------------
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

    return results
