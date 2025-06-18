import argparse
import random
import os
import sys
import csv
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import *
from dataloader import get_all_datasets
from utils import compute_device, evaluate_nmse_vs_snr_masked
from torch.nn.utils import clip_grad_norm_
from loss import *
# Set device and hyperparameters
import math
device = compute_device()
BATCH_SIZE = 2048
SEQ_LEN    = 32
NUM_EPOCHS = 100
ALPHA      = 0.2         # weight for BCE mask loss
LR         = 1e-4
SNR_LIST   = [0,5,10,12,14,16,18,20,22,24,26,28,30]

# Load datasets
print("Loading datasets...")
train_S1, test_S1, train_loader_S1, test_loader_S1, \
train_S2, test_S2, train_loader_S2, test_loader_S2, \
train_S3, test_S3, train_loader_S3, test_loader_S3 = \
    get_all_datasets(
        data_dir      = "../dataset/outputs/",
        batch_size    = BATCH_SIZE,
        dataset_id    = "all",
        normalization = "log_min_max",
        per_user      = True,
        seq_len       = SEQ_LEN
    )
print("Loaded datasets successfully.")



parser = argparse.ArgumentParser()
# CLI arguments\parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='LSTM',
                    choices=['LSTM', 'GRU', 'TRANS'],
                    help='Model type for continual learning')
parser.add_argument('--strategy', type=str, default='ewc_si',
                    choices=['ewc', 'ewc_si'],
                    help='Continual learning strategy: ewc or ewc_si')
args = parser.parse_args()

# Elastic Weight Consolidation (EWC) helper
class EWC:
    """Elastic Weight Consolidation helper to store Fisher information and original parameters."""
    def __init__(self, model: nn.Module, data_loader: DataLoader, device: torch.device, sample_size=None):
        """
        Compute the Fisher information diagonal and parameter snapshot on the given data (previous task).
        Args:
            model: Trained model on the previous task (whose parameters we want to remember).
            data_loader: DataLoader for the previous task's training data.
            device: Device on which to perform computations.
            sample_size: If set, limit the number of samples used to compute Fisher (for efficiency).
        """
        self.device = device
        # Store the reference parameters (theta^* from the old task)
        self.params_snapshot = {name: p.clone().detach() for name, p in model.named_parameters()}
        # Initialize Fisher information for each parameter to zero
        self.fisher_diag = {name: torch.zeros_like(p, device=device) for name, p in model.named_parameters()}
        
        # Set model to evaluation mode and compute Fisher information
        # model.eval()
        total_samples = len(data_loader.dataset) if sample_size is None else sample_size
        count = 0
        for X_batch, Y_batch in tqdm(data_loader, desc="Computing Fisher information"):
            # Limit the number of samples if sample_size is specified
            if sample_size is not None and count >= sample_size:
                break
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            
            mag_t, mask_t = Y_batch[:,0], Y_batch[:,1]
            model.zero_grad()
            # Forward pass
            mag_p, mask_logits = model(X_batch)
            # original masked NMSE
            loss_mag  = masked_nmse(mag_p, mag_t, mask_t)
            loss_mask = bce_loss(mask_logits, mask_t)
            # composite loss
            loss = (
                loss_mag
                + ALPHA * loss_mask
            )
            # Backward pass to compute gradients
            loss.backward()
            # Accumulate squared gradients
            for name, p in model.named_parameters():
                if p.grad is not None:
                    # sum of squared grad
                    self.fisher_diag[name] += (p.grad.detach() ** 2)
            count += X_batch.size(0)
        # Average the Fisher information
        for name in self.fisher_diag:
            self.fisher_diag[name] /= float(min(total_samples, count))
        model.train()
    
    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute the EWC penalty term for the given model's current parameters.
        This will be added to the loss to penalize deviation from stored parameters.
        """
        penalty = 0.0
        for name, p in model.named_parameters():
            # Fisher * (theta - theta_old)^2
            if name in self.fisher_diag:
                diff = p - self.params_snapshot[name]
                penalty += torch.sum(self.fisher_diag[name] * (diff ** 2))
        return penalty



# Synaptic Intelligence (SI) helper
class SI:
    def __init__(self, model: nn.Module, xi: float = 0.1):
        self.xi = xi
        self.prev_params = {name: p.clone().detach() for name, p in model.named_parameters()}
        self.omega = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
        self._small_omega = {name: torch.zeros_like(p) for name, p in model.named_parameters()}

    def accumulate(self, model: nn.Module, lr: float):
        for name, p in model.named_parameters():
            if p.grad is not None:
                self._small_omega[name] += (p.grad.detach() ** 2) * lr

    def update_omega(self, model: nn.Module):
        for name, p in model.named_parameters():
            delta = p.detach() - self.prev_params[name]
            denom = delta**2 + self.xi
            self.omega[name] += self._small_omega[name] / denom
            self._small_omega[name].zero_()
            self.prev_params[name] = p.clone().detach()

    def penalty(self, model: nn.Module) -> torch.Tensor:
        loss = 0.0
        for name, p in model.named_parameters():
            loss += (self.omega[name] * (p - self.prev_params[name])**2).sum()
        return loss

# Model instantiation
if args.model_type == 'GRU':
    model = GRUModel(input_dim=1, hidden_dim=32, output_dim=1, n_layers=3, H=16, W=9).to(device)
elif args.model_type == 'LSTM':
    model = LSTMChannelPredictor().to(device)
else:
    model = TransformerModel(dim_val=128, n_heads=4, n_encoder_layers=1,
                             n_decoder_layers=1, out_channels=2, H=16, W=9).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
lambda_reg = 0.4
bce_loss  = torch.nn.BCEWithLogitsLoss()


# sched     = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

# Initialize SI if strategy is ewc_si
si_helper = None
if args.strategy == 'ewc_si':
    si_helper = SI(model)

# Task 1: S1
print("=== Task 1: S1 ===")
sched = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    total_batches = len(train_loader_S1)

    loop = tqdm(
        enumerate(train_loader_S1, 1),
        total=total_batches,
        desc=f"S1 Epoch {epoch}/{NUM_EPOCHS}"
    )

    for batch_idx, (X_batch, Y_batch) in loop:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        mag_t, mask_t = Y_batch[:,0], Y_batch[:,1]

        optimizer.zero_grad()
        mag_p, mask_logits = model(X_batch)
        # original masked NMSE
        loss_mag  = masked_nmse(mag_p, mag_t, mask_t)
        loss_mask = bce_loss(mask_logits, mask_t)
        # composite loss
        base_loss = (
            loss_mag
            + ALPHA * loss_mask
        )
        # SI accumulation before the optimizer step
        if si_helper:
            si_helper.accumulate(model, optimizer.param_groups[0]['lr'])
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # update running loss and tqdm postfix
        running_loss += base_loss.detach().item()
        loop.set_postfix(
            nmse = loss_mag.item(),
            bce  = loss_mask.item(),
        )
        # loop.set_postfix(batch_loss=f"{base_loss.item()}")
    # epoch stats
    avg_loss = running_loss / total_batches
    sched.step(avg_loss)

    print(f"Epoch {epoch} S1 — Total Loss: {running_loss} | Avg Loss: {avg_loss}")

# Setup continual helper
ewc_helper = None
if args.strategy == 'ewc':
    ewc_helper = EWC(model, train_loader_S1, device)
elif args.strategy == 'ewc_si':
    si_helper.update_omega(model)

# Task 2: S2
print("=== Task 2: S2 ===")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    total_batches = len(train_loader_S2)

    loop = tqdm(
        enumerate(train_loader_S2, 1),
        total=total_batches,
        desc=f"S2 Epoch {epoch}/{NUM_EPOCHS}"
    )

    for batch_idx, (X_batch, Y_batch) in loop:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        mag_t, mask_t = Y_batch[:,0], Y_batch[:,1]

        optimizer.zero_grad()
        mag_p, mask_logits = model(X_batch)
        # original masked NMSE
        loss_mag  = masked_nmse(mag_p, mag_t, mask_t)
        loss_mask = bce_loss(mask_logits, mask_t)
        # composite loss
        base_loss = (
            loss_mag
            + ALPHA * loss_mask
        )

        if args.strategy == 'ewc':
            penalty = ewc_helper.penalty(model)
        else:
            penalty = si_helper.penalty(model)

        loss = base_loss + lambda_reg * penalty
        loss.backward()

        # Accumulate SI information if using SI
        if args.strategy != 'ewc' and si_helper:
            si_helper.accumulate(model, optimizer.param_groups[0]['lr'])
        
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # update running loss & tqdm postfix
        running_loss += loss.item()
        loop.set_postfix(
            nmse = loss_mag.item(),
            bce  = loss_mask.item(),
        )
    avg_loss = running_loss / total_batches
    sched.step(avg_loss)
    print(f"Epoch {epoch} S2 — Total Loss: {running_loss} | Avg Loss: {avg_loss}")

# Update after Task 2
ewc_helper2 = None
if args.strategy == 'ewc':
    ewc_helper2 = EWC(model, train_loader_S2, device)
elif args.strategy == 'ewc_si':
    si_helper.update_omega(model)

# Task 3: S3
print("=== Task 3: S3 ===")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    total_batches = len(train_loader_S3)

    loop = tqdm(
        enumerate(train_loader_S3, 1),
        total=total_batches,
        desc=f"S3 Epoch {epoch}/{NUM_EPOCHS}"
    )

    for batch_idx, (X_batch, Y_batch) in loop:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        mag_t, mask_t = Y_batch[:,0], Y_batch[:,1]

        optimizer.zero_grad()
        mag_p, mask_logits = model(X_batch)
        # original masked NMSE
        loss_mag  = masked_nmse(mag_p, mag_t, mask_t)
        loss_mask = bce_loss(mask_logits, mask_t)
        # composite loss
        base_loss = (
            loss_mag
            + ALPHA * loss_mask
        )
        if args.strategy == 'ewc':
            penalty = ewc_helper.penalty(model) + ewc_helper2.penalty(model)
        else:
            penalty = si_helper.penalty(model)

        loss = base_loss + lambda_reg * penalty
        loss.backward()

        # Accumulate SI information if using SI
        if args.strategy != 'ewc' and si_helper:
            si_helper.accumulate(model, optimizer.param_groups[0]['lr'])
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # update running loss & tqdm postfix
        running_loss += loss.item()
        loop.set_postfix(
            nmse = loss_mag.item(),
            bce  = loss_mask.item(),
        )

    avg_loss = running_loss / total_batches
    sched.step(avg_loss)
    print(f"Epoch {epoch} S3 — Total Loss: {running_loss} | Avg Loss: {avg_loss}")


print("\n=== NMSE Evaluation ===")
nmse_results = {
    'S1_Compact': evaluate_nmse_vs_snr_masked(model, test_loader_S1, device, SNR_LIST),
    'S2_Dense': evaluate_nmse_vs_snr_masked(model, test_loader_S2, device, SNR_LIST),
    'S3_Standard': evaluate_nmse_vs_snr_masked(model, test_loader_S3, device, SNR_LIST),
}

csv_rows = [['Task', 'SNR', 'NMSE Masked', 'NMSE (dB)']]
for task, res in nmse_results.items():
    for snr, nmse in res.items():
        nmse_db = -10 * math.log10(nmse + 1e-12)  # add tiny eps in case nmse==0
        print(f"Task {task} | SNR {snr:2d} → NMSE {nmse} | {nmse_db} dB")
        csv_rows.append([task, snr, f"{nmse}", f"{nmse_db}"])

csv_path = f"{args.strategy}_{args.model_type}_nmse_results.csv"
with open(csv_path, 'w', newline='') as f:
    csv.writer(f).writerows(csv_rows)

print(f"\nResults saved to {csv_path}")
