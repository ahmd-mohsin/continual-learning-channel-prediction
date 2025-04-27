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

from model import *
from dataloader import get_all_datasets
from utils import compute_device, evaluate_model
from nmse import evaluate_nmse_vs_snr
from loss import NMSELoss
# Set device and hyperparameters

device = compute_device()
snr_list = [0, 5, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
batch_size = 2046

# Load datasets
print("Loading datasets...")
data_dir = "../dataset/outputs/"
train_S1, test_S1, train_S2, test_S2, train_S3, test_S3, \
    train_loader_S1, test_loader_S1, train_loader_S2, test_loader_S2, \
    train_loader_S3, test_loader_S3 = get_all_datasets(data_dir, batch_size=batch_size, dataset_id="all")
print("Loaded datasets successfully.")


# ---------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------
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
        criterion = NMSELoss(reduction='mean')  # use MSE as proxy loss
        total_samples = len(data_loader.dataset) if sample_size is None else sample_size
        count = 0
        for X_batch, Y_batch in tqdm(data_loader, desc="Computing Fisher information"):
            # Limit the number of samples if sample_size is specified
            if sample_size is not None and count >= sample_size:
                break
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            model.zero_grad()
            # Forward pass
            pred = model(X_batch)
            # Compute loss (using ground truth, i.e., empirical Fisher)
            loss = criterion(pred, Y_batch)
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
    model = GRUModel(input_dim=1, hidden_dim=32, output_dim=1, n_layers=3, H=16, W=18).to(device)
elif args.model_type == 'LSTM':
    model = LSTMModel(input_dim=1, hidden_dim=32, output_dim=1, n_layers=3, H=16, W=18).to(device)
else:
    model = TransformerModel(dim_val=128, n_heads=4, n_encoder_layers=1,
                             n_decoder_layers=1, out_channels=2, H=16, W=18).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = NMSELoss()
num_epochs = 30
lambda_reg = 0.4

# Initialize SI if strategy is ewc_si
si_helper = None
if args.strategy == 'ewc_si':
    si_helper = SI(model)

# Task 1: S1
print("=== Task 1: S1 ===")
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    total_batches = len(train_loader_S1)

    loop = tqdm(
        enumerate(train_loader_S1, 1),
        total=total_batches,
        desc=f"S1 Epoch {epoch}/{num_epochs}"
    )

    for batch_idx, (X_batch, Y_batch) in loop:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        optimizer.zero_grad()

        pred = model(X_batch)
        loss = criterion(pred, Y_batch)
        loss.backward()

        # SI accumulation before the optimizer step
        if si_helper:
            si_helper.accumulate(model, optimizer.param_groups[0]['lr'])

        optimizer.step()

        # update running loss and tqdm postfix
        running_loss += loss.item()
        loop.set_postfix(batch_loss=f"{loss.item()}")

    # epoch stats
    avg_loss = running_loss / total_batches
    print(f"Epoch {epoch} S1 — Total Loss: {running_loss} | Avg Loss: {avg_loss}")

# Setup continual helper
ewc_helper = None
if args.strategy == 'ewc':
    ewc_helper = EWC(model, train_loader_S1, device)
elif args.strategy == 'ewc_si':
    si_helper.update_omega(model)

# Task 2: S2
print("=== Task 2: S2 ===")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    total_batches = len(train_loader_S2)

    loop = tqdm(
        enumerate(train_loader_S2, 1),
        total=total_batches,
        desc=f"S2 Epoch {epoch}/{num_epochs}"
    )

    for batch_idx, (X_batch, Y_batch) in loop:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        optimizer.zero_grad()

        pred = model(X_batch)
        base_loss = criterion(pred, Y_batch)

        if args.strategy == 'ewc':
            penalty = ewc_helper.penalty(model)
        else:
            penalty = si_helper.penalty(model)

        loss = base_loss + lambda_reg * penalty
        loss.backward()

        # Accumulate SI information if using SI
        if args.strategy != 'ewc' and si_helper:
            si_helper.accumulate(model, optimizer.param_groups[0]['lr'])

        optimizer.step()

        # update running loss & tqdm postfix
        running_loss += loss.item()
        loop.set_postfix(batch_loss=f"{loss.item()}")

    avg_loss = running_loss / total_batches
    print(f"Epoch {epoch} S2 — Total Loss: {running_loss} | Avg Loss: {avg_loss}")

# Update after Task 2
ewc_helper2 = None
if args.strategy == 'ewc':
    ewc_helper2 = EWC(model, train_loader_S2, device)
elif args.strategy == 'ewc_si':
    si_helper.update_omega(model)

# Task 3: S3
print("=== Task 3: S3 ===")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    total_batches = len(train_loader_S3)

    loop = tqdm(
        enumerate(train_loader_S3, 1),
        total=total_batches,
        desc=f"S3 Epoch {epoch}/{num_epochs}"
    )

    for batch_idx, (X_batch, Y_batch) in loop:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        optimizer.zero_grad()

        pred = model(X_batch)
        base_loss = criterion(pred, Y_batch)

        if args.strategy == 'ewc':
            penalty = ewc_helper.penalty(model) + ewc_helper2.penalty(model)
        else:
            penalty = si_helper.penalty(model)

        loss = base_loss + lambda_reg * penalty
        loss.backward()

        # Accumulate SI information if using SI
        if args.strategy != 'ewc' and si_helper:
            si_helper.accumulate(model, optimizer.param_groups[0]['lr'])

        optimizer.step()

        # update running loss & tqdm postfix
        running_loss += loss.item()
        loop.set_postfix(batch_loss=f"{loss.item()}")

    avg_loss = running_loss / total_batches
    print(f"Epoch {epoch} S3 — Total Loss: {running_loss} | Avg Loss: {avg_loss}")

# Evaluation
print("=== Evaluation ===")
test_loss_csv = f"{args.strategy}_{args.model_type}_loss.csv"
results = {
    'S1_Compact': evaluate_model(model, test_loader_S1, device),
    'S2_Dense': evaluate_model(model, test_loader_S2, device),
    'S3_Standard': evaluate_model(model, test_loader_S3, device)
}
with open(test_loss_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Task', 'Loss'])
    for t, l in results.items():
        print(f"{t} Loss: {l}")
        writer.writerow([t, l])
print(f"Saved losses to {test_loss_csv}")

nmse_results_ewc = {}
nmse_results_ewc['S1_Compact'] = evaluate_nmse_vs_snr(model, test_loader_S1, device, snr_list)
nmse_results_ewc['S2_Dense'] = evaluate_nmse_vs_snr(model, test_loader_S2, device, snr_list)
nmse_results_ewc['S3_Standard'] = evaluate_nmse_vs_snr(model, test_loader_S3, device, snr_list)


csv_rows = [['Task', 'SNR', 'NMSE']]
for task, res in nmse_results_ewc.items():
    for snr, nmse in res.items():
        print(f"Task {task} | SNR {snr:2d} → NMSE {nmse}")
        csv_rows.append([task, snr, f"{nmse}"])

csv_path = f"{args.strategy}_{args.model_type}_nmse_results.csv"
with open(csv_path, 'w', newline='') as f:
    csv.writer(f).writerows(csv_rows)

print(f"\nResults saved to {csv_path}")
