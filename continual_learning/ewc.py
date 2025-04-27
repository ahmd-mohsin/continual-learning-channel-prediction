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


device = compute_device()
snr_list = [0, 5, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
batch_size = 2046

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
parser.add_argument('--model_type', type=str, default='LSTM',
                    choices=['LSTM', 'GRU', 'TRANS'],
                    help='MODEL_TYPE: LSTM')


parser.add_argument('--sampling', type=str, default='ewc',
                    choices=['ewc'],
                    help='Strategy')
args = parser.parse_args()


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


if args.model_type == "GRU":
    model_ewc = GRUModel(
        input_dim=1,      # not strictly used—since we flatten to 2*H*W
        hidden_dim=32,
        output_dim=1,
        n_layers=3,
        H=16,
        W=18
    ).to(device)

elif args.model_type == "LSTM":
    model_ewc = LSTMModel(
        input_dim=1,
        hidden_dim=32,
        output_dim=1,
        n_layers=3,
        H=16,
        W=18
    ).to(device)

elif args.model_type == "TRANS":
    model_ewc = TransformerModel(
            dim_val=128,
            n_heads=4,
            n_encoder_layers=1,
            n_decoder_layers=1,
            out_channels=2,  # Because dataloader outputs (4,18,2)
            H=16,
            W=18,
        ).to(device)


# Training the model sequentially on S1 -> S2 -> S3 using EWC
# model_ewc = LSTMModel().to(device)
optimizer = torch.optim.Adam(model_ewc.parameters(), lr=1e-5)
criterion = NMSELoss()

num_epochs = 30  # epochs per task (adjust as needed)
ewc_lambda = 0.4  # regularization strength for EWC (tunable hyperparameter)

# Train on Task 1 (S1) normally
print("Train on Task 1 (S1) normally")
for epoch in range(1, num_epochs + 1):
    model_ewc.train()
    running_loss = 0.0
    total_batches = len(train_loader_S1)

    loop = tqdm(enumerate(train_loader_S1, 1),
                total=total_batches,
                desc=f"Epoch {epoch}/{num_epochs} - Training S1")

    for batch_idx, (X_batch, Y_batch) in loop:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        optimizer.zero_grad()
        pred = model_ewc(X_batch)
        loss = criterion(pred, Y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(batch_loss=f"{loss.item()}")

    avg_loss = running_loss / total_batches
    print(f"Epoch {epoch} S1 completed — Total Loss: {running_loss} | Avg Loss: {avg_loss}")
    # (Optional) evaluate on S1 validation here

# After Task 1, compute Fisher info on S1 for EWC
ewc_S1 = EWC(model_ewc, train_loader_S1, device=device)

# Train on Task 2 (S2) with EWC regularization
print("Train on Task 2 (S2) with EWC regularization")
# We reinitialize optimizer for a new task to avoid carrying momentum from previous task
optimizer = torch.optim.Adam(model_ewc.parameters(), lr=1e-5)
criterion = NMSELoss()

for epoch in range(1, num_epochs + 1):
    model_ewc.train()
    running_loss = 0.0
    total_batches = len(train_loader_S2)

    # Wrap the loader in tqdm
    loop = tqdm(enumerate(train_loader_S2, 1), 
                total=total_batches,
                desc=f"Epoch {epoch}/{num_epochs}")

    for batch_idx, (X_batch, Y_batch) in loop:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        optimizer.zero_grad()
        pred = model_ewc(X_batch)

        base_loss = criterion(pred, Y_batch)
        penalty   = ewc_S1.penalty(model_ewc)
        loss      = base_loss + ewc_lambda * penalty

        loss.backward()
        optimizer.step()

        # update running loss
        running_loss += loss.item()

        # display current batch loss
        loop.set_postfix(batch_loss=f"{loss.item()}")

    # compute and print epoch total/average loss
    avg_loss = running_loss / total_batches
    print(f"Epoch {epoch} completed — Total Loss: {running_loss} | Avg Loss: {avg_loss}")
    # (Optional) evaluate on S1 and S2 validation here to monitor forgetting

# After Task 2, compute Fisher info on S2 and combine with S1 for EWC
ewc_S2 = EWC(model_ewc, train_loader_S2, device=device)
# We can combine EWC from S1 and S2 by summing their penalties
# (Alternatively, create a single EWC object that stores multiple tasks)

# Train on Task 3 (S3) with EWC regularization (Tasks 1 & 2 penalties)
print("Train on Task 3 (S3) with EWC regularization (Tasks 1 & 2 penalties)")
optimizer = torch.optim.Adam(model_ewc.parameters(), lr=1e-5)
criterion = NMSELoss()

for epoch in range(1, num_epochs + 1):
    model_ewc.train()
    running_loss = 0.0
    total_batches = len(train_loader_S3)

    loop = tqdm(
        enumerate(train_loader_S3, 1),
        total=total_batches,
        desc=f"Epoch {epoch}/{num_epochs} - Training S3"
    )

    for batch_idx, (X_batch, Y_batch) in loop:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        optimizer.zero_grad()
        pred = model_ewc(X_batch)

        base_loss = criterion(pred, Y_batch)
        penalty   = ewc_S1.penalty(model_ewc) + ewc_S2.penalty(model_ewc)
        loss      = base_loss + ewc_lambda * penalty

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(batch_loss=f"{loss.item()}")

    avg_loss = running_loss / total_batches
    print(f"Epoch {epoch} S3 completed — Total Loss: {running_loss} | Avg Loss: {avg_loss}")


# Evaluate final model on all tasks (NMSE vs SNR)
print("EWC Method - NMSE on each task across SNRs:")




test_loss_csv_path = f"{args.sampling}_{args.model_type}_loss_results.csv"

loss_results = {
    'S1_Compact_loss': evaluate_model(model_ewc, test_loader_S1, device),
    'S2_Dense_loss': evaluate_model(model_ewc, test_loader_S2, device),
    'S3_Standard_loss': evaluate_model(model_ewc, test_loader_S3, device),
}

# Prepare the data for CSV
csv_rows = [['Task', 'Loss']]
for task, loss in loss_results.items():
    print(f"Task {task} → Loss {loss}")
    csv_rows.append([task, loss])

# Write to CSV file
with open(test_loss_csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)
print(f"Loss results saved to {test_loss_csv_path}")


nmse_results_ewc = {}
nmse_results_ewc['S1_Compact'] = evaluate_nmse_vs_snr(model_ewc, test_loader_S1, device, snr_list)
nmse_results_ewc['S2_Dense'] = evaluate_nmse_vs_snr(model_ewc, test_loader_S2, device, snr_list)
nmse_results_ewc['S3_Standard'] = evaluate_nmse_vs_snr(model_ewc, test_loader_S3, device, snr_list)


csv_rows = [['Task', 'SNR', 'NMSE']]
for task, res in nmse_results_ewc.items():
    for snr, nmse in res.items():
        print(f"Task {task} | SNR {snr:2d} → NMSE {nmse}")
        csv_rows.append([task, snr, f"{nmse}"])

csv_path = f"{args.sampling}_{args.model_type}_nmse_results.csv"
with open(csv_path, 'w', newline='') as f:
    csv.writer(f).writerows(csv_rows)

print(f"\nResults saved to {csv_path}")
