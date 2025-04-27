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
import copy
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
parser.add_argument('--strategy', type=str, default='lwf',
                    choices=['lwf'],
                    help='Continual learning strategy: lwf')
args = parser.parse_args()


# Training the model sequentially on S1 -> S2 -> S3 using LwF (Knowledge Distillation)
# Model instantiation
if args.model_type == 'GRU':
    model_lwf = GRUModel(input_dim=1, hidden_dim=32, output_dim=1, n_layers=3, H=16, W=9).to(device)
elif args.model_type == 'LSTM':
    model_lwf = LSTMModel(input_dim=1, hidden_dim=32, output_dim=1, n_layers=3, H=16, W=9).to(device)
else:
    model_lwf = TransformerModel(dim_val=128, n_heads=4, n_encoder_layers=1,
                             n_decoder_layers=1, out_channels=2, H=16, W=9).to(device)

optimizer = torch.optim.Adam(model_lwf.parameters(), lr=1e-3)
criterion = NMSELoss()

num_epochs = 30
distill_lambda = 0.5  # weight for distillation loss

# Train on Task 1 (S1) normally (no old model yet)
for epoch in range(num_epochs):
    en_idx = 0
    for X_batch, Y_batch in tqdm(train_loader_S1, desc="Training Task 1"):
        # en_idx += 1
        # if en_idx > 10:
        #     break
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        optimizer.zero_grad()
        pred = model_lwf(X_batch)
        loss = criterion(pred, Y_batch)
        loss.backward()
        optimizer.step()

# Save the Task 1 model as the old model for LwF
old_model = copy.deepcopy(model_lwf).to(device)
old_model.eval()
for param in old_model.parameters():
    param.requires_grad = False  # freeze old model

# Train on Task 2 (S2) with LwF
optimizer = torch.optim.Adam(model_lwf.parameters(), lr=1e-3)
for epoch in range(num_epochs):
    en_idx = 0
    for X_batch, Y_batch in tqdm(train_loader_S2, desc="Training Task 2"):
        # en_idx += 1
        # if en_idx > 10:
        #     break
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        optimizer.zero_grad()
        # Forward pass new model
        pred = model_lwf(X_batch)
        # Forward pass old model (with no grad)
        with torch.no_grad():
            old_pred = old_model(X_batch)
        # Compute losses
        task_loss = criterion(pred, Y_batch)  # new task supervision loss
        distill_loss = criterion(pred, old_pred)  # L2 loss between new and old outputs
        loss = task_loss + distill_lambda * distill_loss
        loss.backward()
        optimizer.step()

# Update old_model to the model after Task 2
old_model = copy.deepcopy(model_lwf).to(device)
old_model.eval()
for param in old_model.parameters():
    param.requires_grad = False

# Train on Task 3 (S3) with LwF
optimizer = torch.optim.Adam(model_lwf.parameters(), lr=1e-3)
for epoch in range(num_epochs):
    en_idx = 0
    for X_batch, Y_batch in tqdm(train_loader_S3, desc="Training Task 3"):
        # en_idx += 1
        # if en_idx > 10:
        #     break
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        optimizer.zero_grad()
        pred = model_lwf(X_batch)
        with torch.no_grad():
            old_pred = old_model(X_batch)
        task_loss = criterion(pred, Y_batch)
        distill_loss = criterion(pred, old_pred)
        loss = task_loss + distill_lambda * distill_loss
        loss.backward()
        optimizer.step()

# Evaluate final model on all tasks (NMSE vs SNR)
# print("LwF Method - NMSE on each task across SNRs:")
# nmse_results_lwf = {}
# nmse_results_lwf['S1'] = evaluate_nmse_vs_snr(model_lwf, test_loader_S1, device, snr_list)
# nmse_results_lwf['S2'] = evaluate_nmse_vs_snr(model_lwf, test_loader_S2, device, snr_list)
# nmse_results_lwf['S3'] = evaluate_nmse_vs_snr(model_lwf, test_loader_S3, device, snr_list)
# for task, nmse_vs_snr in nmse_results_lwf.items():
#     print(f"Task {task}: " + ", ".join([f"SNR {snr}: NMSE {nmse:.4f}" 
#                                         for snr, nmse in nmse_vs_snr.items()]))





# Evaluation
print("=== Evaluation ===")
test_loss_csv = f"{args.strategy}_{args.model_type}_loss.csv"
results = {
    'S1_Compact': evaluate_model(model_lwf, test_loader_S1, device),
    'S2_Dense': evaluate_model(model_lwf, test_loader_S2, device),
    'S3_Standard': evaluate_model(model_lwf, test_loader_S3, device)
}
with open(test_loss_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Task', 'Loss'])
    for t, l in results.items():
        print(f"{t} Loss: {l}")
        writer.writerow([t, l])
print(f"Saved losses to {test_loss_csv}")

nmse_results_ewc = {}
nmse_results_ewc['S1_Compact'] = evaluate_nmse_vs_snr(model_lwf, test_loader_S1, device, snr_list)
nmse_results_ewc['S2_Dense'] = evaluate_nmse_vs_snr(model_lwf, test_loader_S2, device, snr_list)
nmse_results_ewc['S3_Standard'] = evaluate_nmse_vs_snr(model_lwf, test_loader_S3, device, snr_list)


csv_rows = [['Task', 'SNR', 'NMSE']]
for task, res in nmse_results_ewc.items():
    for snr, nmse in res.items():
        print(f"Task {task} | SNR {snr:2d} → NMSE {nmse}")
        csv_rows.append([task, snr, f"{nmse}"])

csv_path = f"{args.strategy}_{args.model_type}_nmse_results.csv"
with open(csv_path, 'w', newline='') as f:
    csv.writer(f).writerows(csv_rows)

print(f"\nResults saved to {csv_path}")
