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
from utils import compute_device, evaluate_nmse_vs_snr_masked
from nmse import evaluate_nmse_vs_snr
from loss import *
from torch.optim.lr_scheduler import CosineAnnealingLR

# Set device and hyperparameters

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
parser.add_argument('--strategy', type=str, default='lwf',
                    choices=['lwf'],
                    help='Continual learning strategy: lwf')
args = parser.parse_args()


# Training the model sequentially on S1 -> S2 -> S3 using LwF (Knowledge Distillation)
# Model instantiation
if args.model_type == 'GRU':
    model_lwf = GRUModel(input_dim=1, hidden_dim=32, output_dim=1, n_layers=3, H=16, W=9).to(device)
elif args.model_type == 'LSTM':
    model_lwf = LSTMChannelPredictor().to(device)
else:
    model_lwf = TransformerModel(dim_val=128, n_heads=4, n_encoder_layers=1,
                             n_decoder_layers=1, out_channels=2, H=16, W=9).to(device)

optimizer = torch.optim.Adam(model_lwf.parameters(), lr=1e-3)
# criterion = NMSELoss()
lambda_reg = 0.4
bce_loss  = torch.nn.BCEWithLogitsLoss()

sched = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
distill_lambda = 0.5  # weight for distillation loss

# Train on Task 1 (S1) normally (no old model yet)
for epoch in range(NUM_EPOCHS):
    total_loss = 0.0
    en_idx = 0
    loop = tqdm(train_loader_S1, desc="Training Task 1")
    for X_batch, Y_batch in loop:
        mag_t, mask_t = Y_batch[:,0], Y_batch[:,1]

        optimizer.zero_grad()
        mag_p, mask_logits = model_lwf(X_batch)
        # original masked NMSE
        loss_mag  = masked_nmse(mag_p, mag_t, mask_t)
        loss_mask = bce_loss(mask_logits, mask_t)
        # composite loss
        base_loss = (
            loss_mag
            + ALPHA * loss_mask
        )
        base_loss.backward()
        total_loss += base_loss.item()
        optimizer.step()
        loop.set_postfix(
            nmse = loss_mag.item(),
            bce  = loss_mask.item(),
        )
    sched.step(total_loss/len(train_loader_S1))
# Save the Task 1 model as the old model for LwF
old_model = copy.deepcopy(model_lwf).to(device)
old_model.eval()
for param in old_model.parameters():
    param.requires_grad = False  # freeze old model

# Train on Task 2 (S2) with LwF
optimizer = torch.optim.Adam(model_lwf.parameters(), lr=1e-3)
sched = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
for epoch in range(NUM_EPOCHS):
    total_loss = 0.0
    en_idx = 0
    loop =  tqdm(train_loader_S2, desc="Training Task 2")
    for X_batch, Y_batch in loop:
        mag_t, mask_t = Y_batch[:,0], Y_batch[:,1]

        optimizer.zero_grad()
        mag_p, mask_logits = model_lwf(X_batch)
        # original masked NMSE
        loss_mag  = masked_nmse(mag_p, mag_t, mask_t)
        loss_mask = bce_loss(mask_logits, mask_t)
        # composite loss
        base_loss = (
            loss_mag
            + ALPHA * loss_mask
        )
        # Forward pass old model (with no grad)
        with torch.no_grad():
            old_pred_mag, old_pred_mask = old_model(X_batch)
        # Compute losses
        distill_loss = masked_nmse(old_pred_mag , mag_p,mask_logits )  # L2 loss between new and old outputs
        distill_loss = bce_loss(old_pred_mask, mask_logits)  # L2 loss between new and old outputs
        loss = base_loss + distill_lambda * distill_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(
            nmse = loss_mag.item(),
            bce  = loss_mask.item(),
            distill = distill_loss.item(),
        )

    sched.step(total_loss/len(train_loader_S2))
# Update old_model to the model after Task 2
old_model = copy.deepcopy(model_lwf).to(device)
old_model.eval()
for param in old_model.parameters():
    param.requires_grad = False

# Train on Task 3 (S3) with LwF
optimizer = torch.optim.Adam(model_lwf.parameters(), lr=1e-3)
sched = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
for epoch in range(NUM_EPOCHS):
    en_idx = 0
    total_loss = 0.0
    loop = tqdm(train_loader_S3, desc="Training Task 3")
    for X_batch, Y_batch in loop:
        mag_t, mask_t = Y_batch[:,0], Y_batch[:,1]

        optimizer.zero_grad()
        mag_p, mask_logits = model_lwf(X_batch)
        # original masked NMSE
        loss_mag  = masked_nmse(mag_p, mag_t, mask_t)
        loss_mask = bce_loss(mask_logits, mask_t)
        # composite loss
        base_loss = (
            loss_mag
            + ALPHA * loss_mask
        )
        # Forward pass old model (with no grad)
        with torch.no_grad():
            old_pred_mag, old_pred_mask = old_model(X_batch)
        # Compute losses
        distill_loss = masked_nmse(old_pred_mag , mag_p,mask_logits )  # L2 loss between new and old outputs
        distill_loss = bce_loss(old_pred_mask, mask_logits)  # L2 loss between new and old outputs
        loss = base_loss + distill_lambda * distill_loss
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        loop.set_postfix(
            nmse = loss_mag.item(),
            bce  = loss_mask.item(),
            distill = distill_loss.item(),
        )
    sched.step(total_loss/len(train_loader_S3))

print("\n=== NMSE Evaluation ===")
nmse_results = {
    'S1_Compact': evaluate_nmse_vs_snr_masked(model_lwf, test_loader_S1, device, SNR_LIST),
    'S2_Dense': evaluate_nmse_vs_snr_masked(model_lwf, test_loader_S2, device, SNR_LIST),
    'S3_Standard': evaluate_nmse_vs_snr_masked(model_lwf, test_loader_S3, device, SNR_LIST),
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
