import argparse
import random
import torch
import torch.nn as nn
import os
import sys
import csv
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from tqdm import tqdm

from model import LSTMModel
from dataloader import ChannelSequenceDataset, get_all_datasets
from utils import compute_device
from nmse import evaluate_nmse_vs_snr

# ----------------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--sampling', type=str, default='reservoir', choices=['reservoir', 'lars'],
                    help='Sampling strategy: reservoir or lars')
parser.add_argument('--run_name', type=str, default='er',
                    help='Name tag for output CSV file')
args = parser.parse_args()

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
snr_list = [0, 5, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
batch_size = 16

print("Loading datasets...")
data_dir = "../dataset/outputs/"
train_S1, test_S1, train_S2, test_S2, train_S3, test_S3, \
train_loader_S1, test_loader_S1, train_loader_S2, test_loader_S2, \
train_loader_S3, test_loader_S3 = get_all_datasets(data_dir, dataset_id=1)
print("Loaded datasets successfully.")

device = compute_device()
model_er = LSTMModel(input_dim=1, hidden_dim=32, output_dim=1, n_layers=3, H=36, W=64).to(device)
optimizer = torch.optim.Adam(model_er.parameters(), lr=1e-5)
criterion = nn.MSELoss(reduction='none')

num_epochs = 10
memory_x, memory_y, memory_loss = [], [], []
seen_examples = 0
memory_capacity = 32000

# ----------------------------------------------------------------------------
# Replay Buffer Functions
# ----------------------------------------------------------------------------
def reservoir_add(x, y, loss_val):
    global seen_examples
    seen_examples += 1
    if len(memory_x) < memory_capacity:
        memory_x.append(x.cpu())
        memory_y.append(y.cpu())
        memory_loss.append(loss_val)
        return
    j = random.randint(0, seen_examples - 1)
    if j < memory_capacity:
        victim = lars_pick_victim() if args.sampling == 'lars' else j
        memory_x[victim] = x.cpu()
        memory_y[victim] = y.cpu()
        memory_loss[victim] = loss_val

def lars_pick_victim():
    losses = torch.tensor(memory_loss, dtype=torch.float)
    inv_loss = 1.0 / (losses + 1e-8)
    probs = inv_loss / inv_loss.sum()
    return torch.multinomial(probs, 1).item()
    # probs = torch.tensor([0.2, 0.5, 0.3])
    # a = torch.multinomial(probs, 1).item() a-> 1
# ----------------------------------------------------------------------------
# Training Function
# ----------------------------------------------------------------------------
def train(model, loader):
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0
        for X_batch, Y_batch in tqdm(loader, desc="Training", leave=False):
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)
            raw_loss = criterion(pred, Y_batch)
            per_sample_loss = raw_loss.view(raw_loss.size(0), -1).mean(1)
            batch_loss = per_sample_loss.mean()
            batch_loss.backward()
            optimizer.step()

            for i in range(X_batch.size(0)):
                reservoir_add(X_batch[i].detach(), Y_batch[i].detach(), per_sample_loss[i].item())

            running_loss += batch_loss.item()
        avg_loss = running_loss / len(loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

# ----------------------------------------------------------------------------
# Task 1: Train on S1
# ----------------------------------------------------------------------------
print("Train on Task 1 (S1) normally")
train(model_er, train_loader_S1)

# ----------------------------------------------------------------------------
# Task 2: Train on S2 + Replay from S1
# ----------------------------------------------------------------------------
print("Train on Task 2 (S2) with memory replay")
X_mem = torch.stack(memory_x)
Y_mem = torch.stack(memory_y)
X_mem = X_mem.to(device)
Y_mem = Y_mem.to(device)
memory_dataset = TensorDataset(X_mem, Y_mem)
combined_train_S2 = ConcatDataset([train_S2, memory_dataset])
train_loader_combined_S2 = DataLoader(combined_train_S2, batch_size=batch_size, shuffle=True, drop_last=True)
optimizer = torch.optim.Adam(model_er.parameters(), lr=1e-5)
train(model_er, train_loader_combined_S2)

# ----------------------------------------------------------------------------
# Task 3: Train on S3 + Replay from S1+S2
# ----------------------------------------------------------------------------
print("Train on Task 3 (S3) with memory replay")
X_mem = torch.stack(memory_x)
Y_mem = torch.stack(memory_y)
X_mem = X_mem.to(device)
Y_mem = Y_mem.to(device)
memory_dataset = TensorDataset(X_mem, Y_mem)
combined_train_S3 = ConcatDataset([train_S3, memory_dataset])
train_loader_combined_S3 = DataLoader(combined_train_S3, batch_size=batch_size, shuffle=True, drop_last=True)
optimizer = torch.optim.Adam(model_er.parameters(), lr=1e-5)
train(model_er, train_loader_combined_S3)

# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
print("Experience Replay Method - NMSE on each task across SNRs:")
nmse_results_er = {}
nmse_results_er['S1'] = evaluate_nmse_vs_snr(model_er, test_loader_S1, device, snr_list)
nmse_results_er['S2'] = evaluate_nmse_vs_snr(model_er, test_loader_S2, device, snr_list)
nmse_results_er['S3'] = evaluate_nmse_vs_snr(model_er, test_loader_S3, device, snr_list)

csv_rows = [['Task', 'SNR', 'NMSE']]
for task, nmse_vs_snr in nmse_results_er.items():
    for snr, nmse in nmse_vs_snr.items():
        print(f"Task {task}: SNR {snr}: NMSE {nmse:.4f}")
        csv_rows.append([task, snr, f"{nmse:.6f}"])

csv_filename = f"{args.run_name}_nmse_results.csv"
with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)

print(f"NMSE results saved to '{csv_filename}'")
