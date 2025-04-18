#!/usr/bin/env python3
# er.py  — Experience Replay with optional CLEAR‑style self‑distillation

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

from model import LSTMModel
from dataloader import get_all_datasets
from utils import compute_device
from nmse import evaluate_nmse_vs_snr

# ---------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--sampling', type=str, default='reservoir',
                    choices=['reservoir', 'lars'],
                    help='Sampling strategy for the replay buffer')
parser.add_argument('--run_name', type=str, default='er',
                    help='Name tag for output CSV file')
parser.add_argument('--use_distill', action='store_true',
                    help='Enable CLEAR‑style behavioral cloning on replay')
parser.add_argument('--lambda_bc', type=float, default=1.0,
                    help='Weight of behavioral‑cloning loss term')
args = parser.parse_args()

# ---------------------------------------------------------------------
# Hyperparameters & globals
# ---------------------------------------------------------------------
snr_list        = [0, 5, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
batch_size      = 8
num_epochs      = 10
memory_capacity = 1000

device   = compute_device()
model_er = LSTMModel(input_dim=1, hidden_dim=32, output_dim=1,
                     n_layers=3, H=36, W=64).to(device)
optimizer = torch.optim.Adam(model_er.parameters(), lr=1e-5)
criterion = nn.MSELoss(reduction='none')

# ---------------------------------------------------------------------
# Replay buffer state
# ---------------------------------------------------------------------
memory_x:       List[torch.Tensor] = []
memory_y:       List[torch.Tensor] = []
memory_teacher: List[torch.Tensor] = []
memory_loss:    List[float]        = []
seen_examples   = 0

def lars_pick_victim() -> int:
    losses = torch.tensor(memory_loss, dtype=torch.float)
    inv    = 1.0 / (losses + 1e-8)
    probs  = inv / inv.sum()
    return torch.multinomial(probs, 1).item()

def reservoir_add(x, y, teacher_pred, loss_val):
    """Insert into replay buffer via reservoir or LARS."""
    global seen_examples
    seen_examples += 1

    if len(memory_x) < memory_capacity:
        memory_x.append(x)
        memory_y.append(y)
        memory_teacher.append(teacher_pred)
        memory_loss.append(loss_val)
        return

    j = random.randint(0, seen_examples - 1)
    if j < memory_capacity:
        victim = lars_pick_victim() if args.sampling == 'lars' else j
        memory_x[victim]       = x
        memory_y[victim]       = y
        memory_teacher[victim] = teacher_pred
        memory_loss[victim]    = loss_val
    # if len(memory_x) < memory_capacity:
    #     memory_x.append(x.cpu())
    #     memory_y.append(y.cpu())
    #     memory_teacher.append(teacher_pred.cpu())
    #     memory_loss.append(loss_val)
    #     return

    # j = random.randint(0, seen_examples - 1)
    # if j < memory_capacity:
    #     victim = lars_pick_victim() if args.sampling == 'lars' else j
    #     memory_x[victim]       = x.cpu()
    #     memory_y[victim]       = y.cpu()
    #     memory_teacher[victim] = teacher_pred.cpu()
    #     memory_loss[victim]    = loss_val

# ---------------------------------------------------------------------
# Dataset wrapper for distillation
# ---------------------------------------------------------------------
class DistillDataset(Dataset):
    """Returns (x, y, y_teacher, is_replay flag)."""
    def __init__(self, current_ds, replay_ds=None):
        self.cur     = current_ds
        self.rep     = replay_ds
        self.len_cur = len(current_ds)
        self.len_rep = 0 if replay_ds is None else len(replay_ds)

    def __len__(self):
        return self.len_cur + self.len_rep

    def __getitem__(self, idx):
        if idx < self.len_cur:
            x, y = self.cur[idx]
            return x, y, torch.zeros_like(y), False
        else:
            x, y, y_t = self.rep[idx - self.len_cur]
            return x, y, y_t, True

# ---------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------
def build_loader(task_ds):
    # prepare replay TensorDataset if we have buffer entries
    replay_ds = None
    if len(memory_x) > 0:
        replay_ds = TensorDataset(
            torch.stack(memory_x),
            torch.stack(memory_y),
            torch.stack(memory_teacher)
        )
        print("Replay buffer size:", len(replay_ds))
    else:
        print("Replay buffer is empty.")
            
    # wrap in DistillDataset (replay_ds may be None)
    full_ds = DistillDataset(task_ds, replay_ds)
    return DataLoader(full_ds,
                      batch_size=batch_size,
                      shuffle=True,
                      drop_last=True)
# ---------------------------------------------------------------------
# Single epoch training
# ---------------------------------------------------------------------
def train_epoch(loader):
    model_er.train()
    total_loss = 0.0

    for X, Y, Y_teacher, is_rep in tqdm(loader):
        X, Y, Y_teacher, is_rep = (X.to(device), Y.to(device),
                                   Y_teacher.to(device),
                                   is_rep.to(device))

        optimizer.zero_grad()
        y_pred       = model_er(X)
        teacher_pred = y_pred.detach().clone()

        # supervised loss
        loss_sup = criterion(y_pred, Y).view(y_pred.size(0), -1).mean(1).mean()

        # distillation loss on replay
        if args.use_distill and is_rep.any():
            mask     = is_rep.view(-1, 1, 1)
            loss_dist = ((y_pred - Y_teacher)**2 * mask).sum() / mask.sum()
            loss      = loss_sup + args.lambda_bc * loss_dist
        else:
            loss = loss_sup

        loss.backward()
        optimizer.step()

        # add to buffer
        per_sample = criterion(y_pred, Y).view(y_pred.size(0), -1).mean(1)
        for i in range(X.size(0)):
            reservoir_add(
                X[i].detach(),
                Y[i].detach(),
                teacher_pred[i].detach(),
                per_sample[i].item()
            )
        total_loss += loss.item()

    return total_loss / len(loader)

# ---------------------------------------------------------------------
# Task training wrapper
# ---------------------------------------------------------------------
def train_on_task(train_ds, task_name):
    phase = "initial" if len(memory_x)==0 else "replay"
    print(f"\n=== Training on {task_name} ({phase}) ===")
    loader = build_loader(train_ds)
    for ep in range(1, num_epochs+1):
        ep_loss = train_epoch(loader)
        print(f"  Epoch {ep:02d}/{num_epochs} → loss {ep_loss}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading datasets...")
    data_dir = "../dataset/outputs/"
    train_S1, test_S1, train_S2, test_S2, train_S3, test_S3, \
    _,        test_loader_S1, _,         test_loader_S2, _,         test_loader_S3 = \
        get_all_datasets(data_dir,batch_size=batch_size, dataset_id="all")
    print("Datasets loaded.")

    # Task 1: S1 (no replay)
    train_on_task(train_S1, "S1")

    # Task 2: S2 + replay from S1
    train_on_task(train_S2, "S2")

    # Task 3: S3 + replay from S1+S2
    train_on_task(train_S3, "S3")

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------
    print("\n=== NMSE Evaluation ===")
    nmse_results = {
        'S1': evaluate_nmse_vs_snr(model_er, test_loader_S1, device, snr_list),
        'S2': evaluate_nmse_vs_snr(model_er, test_loader_S2, device, snr_list),
        'S3': evaluate_nmse_vs_snr(model_er, test_loader_S3, device, snr_list),
    }

    csv_rows = [['Task', 'SNR', 'NMSE']]
    for task, res in nmse_results.items():
        for snr, nmse in res.items():
            print(f"Task {task} | SNR {snr:2d} → NMSE {nmse:.4f}")
            csv_rows.append([task, snr, f"{nmse:.6f}"])

    if args.use_distill:
        csv_path = f"{args.sampling}_distill_nmse_results.csv"
    else:
        csv_path = f"{args.sampling}_nmse_results.csv"
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerows(csv_rows)

    print(f"\nResults saved to {csv_path}")
