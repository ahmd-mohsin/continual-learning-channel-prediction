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

from model import *
from dataloader import get_all_datasets
from utils import compute_device, evaluate_model
from nmse import evaluate_nmse_vs_snr
from loss import NMSELoss
# ---------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--sampling', type=str, default='reservoir',
                    choices=['reservoir', 'lars'],
                    help='Sampling strategy for the replay buffer')
parser.add_argument('--model_type', type=str, default='LSTM',
                    choices=['LSTM', 'GRU', 'TRANS'],
                    help='MODEL_TYPE: LSTM')


parser.add_argument('--run_name', type=str, default='er',
                    help='Name tag\ for output CSV file')
parser.add_argument('--use_distill', action='store_true',
                    help='Enable CLEAR‑style behavioral cloning on replay')
parser.add_argument('--lambda_bc', type=float, default=1.0,
                    help='Weight of behavioral‑cloning loss term')
args = parser.parse_args()

# ---------------------------------------------------------------------
# Hyperparameters & globals
# ---------------------------------------------------------------------
snr_list        = [0, 5, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
batch_size      = 2046
num_epochs      = 30
memory_capacity = 5000

device   = compute_device()
# model_er = LSTMModel(input_dim=1, hidden_dim=32, output_dim=1,
                    #  n_layers=3, H=16, W=18).to(device)

if args.model_type == "GRU":
    model_er = GRUModel(
        input_dim=1,      # not strictly used—since we flatten to 2*H*W
        hidden_dim=32,
        output_dim=1,
        n_layers=3,
        H=16,
        W=9
    ).to(device)

elif args.model_type == "LSTM":
    model_er = LSTMModel(
        input_dim=1,
        hidden_dim=32,
        output_dim=1,
        n_layers=3,
        H=16,
        W=9
    ).to(device)

elif args.model_type == "TRANS":
    model_er = TransformerModel(
            dim_val=128,
            n_heads=4,
            n_encoder_layers=1,
            n_decoder_layers=1,
            out_channels=2,  # Because dataloader outputs (4,18,2)
            H=16,
            W=9,
        ).to(device)
    

# ---------------------------------------------------------------------
# Replay buffer state
# ---------------------------------------------------------------------
memory_x:       List[torch.Tensor] = []
memory_y:       List[torch.Tensor] = []
memory_teacher: List[torch.Tensor] = []
memory_loss:    List[float]        = []
seen_examples   = 0

def lars_pick_victim() -> int:
    global memory_loss
    losses = torch.tensor(memory_loss, dtype=torch.float)
    inv    = 1.0 / (losses + 1e-8)
    probs  = inv / inv.sum()
    return torch.multinomial(probs, 1).item()

def reservoir_add(x, y, teacher_pred, loss_val):
    """Insert into replay buffer via reservoir or LARS."""
    global seen_examples, memory_x, memory_y, memory_teacher, memory_loss
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
        # --- move replay memory to GPU only once ---
        x_buf = torch.stack(memory_x).to(device, non_blocking=True)
        y_buf = torch.stack(memory_y).to(device, non_blocking=True)
        t_buf = torch.stack(memory_teacher).to(device, non_blocking=True)

        replay_ds = TensorDataset(x_buf, y_buf, t_buf)
        print(f"Replay buffer size (GPU): {len(replay_ds)}")
    else:
        print("Replay buffer is empty.")

    # DistillDataset merges the two sources transparently
    full_ds = DistillDataset(task_ds, replay_ds)
    pin = (replay_ds is None)     # only pin when every sample is on CPU
    return DataLoader(full_ds,
                      batch_size=batch_size,
                      shuffle=True,
                      drop_last=True,
                      pin_memory=False)   # faster H2D for task_ds batches
# ---------------------------------------------------------------------
# Single epoch training
# ---------------------------------------------------------------------
def train_epoch(loader):
    optimizer = torch.optim.Adam(model_er.parameters(), lr=1e-5)
    criterion = NMSELoss(reduction='none')
    model_er.train()
    total_loss = 0.0
    en_idx = 0
    for X, Y, Y_teacher, is_rep in tqdm(loader):
        X, Y, Y_teacher, is_rep = (X.to(device), Y.to(device),
                                   Y_teacher.to(device),
                                   is_rep.to(device))

        # if en_idx > 5:
        #     break
        # en_idx += 1


        optimizer.zero_grad()
        y_pred       = model_er(X)
        # print(f"X: {X.shape} , y_pred: {y_pred.shape}, Y: {Y.shape}, Y_teacher: {Y_teacher.shape}")
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
                X[i].detach().cpu(),
                Y[i].detach().cpu(),
                teacher_pred[i].detach().cpu(),
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
    print("\n=== Loss Evaluation ===")
    if args.use_distill:
        test_loss_csv_path = f"{args.sampling}_{args.model_type}_distill_loss_results.csv"
    else:
        test_loss_csv_path = f"{args.sampling}_{args.model_type}_loss_results.csv"
    loss_results = {
        'S1_Compact_loss': evaluate_model(model_er, test_loader_S1, device),
        'S2_Dense_loss': evaluate_model(model_er, test_loader_S2, device),
        'S3_Standard_loss': evaluate_model(model_er, test_loader_S3, device),
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



    print("\n=== NMSE Evaluation ===")
    nmse_results = {
        'S1_Compact': evaluate_nmse_vs_snr(model_er, test_loader_S1, device, snr_list),
        'S2_Dense': evaluate_nmse_vs_snr(model_er, test_loader_S2, device, snr_list),
        'S3_Standard': evaluate_nmse_vs_snr(model_er, test_loader_S3, device, snr_list),
    }

    csv_rows = [['Task', 'SNR', 'NMSE']]
    for task, res in nmse_results.items():
        for snr, nmse in res.items():
            print(f"Task {task} | SNR {snr:2d} → NMSE {nmse}")
            csv_rows.append([task, snr, f"{nmse}"])

    if args.use_distill:
        csv_path = f"{args.sampling}_{args.model_type}_distill_nmse_results.csv"
    else:
        csv_path = f"{args.sampling}_{args.model_type}_nmse_results.csv"
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerows(csv_rows)

    print(f"\nResults saved to {csv_path}")
