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
from utils import compute_device, evaluate_nmse_vs_snr_masked
from model      import LSTMChannelPredictor          # 2-channel LSTM (mag + mask)

from loss import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn

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

BATCH_SIZE = 2048
SEQ_LEN    = 32
NUM_EPOCHS = 30
ALPHA      = 0.2         # weight for BCE mask loss
LR         = 1e-4
SNR_LIST   = [0,5,10,12,14,16,18,20,22,24,26,28,30]
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
    model_er = LSTMChannelPredictor().to(device)

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
    


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.LSTM)):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
model_er.apply(init_weights)

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

def reservoir_add(x, y, teacher_pred , loss_val):
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
            x, y, y_t, = self.rep[idx - self.len_cur]
            return x, y, y_t, True


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
                      batch_size=BATCH_SIZE,
                      shuffle=True,
                      drop_last=True,
                      pin_memory=False)   # faster H2D for task_ds batches


# optimizer = torch.optim.Adam(model_er.parameters(), lr=LR)
# bce_loss  = torch.nn.BCEWithLogitsLoss(reduction='none')
# sched = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)


def train_epoch(epoch, loader, optimizer=None, bce_loss=None, sched=None):
    model_er.train()
    total_loss = 0.0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

    for X, Y, Y_teacher, is_rep in pbar:
        X, Y, Y_teacher, is_rep = (X.to(device), Y.to(device),
                                   Y_teacher.to(device),
                                   is_rep.to(device))



        mag_t, mask_t = Y[:,0], Y[:,1]
        Y_teacher_mag,Y_teacher_mask = Y_teacher[:,0], Y_teacher[:,1]

        optimizer.zero_grad()
        mag_p, mask_logits = model_er(X)

        # original masked NMSE
        loss_mag_per_sample   = masked_nmse_per_sample(mag_p, mag_t, mask_t)
        loss_mag_mean   = masked_nmse(mag_p, mag_t, mask_t)
        
        
        # BCE mean loss
        loss_mask_mean = bce_loss(mask_logits, mask_t).view(mask_logits.size(0), -1).mean(1).mean()
        # tt = bce_loss(mask_logits, mask_t) 
        # print(tt.shape, mask_logits.shape, mask_t.shape)
        # BCE per sample loss
        loss_mask_per_sample = bce_loss(mask_logits, mask_t).view(mask_logits.size(0), -1).mean(1)
        
        total_supervised_loss_mean_loss = loss_mag_mean + ALPHA * loss_mask_mean
        
        
        # distillation loss on replay
        if args.use_distill and is_rep.any():            
            teacher_mag_mean_loss   = masked_nmse(Y_teacher_mag, mag_p, mask_logits)
            teacher_mask_mean_loss = bce_loss(Y_teacher_mask, mask_logits).view(mask_logits.size(0), -1).mean(1).mean()
            total_dist_mean_loss = teacher_mag_mean_loss + ALPHA * teacher_mask_mean_loss
            final_mean_loss      = total_supervised_loss_mean_loss + args.lambda_bc * total_dist_mean_loss
        else:
            final_mean_loss = total_supervised_loss_mean_loss

        teacher_pred_mag, teacher_pred_mask  = mag_p.detach().clone(), mask_logits.detach().clone()
        final_mean_loss.backward()
        clip_grad_norm_(model_er.parameters(), max_norm=1.0)
        optimizer.step()
        
        teacher_pred_stacked = torch.stack([teacher_pred_mag, teacher_pred_mask], dim=1)
        for i in range(X.size(0)):
            reservoir_add(
                X[i].detach().cpu(),
                Y[i].detach().cpu(),
                teacher_pred_stacked[i].detach().cpu(),
                loss_mag_per_sample[i].item() + ALPHA * loss_mask_per_sample[i].item()
            )
        
        pbar.set_postfix(
            nmse = loss_mag_mean.item(),
            bce  = loss_mask_mean.item(),
        )
        total_loss += final_mean_loss.item()
        # print(f"  ↳ avg train-loss: {total_loss}")
    sched.step(total_loss / len(loader))

    return total_loss / len(loader)

# ---------------------------------------------------------------------
# Task training wrapper
# ---------------------------------------------------------------------
def train_on_task(train_ds, task_name):
    optimizer = torch.optim.Adam(model_er.parameters(), lr=LR)
    bce_loss  = torch.nn.BCEWithLogitsLoss(reduction='none')
    sched = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    phase = "initial" if len(memory_x)==0 else "replay"
    print(f"\n=== Training with {args.model_type} {args.sampling} on {task_name} ({phase}) ===")
    loader = build_loader(train_ds)
    for ep in range(1, NUM_EPOCHS+1):
        ep_loss = train_epoch(ep, loader, optimizer, bce_loss, sched)


if __name__ == "__main__":
    print("Loading datasets...")
    data_dir = "../dataset/outputs/"
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
    print("Datasets loaded.")

    # Task 1: S1 (no replay)
    train_on_task(train_S1, "S1")

    # Task 2: S2 + replay from S1
    train_on_task(train_S2, "S2")

    # Task 3: S3 + replay from S1+S2
    train_on_task(train_S3, "S3")


    print("\n=== NMSE Evaluation ===")
    nmse_results = {
        'S1_Compact': evaluate_nmse_vs_snr_masked(model_er, test_loader_S1, device, SNR_LIST),
        'S2_Dense': evaluate_nmse_vs_snr_masked(model_er, test_loader_S2, device, SNR_LIST),
        'S3_Standard': evaluate_nmse_vs_snr_masked(model_er, test_loader_S3, device, SNR_LIST),
    }

    csv_rows = [['Task', 'SNR', 'NMSE Masked', 'NMSE (dB)']]
    for task, res in nmse_results.items():
        for snr, nmse in res.items():
            nmse_db = -10 * math.log10(nmse + 1e-12)  # add tiny eps in case nmse==0
            print(f"Task {task} | SNR {snr:2d} → NMSE {nmse} | {nmse_db} dB")
            csv_rows.append([task, snr, f"{nmse}", f"{nmse_db}"])
    if args.use_distill:
        csv_path = f"{args.sampling}_{args.model_type}_distill_nmse_results.csv"
    else:
        csv_path = f"{args.sampling}_{args.model_type}_nmse_results.csv"
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerows(csv_rows)

    print(f"\nResults saved to {csv_path}")
