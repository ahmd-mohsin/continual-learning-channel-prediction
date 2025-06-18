#!/usr/bin/env python3
"""
Continual Learning Training Script using Learning without Forgetting (LwF).
Trains a channel prediction model sequentially on datasets S1, S2, and S3.
"""

import argparse
import copy
import csv
import math
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import GRUModel, LSTMChannelPredictor, TransformerModel
from dataloader import get_all_datasets
from utils import compute_device, evaluate_nmse_vs_snr_masked
from loss import masked_nmse  # masked NMSE loss function


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for model configuration and strategy.
    """
    parser = argparse.ArgumentParser(
        description="Continual learning with LwF across three sequential tasks"
    )
    parser.add_argument(
        '--model_type', type=str, default='LSTM',
        choices=['LSTM', 'GRU', 'TRANS'],
        help='Model architecture: LSTM, GRU, or Transformer'
    )
    parser.add_argument(
        '--strategy', type=str, default='lwf',
        choices=['lwf'],
        help='Continual learning strategy (currently only lwf supported)'
    )
    return parser.parse_args()


def build_model(model_type: str, device: torch.device) -> nn.Module:
    """
    Instantiate the chosen model and move it to the target device.
    """
    if model_type == 'GRU':
        return GRUModel(input_dim=1, hidden_dim=32, output_dim=1,
                        n_layers=3, H=16, W=9).to(device)
    if model_type == 'LSTM':
        return LSTMChannelPredictor().to(device)
    # TRANS (Transformer) fallback
    return TransformerModel(dim_val=128, n_heads=4,
                            n_encoder_layers=1, n_decoder_layers=1,
                            out_channels=2, H=16, W=9).to(device)


def train_single_task(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    alpha: float,
    desc: str
) -> None:
    """
    Train model on one task without distillation (Task 1).
    """
    model.train()
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_loss = 0.0
        progress = tqdm(loader, desc=f"{desc} Epoch {epoch}/{NUM_EPOCHS}")
        for X, Y in progress:
            mag_true, mask_true = Y[:, 0], Y[:, 1]
            optimizer.zero_grad()
            mag_pred, mask_logits = model(X)

            # Compute masked NMSE and BCE losses
            loss_mag = masked_nmse(mag_pred, mag_true, mask_true)
            loss_mask = nn.BCEWithLogitsLoss()(mask_logits, mask_true)
            loss = loss_mag + alpha * loss_mask

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix(nmse=loss_mag.item(), bce=loss_mask.item())

        # Adjust learning rate based on average loss
        scheduler.step(epoch_loss / len(loader))


def train_with_lwf(
    model: nn.Module,
    old_model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    alpha: float,
    distill_lambda: float,
    desc: str
) -> None:
    """
    Train model on new task with LwF distillation from old_model.
    """
    model.train()
    old_model.eval()
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_loss = 0.0
        progress = tqdm(loader, desc=f"{desc} Epoch {epoch}/{NUM_EPOCHS}")
        for X, Y in progress:
            mag_true, mask_true = Y[:, 0], Y[:, 1]
            optimizer.zero_grad()

            # Forward new model
            mag_pred, mask_logits = model(X)

            # Task loss
            loss_mag = masked_nmse(mag_pred, mag_true, mask_true)
            loss_mask = nn.BCEWithLogitsLoss()(mask_logits, mask_true)
            base_loss = loss_mag + alpha * loss_mask

            # Distillation loss from old model
            with torch.no_grad():
                old_mag, old_mask = old_model(X)
            distill_mag = masked_nmse(old_mag, mag_pred, mask_logits)
            distill_mask = nn.BCEWithLogitsLoss()(old_mask, mask_logits)
            distill_loss = distill_mag + alpha * distill_mask

            # Combined loss
            loss = base_loss + distill_lambda * distill_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix(
                nmse=loss_mag.item(),
                bce=loss_mask.item(),
                distill=distill_loss.item()
            )

        scheduler.step(epoch_loss / len(loader))


def main() -> None:
    # Parse arguments
    args = parse_args()

    # Set device and hyperparameters
    device = compute_device()
    global NUM_EPOCHS  # number of epochs defined below
    NUM_EPOCHS = 100
    alpha = ALPHA
    distill_lambda = 0.5

    # Load datasets for tasks S1, S2, S3
    print("Loading datasets...")
    (train_S1, test_S1, loader_S1,
     train_S2, test_S2, loader_S2,
     train_S3, test_S3, loader_S3) = get_all_datasets(
         data_dir="../dataset/outputs/",
         batch_size=BATCH_SIZE,
         dataset_id="all",
         normalization="log_min_max",
         per_user=True,
         seq_len=SEQ_LEN
    )
    print("Datasets loaded successfully.")

    # Instantiate model and optimizer for Task 1
    model = build_model(args.model_type, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    # Task 1: train without distillation
    train_single_task(model, loader_S1, optimizer, scheduler,
                      alpha, desc="Task 1 (S1)")

    # Preserve model as old_model for distillation
    old_model = copy.deepcopy(model).to(device)
    for param in old_model.parameters():
        param.requires_grad = False

    # Task 2: train with LwF
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )
    train_with_lwf(model, old_model, loader_S2, optimizer, scheduler,
                   alpha, distill_lambda, desc="Task 2 (S2)")

    # Update old_model after Task 2
    old_model = copy.deepcopy(model).to(device)
    for param in old_model.parameters():
        param.requires_grad = False

    # Task 3: train with LwF
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )
    train_with_lwf(model, old_model, loader_S3, optimizer, scheduler,
                   alpha, distill_lambda, desc="Task 3 (S3)")

    # Final evaluation and CSV export
    print("\n=== NMSE Evaluation ===")
    results: Dict[str, Dict[int, float]] = {
        'S1_Compact': evaluate_nmse_vs_snr_masked(model, test_S1, device, SNR_LIST),
        'S2_Dense'  : evaluate_nmse_vs_snr_masked(model, test_S2, device, SNR_LIST),
        'S3_Standard': evaluate_nmse_vs_snr_masked(model, test_S3, device, SNR_LIST)
    }

    csv_rows: List[List] = [['Task', 'SNR', 'NMSE Masked', 'NMSE (dB)']]
    for task, res in results.items():
        for snr, nmse in res.items():
            nmse_db = -10 * math.log10(nmse + 1e-12)
            print(f"Task {task} | SNR {snr:2d} → NMSE {nmse:.6f} | {nmse_db:.2f} dB")
            csv_rows.append([task, snr, f"{nmse:.6f}", f"{nmse_db:.2f}"])

    output_csv = f"{args.strategy}_{args.model_type}_nmse_results.csv"
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"\nResults saved to {output_csv}")


if __name__ == '__main__':
    main()
