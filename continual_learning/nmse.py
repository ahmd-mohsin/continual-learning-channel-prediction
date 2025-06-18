#!/usr/bin/env python3
"""
Evaluate a channel prediction model's performance (NMSE) across a range of SNR levels.
"""

import argparse
import os
import csv
import math
from typing import List, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import ChannelSequenceDataset
from model import GRUModel, LSTMChannelPredictor, TransformerModel
from utils import compute_device
from loss import NMSELoss


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for evaluation.
    """
    parser = argparse.ArgumentParser(
        description="Compute NMSE vs. SNR for a trained channel prediction model."
    )
    parser.add_argument(
        "--ext", type=str, required=True, choices=["npy", "mat"],
        help="Dataset file extension."
    )
    parser.add_argument(
        "--file_path", type=str, required=True,
        help="Base path to test dataset (without extension)."
    )
    parser.add_argument(
        "--model_type", type=str, required=True,
        choices=["MLP", "CNN", "GRU", "LSTM", "TRANS"],
        help="Model architecture to evaluate."
    )
    parser.add_argument(
        "--snr_db", type=float, nargs='+',
        default=[0, 5, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
        help="List of SNR values (in dB) for evaluation."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for DataLoader."
    )
    return parser.parse_args()


def load_model(model_type: str, device: torch.device, checkpoint_path: str) -> torch.nn.Module:
    """
    Instantiate the model architecture and load saved weights.
    """
    # Choose model class
    if model_type == "GRU":
        model = GRUModel(
            input_dim=1,
            hidden_dim=32,
            output_dim=1,
            n_layers=3,
            H=16,
            W=36
        )
    elif model_type == "LSTM":
        model = LSTMChannelPredictor(
            input_dim=1,
            hidden_dim=32,
            output_dim=1,
            n_layers=3,
            H=16,
            W=36
        )
    else:  # TRANS
        model = TransformerModel(
            dim_val=128,
            n_heads=4,
            n_encoder_layers=1,
            n_decoder_layers=1,
            out_channels=4,
            H=18,
            W=16,
            seq_len=16
        )

    # Move model to device
    model = model.to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model


def evaluate_nmse_vs_snr(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    snr_db_list: List[float]
) -> Dict[float, float]:
    """
    Evaluate NMSE of `model` on `dataloader` across specified SNR values.

    Returns:
        A dict mapping SNR (dB) to average NMSE.
    """
    model.eval()
    criterion = NMSELoss()
    results: Dict[float, float] = {}

    with torch.no_grad():
        for snr_db in snr_db_list:
            sum_nmse = 0.0
            count = 0
            # Iterate batches, adding Gaussian noise per SNR
            for inputs, targets in tqdm(
                dataloader,
                desc=f"Eval SNR={snr_db}dB",
                leave=False
            ):
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Skip if target is zero to avoid division error
                if torch.sum(targets**2) == 0:
                    continue

                # Compute noise statistics
                signal_power = torch.mean(inputs**2)
                noise_power = signal_power / (10 ** (snr_db / 10))
                noise_std = torch.sqrt(noise_power)
                noisy_inputs = inputs + noise_std * torch.randn_like(inputs)

                # Model prediction
                pred = model(noisy_inputs)

                # Compute batch NMSE
                batch_nmse = criterion(pred, targets).item()
                sum_nmse += batch_nmse
                count += 1

            # Average NMSE for this SNR
            avg_nmse = sum_nmse / count if count > 0 else float('nan')
            results[snr_db] = avg_nmse
            print(f"SNR={snr_db} dB → NMSE={avg_nmse:.6f}")

    return results


def main() -> None:
    # Parse CLI args
    args = parse_args()

    # Determine device
    device = compute_device()

    # Prepare test dataset and DataLoader
    test_dataset = ChannelSequenceDataset(
        base_path=args.file_path,
        ext=args.ext,
        device=device
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True
    )

    # Build path to saved model checkpoint
    output_dir = os.path.basename(args.file_path)
    ckpt_dir = os.path.join(output_dir, args.model_type)
    ckpt_path = os.path.join(ckpt_dir, "best_channel_predictor.pth")

    # Load model
    model = load_model(args.model_type, device, ckpt_path)

    # Evaluate across SNR values
    nmse_results = evaluate_nmse_vs_snr(
        model, test_loader, device, args.snr_db
    )

    # Save results to CSV
    csv_path = os.path.join(
        output_dir,
        args.model_type,
        f"{os.path.basename(args.file_path)}_nmse_vs_snr.csv"
    )
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(["SNR (dB)", "NMSE"])
        for snr, nmse in sorted(nmse_results.items()):
            writer.writerow([snr, f"{nmse:.6f}"])

    print(f"Results saved to {csv_path}")


if __name__ == '__main__':
    main()
