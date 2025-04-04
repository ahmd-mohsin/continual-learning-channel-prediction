import argparse
import os
import torch
import csv
from dataloader import ChannelSequenceDataset
from utils import compute_device

def evaluate_nmse_vs_snr(model, dataloader, device, snr_db_list):
    """
    Evaluate the model’s NMSE over a range of SNR values.
    
    For each SNR (in dB), we add Gaussian noise to the inputs.
    NMSE for each batch is computed as:
      NMSE = sum((prediction - Y_true)^2) / sum(Y_true^2)
    and then averaged over the dataset.
    """
    model.eval()
    nmse_results = {}
    with torch.no_grad():
        for snr_db in snr_db_list:
            total_nmse = 0.0
            total_samples = 0
            for X_batch, Y_batch in dataloader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                # Compute the power of the input signal
                signal_power = torch.mean(X_batch**2)
                # Determine noise power based on SNR in dB
                noise_power = signal_power / (10 ** (snr_db / 10))
                noise_std = torch.sqrt(noise_power)
                noise = torch.randn_like(X_batch) * noise_std
                # Create noisy input
                X_noisy = X_batch + noise
                # Get channel prediction using the noisy input
                prediction = model(X_noisy)
                # Compute NMSE per batch: (error norm squared) / (true norm squared)
                batch_nmse = torch.sum((prediction - Y_batch)**2) / torch.sum(Y_batch**2)
                # Multiply by batch size and sum for weighted average
                total_nmse += batch_nmse.item() * X_batch.size(0)
                total_samples += X_batch.size(0)
            nmse_results[snr_db] = total_nmse / total_samples
            print(f"SNR: {snr_db} dB, NMSE: {nmse_results[snr_db]:.6f}")
    return nmse_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NMSE vs SNR for channel prediction model.")
    parser.add_argument("--ext", type=str, required=True, choices=["npy", "mat"],
                        help="Dataset file extension")
    parser.add_argument("--file_path", type=str, required=True,
                        help="Test dataset file path without extension")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["MLP", "CNN", "GRU", "LSTM", "TRANS"],
                        help="Model architecture type")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the saved model checkpoint")
    parser.add_argument("--snr_db", type=float, nargs="+", default=[0, 5, 10, 15, 20, 25, 30],
                        help="List of SNR values (in dB) to evaluate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    args = parser.parse_args()

    device = compute_device()

    # Load the test dataset
    test_dataset = ChannelSequenceDataset(args.file_path, args.ext, device)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                  shuffle=False, drop_last=True)

    # Instantiate the desired model architecture
    if args.model_type == "MLP":
        from model import MLPModel
        model = MLPModel(input_dim=16 * 2 * 18 * 8, hidden_dim=128, H=18, W=8).to(device)
    elif args.model_type == "CNN":
        from model import CNNModel
        model = CNNModel(in_channels=2, H=18, W=8, seq_len=16, hidden_channels=32).to(device)
    elif args.model_type == "GRU":
        from model import GRUModel
        model = GRUModel(input_dim=1, hidden_dim=32, output_dim=1, n_layers=3, H=16, W=36).to(device)
    elif args.model_type == "LSTM":
        from model import LSTMModel
        model = LSTMModel(input_dim=1, hidden_dim=32, output_dim=1, n_layers=3, H=16, W=36).to(device)
    elif args.model_type == "TRANS":
        from model import TransformerModel
        # Adjust parameters as necessary for your data shape.
        model = TransformerModel(dim_val=128, n_heads=4, n_encoder_layers=1,
                                 n_decoder_layers=1, out_channels=4, H=18, W=16, seq_len=16).to(device)
    
    # Load the trained model checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model checkpoint from {args.checkpoint}")

    # Evaluate NMSE over the provided SNR values
    nmse_results = evaluate_nmse_vs_snr(model, test_dataloader, device, args.snr_db)
    
    # Save the NMSE vs. SNR results to a CSV file
    output_csv = "nmse_vs_snr.csv"
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["SNR (dB)", "NMSE"])
        for snr, nmse in sorted(nmse_results.items()):
            writer.writerow([snr, nmse])
    print(f"NMSE vs. SNR results saved to {output_csv}")
