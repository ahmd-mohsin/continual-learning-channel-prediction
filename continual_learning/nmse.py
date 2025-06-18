import argparse
import os
import torch
import csv
from dataloader import ChannelSequenceDataset
from utils import compute_device
from tqdm import tqdm
# Import the new model classes from model.py:
from model import *
from loss import NMSELoss

def evaluate_nmse_vs_snr(model, dataloader, device, snr_db_list):
    """
    Evaluate the model’s NMSE over a range of SNR values.
    
    For each SNR (in dB), we add Gaussian noise to the inputs.
    NMSE for each batch is computed as:
      NMSE = sum((prediction - Y_true)^2) / sum(Y_true^2)
    and then averaged over the dataset.
    """
    # model.eval()
    nmse_results = {}
    criterion = NMSELoss()  
    with torch.no_grad():
        for snr_db in snr_db_list:
            total_nmse = 0.0
            total_samples = len(dataloader)
            test_total_sample = 0
            # print("########################")
            # print(total_samples)
            for X_batch, Y_batch in tqdm(dataloader, desc="Evaluating NMSE for SNR_DB: " + str(snr_db)):
                
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                if torch.sum(Y_batch**2) == 0:
                    # print(f"Skipping batch because Y_batch sum is equal to zero.")
                    continue
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
                # batch_nmse = torch.sum((prediction - Y_batch)**2) / (torch.sum(Y_batch**2))
                batch_nmse = criterion(prediction, Y_batch)
                # print("pred: ",torch.max(prediction), torch.min(prediction))
                # print("y_batch: ",torch.max(Y_batch), torch.min(Y_batch))
                # batch_nmse = criterion(prediction, Y_batch)
                # Multiply by batch size and sum for weighted average
                total_nmse += batch_nmse.item()
                test_total_sample += 1
                # print(f"Batch NMSE: {batch_nmse.item():.6f}")
            # nmse_results[snr_db] = total_nmse / total_samples
            nmse_results[snr_db] = total_nmse / test_total_sample
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
    # parser.add_argument("--checkpoint", type=str, required=True,
    #                     help="Path to the saved model checkpoint")
    parser.add_argument("--snr_db", type=float, nargs="+", default=[0, 5, 10,12,14,16,18, 20,22,24,26,28, 30],
                        help="List of SNR values (in dB) to evaluate")
    
    parser.add_argument("--test_file_path", type=str, default="../dataset/outputs/umi_compact_conf_2tx_2rx.", #  umi_standard_conf_16tx_2rx
                        help="Test file path (used only if --test_only is set)")
    
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    args = parser.parse_args()

    device = compute_device()

    # Load the test dataset
    # if args.file_path == args.test_file_path:
    #     print("Loading test same dataset...")
    #     full_dataset = ChannelSequenceDataset(args.test_file_path, args.ext, device)
    #     train_size = int(0.8 * len(full_dataset))
    #     val_size = len(full_dataset) - train_size
    #     train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    # else:
    #     print("Loading test different dataset...")
    #     test_dataset = ChannelSequenceDataset(args.test_file_path, args.ext, device)
    
    test_dataset = ChannelSequenceDataset(args.test_file_path, args.ext, device)
    
    
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                  shuffle=False, drop_last=True)



    if args.model_type == "GRU":
        model = GRUModel(
            input_dim=1,      # not strictly used—since we flatten to 2*H*W
            hidden_dim=32,
            output_dim=1,
            n_layers=3,
            H=16,
            W=36
        ).to(device)

    elif args.model_type == "LSTM":
        model = LSTMChannelPredictor(
            input_dim=1,
            hidden_dim=32,
            output_dim=1,
            n_layers=3,
            H=16,
            W=36
        ).to(device)

    elif args.model_type == "TRANS":
        model = TransformerModel(
                dim_val=128,
                n_heads=4,
                n_encoder_layers=1,
                n_decoder_layers=1,
                out_channels=4,  # Because dataloader outputs (4,18,2)
                H=18,
                W=16,
                seq_len=16
            ).to(device)   
        

    # # Load the trained model checkpoint
    # checkpoint = torch.load(args.checkpoint, map_location=device)
    # model.load_state_dict(checkpoint["model_state_dict"])
    # print(f"Loaded model checkpoint from {args.checkpoint}")

    

    # Load model checkpoint
    output_dir = os.path.basename(args.file_path)
    model_save_path = os.path.join(output_dir,args.model_type, "best_channel_predictor.pth")
    model.load_state_dict(torch.load(model_save_path)["model_state_dict"])

    # Evaluate NMSE over the provided SNR values
    nmse_results = evaluate_nmse_vs_snr(model, test_dataloader, device, args.snr_db)
    
    # Save the NMSE vs. SNR results to a CSV file
    evaluation_log_file = os.path.join(output_dir,args.model_type, f"{os.path.basename(args.test_file_path)}_nmse_vs_snr.csv")

    with open(evaluation_log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["SNR (dB)", "NMSE"])
        for snr, nmse in sorted(nmse_results.items()):
            writer.writerow([snr, nmse])
    print(f"NMSE vs. SNR results saved to {evaluation_log_file}")
