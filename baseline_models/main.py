import argparse
import os
from model import CustomLSTMModel, load_model
from dataloader import ChannelSequenceDataset
from loss import ScaledMSELoss
from utils import train_model, evaluate_model, compute_device
import torch

def main():
    parser = argparse.ArgumentParser(description="Train an LSTM model for channel prediction.")
    parser.add_argument("--ext", type=str, required=True, choices=["npy", "mat"], help="Dataset file extension (npy or mat)")
    parser.add_argument("--file_path", type=str, default="../dataset/outputs/umi_standard_conf_16tx_2rx.", help="Dataset file path without extension")
    args = parser.parse_args()

    torch.manual_seed(42)
    
    device = compute_device()
    file_path = args.file_path
    dataset_file = file_path + '.' + args.ext  # Add extension to the file path
    full_dataset = ChannelSequenceDataset(file_path, args.ext, device)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    batch_size = 16
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    print(f"Created dataloaders with batch size {batch_size}")

    # Initialize model and move to device
    model = CustomLSTMModel().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # Create a folder based on the file path to store log and model files
    output_dir = os.path.join(os.path.basename(file_path))
    os.makedirs(output_dir, exist_ok=True)

    # Define the log file and model save paths
    training_log_file = os.path.join(output_dir, "training_log.csv")
    print(f"Training log file: {training_log_file}")
    evaluation_log_file = os.path.join(output_dir, "evaluation_log.csv")
    model_save_path = os.path.join(output_dir, "best_channel_predictor.pth")

    print(f"Starting training...")
    model = train_model(model=model, dataloader=train_dataloader, device=device, num_epochs=10, learning_rate=1e-3, log_file=training_log_file, model_save_path=model_save_path)
    
    print("Evaluating model...")
    val_loss = evaluate_model(model, val_dataloader, device, log_file=evaluation_log_file)
    
    print("Training completed!")
    print(f"Final validation loss: {val_loss:.6f}")

if __name__ == "__main__":
    main()
