import torch
import torch.nn as nn
import csv
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from dataloader import ChannelSequenceDataset
# from model import CustomLSTMModel
from loss import CustomLoss  # Import your custom loss if needed (adjust based on your loss function)
from utils import compute_device
# from model import load_model
import os
def evaluate_model(model, dataloader, criterion, device, log_file="testing_log.csv"):
    model.eval()  # Ensure the model is in evaluation mode
    total_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():  # Disable gradients for evaluation
        progress_bar = tqdm(dataloader, desc="Testing Model")
        for index, (X_batch, Y_batch) in enumerate(progress_bar):
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, Y_batch)
            total_loss += loss.item()  # Accumulate loss
            progress_bar.set_postfix(batch_loss=loss.item())

    avg_loss = total_loss / num_batches  # Compute average loss

    # Log the results to a CSV file
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Testing Loss", avg_loss])
    
    print(f"Testing Completed. Average Loss: {avg_loss:.4f}")
    return avg_loss

def main():
    parser = argparse.ArgumentParser(description="Test an LSTM model on a new dataset.")
    parser.add_argument("--model_path", default="./umi_dense_conf_8tx_2rx./best_channel_predictor.pth",type=str, help="Path to the trained model file")
    parser.add_argument("--dataset_path",default="../dataset/outputs/umi_standard_conf_16tx_2rx.", type=str, help="Path to the testing dataset")
    parser.add_argument("--ext", type=str, default="mat", choices=["npy", "mat"], help="Dataset file extension (npy or mat)")
    args = parser.parse_args()
    
    device = compute_device()
    
    # Load dataset and prepare DataLoader
    full_dataset = ChannelSequenceDataset(args.dataset_path, args.ext, device)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=True)
    
    print("Initializing model...")
    
    # Select the loss function (you can use your custom loss function here)
    # criterion = CustomLoss()  # For example, using Scaled MSE Loss
    criterion = nn.MSELoss()  # For example, using Scaled MSE Loss
    output_dir = os.path.join(os.path.basename(args.dataset_path))

    # Define the log file and model save paths
    testing_log_file = os.path.join(os.path.dirname(args.model_path), f"{output_dir}_testing_log.csv")
    print("Evaluating model...")
    test_loss = evaluate_model(model, test_dataloader, criterion, device, log_file=testing_log_file)
    
    print("Testing completed!")
    print(f"Final testing loss: {test_loss:.6f}")

if __name__ == "__main__":
    main()
