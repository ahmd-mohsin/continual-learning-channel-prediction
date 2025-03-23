import torch
import torch.nn as nn
import csv
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from train_cuda_numpy import ChannelSequenceDataset  # Update with actual import
from train_cuda_numpy import CustomLSTMModel  # Update with actual import

def compute_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA for computation")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) for computation")
    else:
        device = torch.device("cpu")
        print("Using CPU for computation")
    return device

def load_model(model_path, input_size, hidden_size, num_layers, output_size, device):
    model = CustomLSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded trained model from {model_path}")
    return model

def evaluate_model(model, dataloader, criterion, device, log_file="testing_log.csv"):
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Testing Model")
        for index, (X_batch, Y_batch) in enumerate(progress_bar):
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, Y_batch)
            total_loss += loss.item()
            progress_bar.set_postfix(batch_loss=loss.item())

    avg_loss = total_loss / num_batches

    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Testing Loss", avg_loss])
    
    print(f"Testing Completed. Average Loss: {avg_loss:.4f}")
    return avg_loss

def main():
    parser = argparse.ArgumentParser(description="Test an LSTM model on a new dataset.")
    parser.add_argument("--model_path", default="./best_channel_predictor.pth",type=str, help="Path to the trained model file")
    parser.add_argument("--dataset_path",default="../dataset/outputs/umi_compact_conf_8tx_2rx.mat.", type=str, help="Path to the testing dataset")
    parser.add_argument("--ext", type=str, default="npy", choices=["npy", "mat"], help="Dataset file extension (npy or mat)")
    args = parser.parse_args()
    
    device = compute_device()
    
    test_dataset = ChannelSequenceDataset(args.dataset_path, args.ext, device)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=True)
    
    print("Initializing model...")
    input_size = 1
    hidden_size = 16
    num_layers = 16
    output_size = 1
    
    model = load_model(args.model_path, input_size, hidden_size, num_layers, output_size, device)
    
    criterion = nn.MSELoss()
    print("Evaluating model...")
    test_loss = evaluate_model(model, test_dataloader, criterion, device)
    
    print("Testing completed!")
    print(f"Final testing loss: {test_loss:.6f}")

if __name__ == "__main__":
    main()
