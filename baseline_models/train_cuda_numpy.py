import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import h5py
import numpy as np
import os
from tqdm import tqdm
import argparse
from torch.utils.data import Dataset, DataLoader
import csv
from effecientnet import CustomLSTMModel
from dataloader import ChannelSequenceDataset

def train_model(model, dataloader, device, num_epochs=10, learning_rate=1e-3, log_file="training_log.csv"):
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    best_loss = float('inf')

    # Open CSV file for logging
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Average Loss"])  # Write header

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0  # Accumulate total loss for epoch
            num_batches = len(dataloader)  # Number of batches per epoch

            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for batch_idx, (X_batch, Y_batch) in enumerate(progress_bar):
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device) 
                optimizer.zero_grad()
                
                predictions = model(X_batch) 
                loss = criterion(predictions, Y_batch)  
                loss.backward()  
                optimizer.step()  

                batch_loss = loss.item()  # Convert loss tensor to scalar
                total_loss += batch_loss  # Accumulate loss
                
                progress_bar.set_postfix(batch_loss=batch_loss)  # Display batch loss in progress bar
            
            avg_epoch_loss = total_loss / num_batches  # Compute average loss for the epoch
            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f}")

            # Save loss data in CSV
            writer.writerow([epoch + 1, avg_epoch_loss])

            # Learning rate scheduling
            scheduler.step(avg_epoch_loss)
            
            # Save best model
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, 'best_channel_predictor.pth')
                print(f"Model saved with loss: {best_loss:.6f}")

    return model

def evaluate_model(model, dataloader, criterion, device, log_file="evaluation_log.csv"):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():  # Disable gradient computation for evaluation
        for X_batch, Y_batch in dataloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            predictions = model(X_batch)  # Forward pass
            loss = criterion(predictions, Y_batch)  # Compute loss

            total_loss += loss.item()  # Accumulate loss

    avg_loss = total_loss / num_batches  # Compute average loss

    # Save the loss to a CSV file
    with open(log_file, mode='a', newline='') as file:  # Append mode ('a')
        writer = csv.writer(file)
        writer.writerow(["Evaluation Loss", avg_loss])

    print(f"Evaluation Completed. Average Loss: {avg_loss:.4f}")
    
    return avg_loss

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

def main():
    parser = argparse.ArgumentParser(description="Train an LSTM model for channel prediction.")
    parser.add_argument("--ext", type=str, required=True, choices=["npy", "mat"], help="Dataset file extension (npy or mat)")
    args = parser.parse_args()
    
    torch.manual_seed(42)
    
    device = compute_device()
    # file_path = "../dataset/outputs/umi_compact_conf_8tx_2rx."
    
    file_path = "../dataset/outputs/umi_compact_conf_8tx_2rx.mat."
    full_dataset = ChannelSequenceDataset(file_path, args.ext, device)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    print(f"Created dataloaders with batch size {batch_size}")
    
    # model = LSTMChannelPredictor(in_channels=4, out_channels=4, hidden_dim=256, num_layers=2).to(device)
    input_size = 1  # Each time step has one feature
    hidden_size = 16
    num_layers = 16
    output_size = 1  # Predicting one value per unit

    # Initialize model and move to device
    model = CustomLSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    
    print("Starting training...")
    model = train_model(model=model, dataloader=train_dataloader, device=device, num_epochs=5, learning_rate=1e-3)
    
    print("Evaluating model...")
    val_loss = evaluate_model(model, val_dataloader, device)
    
    print("Training completed!")
    print(f"Final validation loss: {val_loss:.6f}")

if __name__ == "__main__":
    main()
