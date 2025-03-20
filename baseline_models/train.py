import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import h5py
import numpy as np
import os
from tqdm import tqdm

# Import your dataset and model
from dataloader import ChannelSequenceDataset
from effecientnet import LSTMChannelPredictor

def train_model(model, dataloader, device, num_epochs=10, learning_rate=1e-3):
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (X_batch, Y_batch) in enumerate(progress_bar):
            # X_batch shape: (batch, time_pairs, 4, 18, 8)
            # Y_batch shape: (batch, time_pairs, 4, 18, 8)
            
            batch_size, time_pairs = X_batch.shape[0], X_batch.shape[1]
            batch_loss = 0.0
            
            # Initialize hidden state
            hidden = None
            
            # Process each time step in sequence
            for t in range(time_pairs):
                X_t = X_batch[:, t].to(device)  # (batch, 4, 18, 8)
                Y_t = Y_batch[:, t].to(device)  # (batch, 4, 18, 8)
                
                # Forward pass
                optimizer.zero_grad()
                pred, hidden = model(X_t, hidden)
                
                # Detach hidden state from graph to prevent backprop through entire sequence
                hidden = (hidden[0].detach(), hidden[1].detach())
                
                # Calculate loss
                loss = criterion(pred, Y_t)
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                batch_loss += loss.item()
            
            # Average loss across time steps
            avg_batch_loss = batch_loss / time_pairs
            total_loss += avg_batch_loss
            
            # Update progress bar
            progress_bar.set_postfix({"Loss": f"{avg_batch_loss:.6f}"})
        
        # Calculate average loss for the epoch
        avg_epoch_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed - Average Loss: {avg_epoch_loss:.6f}")
        
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

def evaluate_model(model, dataloader, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for X_batch, Y_batch in tqdm(dataloader, desc="Evaluating"):
            batch_size, time_pairs = X_batch.shape[0], X_batch.shape[1]
            batch_loss = 0.0
            hidden = None
            
            for t in range(time_pairs):
                X_t = X_batch[:, t].to(device)
                Y_t = Y_batch[:, t].to(device)
                
                pred, hidden = model(X_t, hidden)
                loss = criterion(pred, Y_t)
                batch_loss += loss.item()
            
            avg_batch_loss = batch_loss / time_pairs
            total_loss += avg_batch_loss
    
    avg_loss = total_loss / len(dataloader)
    print(f"Evaluation - Average Loss: {avg_loss:.6f}")
    return avg_loss

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Load channel data from the .mat file
    filename = "/Users/muahmed/Desktop/Globecom 2025/nas-wireless/dataset/outputs/umi_compact_conf_8tx_2rx.mat"
    print(f"Loading data from {filename}...")
    
    with h5py.File(filename, "r") as f:
        channel_group = f["channel_matrix"]
        real_data = np.array(channel_group["real"])
        imag_data = np.array(channel_group["imag"])
        channel_matrix = real_data + 1j * imag_data
    
    print(f"Channel matrix shape: {channel_matrix.shape}")
    
    # Create dataset and dataloaders
    full_dataset = ChannelSequenceDataset(channel_matrix)
    
    # Split into training and validation sets (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    batch_size = 16
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Created dataloaders with batch size {batch_size}")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = LSTMChannelPredictor(
        in_channels=4,      # 2 real + 2 imaginary channels
        out_channels=4,     # Same format for output
        hidden_dim=256,     # LSTM hidden dimension
        num_layers=2        # Number of LSTM layers
    )
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    
    # Train the model
    print("Starting training...")
    model = train_model(
        model=model,
        dataloader=train_dataloader,
        device=device,
        num_epochs=10,
        learning_rate=1e-3
    )
    
    # Evaluate on validation set
    print("Evaluating model...")
    val_loss = evaluate_model(model, val_dataloader, device)
    
    print("Training completed!")
    print(f"Final validation loss: {val_loss:.6f}")

if __name__ == "__main__":
    main()