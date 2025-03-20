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


# Import your dataset and model
# from dataloader import ChannelSequenceDataset
from effecientnet import LSTMChannelPredictor

class CustomLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CustomLSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        batch_size, channels, height, width, time_steps = x.shape  # (batch, 2, 18, 8, 3000)
        
        # Move input tensor to device
        x = x
        
        # Reshape to feed into LSTM
        x = x.view(batch_size * channels * height * width, time_steps, -1)  # (batch * 2 * 18 * 8, 3000, feature_dim)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Get the last time step output
        lstm_out = lstm_out[:, -1, :]  # (batch * 2 * 18 * 8, hidden_size)
        
        # Fully connected layer to match output shape
        out = self.fc(lstm_out)  # (batch * 2 * 18 * 8, output_size)
        
        # Reshape to match desired output
        out = out.view(batch_size, channels, width, height)  # (batch, 2, 8, 18)
        
        return out


class ChannelSequenceDataset(Dataset):
    def __init__(self, file_path, file_extension, device):
        self.file_path = file_path + file_extension
        self.file_extension = file_extension
        self.device = device
        
        if self.file_extension == "npy":
            self.data = np.load(self.file_path, mmap_mode='r')
            self.num_users = self.data.shape[0]
            self.time_length = self.data.shape[-1]
        elif self.file_extension == "mat":
            with h5py.File(self.file_path, "r") as f:
                self.num_users = f["channel_matrix"].shape[0]
                self.time_length = f["channel_matrix"].shape[-1]
        else:
            raise ValueError("Unsupported file format. Please use npy or mat.")
    
    # def __len__(self):
    #     return self.num_users
    
    def __len__(self):
        # Each sample has (3000 - 9) valid indices to extract overlapping sequences
        return self.num_samples * (self.num_timestamps - 9)

    def __getitem__(self, idx):
        """
        Returns:
            input  -> (2, 18, 8, 8)  (8 time steps)
            output -> (2, 18, 8)    (1 next time step)
        """
        sample_idx = idx // (self.num_timestamps - 9)  # Extract sample index (which of the 256 samples)
        time_idx = idx % (self.num_timestamps - 9)     # Extract time index within the sample

        input_data = self.data[sample_idx, :, :, :, time_idx:time_idx+8]  # Shape: (2, 18, 8, 8)
        output_data = self.data[sample_idx, :, :, :, time_idx+8]  # Shape: (2, 18, 8)

        return input_data, output_data
    # def __getitem__(self, idx):
        
    #     if self.file_extension == "npy":
    #         data = self.data[idx]
    #         real_data = torch.tensor(data.real, device=self.device)
    #         imag_data = torch.tensor(data.imag, device=self.device)
    #     else:
    #         with h5py.File(self.file_path, "r") as f:
    #             real_data = torch.tensor(np.array(f["channel_matrix"]["real"])[idx], device=self.device)
    #             imag_data = torch.tensor(np.array(f["channel_matrix"]["imag"])[idx], device=self.device)
        
    #     time_pairs = self.time_length - 1
    #     X = torch.zeros(time_pairs, 4, 18, 8, device=self.device)
    #     Y = torch.zeros(time_pairs, 4, 18, 8, device=self.device)
        
    #     for t in range(time_pairs):
    #         X[t] = torch.cat([real_data[:, :, :, t], imag_data[:, :, :, t]], dim=0)
    #         Y[t] = torch.cat([real_data[:, :, :, t+1], imag_data[:, :, :, t+1]], dim=0)
        
    #     return X, Y

# def load_data(file_extension, device):
#     default_path = "/Users/muahmed/Desktop/Globecom 2025/nas-wireless/dataset/outputs/umi_compact_conf_8tx_2rx."
#     file_path = default_path + file_extension
    
#     if file_extension == "npy":
#         default_path = "../dataset/outputs/umi_compact_conf_8tx_2rx.mat."
#         file_path = default_path + file_extension
#         print(f"Loading data from {file_path} (NumPy format)...")
#         channel_matrix = np.load(file_path)
#         print(f"Loaded {file_path}")
#     elif file_extension == "mat":
#         print(f"Loading data from {file_path} (MATLAB format)...")
#         with h5py.File(file_path, "r") as f:
#             channel_group = f["channel_matrix"]
#             real_data = np.array(channel_group["real"])
#             imag_data = np.array(channel_group["imag"])
#             channel_matrix = real_data + 1j * imag_data
#     else:
#         raise ValueError("Unsupported file format. Please provide a valid extension: npy or mat.")
#     print(f"Channel matrix shape: {channel_matrix.shape}")
#     channel_matrix = torch.tensor(channel_matrix, dtype=torch.cdouble, device=device)  # Move to CUDA immediately
#     return channel_matrix

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
                print("pred: ",pred.shape)
                print("Y_t:  ",Y_t.shape)
                print("X_t:  ",X_t.shape)
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
    
    batch_size = 4
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    print(f"Created dataloaders with batch size {batch_size}")
    
    model = LSTMChannelPredictor(in_channels=4, out_channels=4, hidden_dim=256, num_layers=2).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    
    print("Starting training...")
    model = train_model(model=model, dataloader=train_dataloader, device=device, num_epochs=10, learning_rate=1e-3)
    
    print("Evaluating model...")
    val_loss = evaluate_model(model, val_dataloader, device)
    
    print("Training completed!")
    print(f"Final validation loss: {val_loss:.6f}")

if __name__ == "__main__":
    main()
