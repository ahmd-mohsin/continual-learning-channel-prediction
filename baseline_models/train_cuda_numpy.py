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
import time

# Import your dataset and model
# from dataloader import ChannelSequenceDataset
# from effecientnet import LSTMChannelPredictor

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
        out = out.view(batch_size, channels, height, width)  # (batch, 2, 18, 8)
        
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
            print("-------------------------")
            with h5py.File(self.file_path, "r") as f:
                self.num_users = f["channel_matrix"].shape[0]
                self.time_length = f["channel_matrix"].shape[-1]
                channel_group = f["channel_matrix"]
                real = np.array(channel_group["real"])
                imag = np.array(channel_group["imag"])
                self.data = real + 1j * imag
            print("+++++++++++++++++++++++++++++")
        else:
            raise ValueError("Unsupported file format. Please use npy or mat.")
    
        self.overlapping_index = 16

    # def __len__(self):
    #     return self.num_users
    
    def __len__(self):
        # Each sample has (3000 - self.overlapping_index) valid indices to extract overlapping sequences
        return self.num_users * (self.time_length - (self.overlapping_index + 1))

    def __getitem__(self, idx):
        """
        Returns:
            input  -> (4, 18, 8, self.overlapping_index)  (self.overlapping_index time steps with real and imaginary parts concatenated)
            output -> (4, 18, 8)    (next time step with real and imaginary parts concatenated)
        """
        sample_idx = idx // (self.time_length - (self.overlapping_index + 1))  # Which of the 256 samples
        time_idx = idx % (self.time_length - (self.overlapping_index + 1))  # Time index within sample

        # Extract real and imaginary parts as NumPy arrays
        real_input = self.data.real[sample_idx, :, :, :, time_idx:time_idx+self.overlapping_index]
        imag_input = self.data.imag[sample_idx, :, :, :, time_idx:time_idx+self.overlapping_index]
        
        real_output = self.data.real[sample_idx, :, :, :, time_idx+self.overlapping_index]
        imag_output = self.data.imag[sample_idx, :, :, :, time_idx+self.overlapping_index]

        # Convert to PyTorch tensors
        real_input = torch.tensor(real_input, dtype=torch.float32, device=self.device)
        imag_input = torch.tensor(imag_input, dtype=torch.float32, device=self.device)
        real_output = torch.tensor(real_output, dtype=torch.float32, device=self.device)
        imag_output = torch.tensor(imag_output, dtype=torch.float32, device=self.device)

        # Concatenate along the first dimension
        input_data = torch.cat([real_input, imag_input], dim=0)  # Shape: (4, 18, 8, self.overlapping_index)
        output_data = torch.cat([real_output, imag_output], dim=0)  # Shape: (4, 18, self.overlapping_index)

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

def train_model(model, dataloader, device, num_epochs=10, learning_rate=1e-3, log_file="training_log.csv"):
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    best_loss = float('inf')
    start_time = time.time()  # Start the timer
    # Open CSV file for logging
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Average Loss"])  # Write header

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            num_batches = len(dataloader)

            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for batch_idx, (X_batch, Y_batch) in enumerate(progress_bar):
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device) 
                optimizer.zero_grad()
                
                predictions = model(X_batch) 
                loss = criterion(predictions, Y_batch)  
                loss.backward()  
                optimizer.step()  

                # Optional: Debug prints
                # print("X_Batch", X_batch.shape)
                # print("Y_batch", Y_batch.shape)
                # print("predictions", predictions.shape)

                loss.backward()
                optimizer.step()

                batch_loss = loss.item()
                total_loss += batch_loss
                progress_bar.set_postfix(batch_loss=batch_loss)

            avg_epoch_loss = total_loss / num_batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f}")
            writer.writerow([epoch + 1, avg_epoch_loss])

            scheduler.step(avg_epoch_loss)

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, 'best_channel_predictor.pth')
                print(f"Model saved with loss: {best_loss:.6f}")

        # Record total training time
        end_time = time.time()
        total_time = end_time - start_time
        formatted_time = time.strftime('%H:%M:%S', time.gmtime(total_time))

        print(f"\nTotal Training Time: {formatted_time} ({total_time:.2f} seconds)")

        # Write training time as final row in CSV (with blank epoch cell)
        writer.writerow([])
        writer.writerow(["Total Training Time", f"{formatted_time} ({total_time:.2f} seconds)"])

    return model
def evaluate_model(model, dataloader, device, log_file="evaluation_log.csv"):
    criterion = nn.MSELoss()
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
    
    file_path = "../dataset/outputs/umi_compact_conf_2tx_2rx."
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
    hidden_size = 32
    num_layers = 16
    output_size = 1  # Predicting one value per unit

    # Initialize model and move to device
    model = CustomLSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

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
