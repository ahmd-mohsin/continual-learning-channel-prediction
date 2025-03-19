import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import h5py
import numpy as np

# Import your dataset and model from their respective modules
from dataloader import ChannelSequenceDataset
from effecientnet import EfficientLSTMChannelPredictor

def main():
    # Load channel data from the .mat file.
    filename = "/Users/muahmed/Desktop/Globecom 2025/nas-wireless/dataset/outputs/umi_compact_conf_8tx_2rx.mat"
    with h5py.File(filename, "r") as f:
        channel_group = f["channel_matrix"]
        real_data = np.array(channel_group["real"])
        imag_data = np.array(channel_group["imag"])
        channel_matrix = real_data + 1j * imag_data

    # Create the dataset.
    # Each sample corresponds to one user with an input sequence of shape (T-1, 4, 18, 8)
    # and target of shape (4, 18, 8), where T=3000.
    dataset = ChannelSequenceDataset(channel_matrix)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

    # Create the model.
    # The model expects input of shape (batch, time_steps, 4, 18, 8)
    # and outputs a prediction of shape (batch, 4, 18, 8).
    model = EfficientLSTMChannelPredictor(
        in_channels=4,
        out_channels=4,  # Output: first 2 channels for real, next 2 for imaginary
        hidden_dim=512,
        num_layers=2,
        pretrained=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss function and optimizer.
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        print(f"\nEpoch {epoch+1}/{num_epochs} ---------------------------")
        for batch_idx, (X_batch, Y_batch) in enumerate(dataloader):
            # Move the batch to the device.
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            
            optimizer.zero_grad()
            # Forward pass: model predicts output of shape (batch, 4, 18, 8)
            pred = model(X_batch)
            
            # Split predictions into real and imaginary parts.
            pred_real = pred[:, :2, :, :]  # (batch, 2, 18, 8)
            pred_imag = pred[:, 2:4, :, :]  # (batch, 2, 18, 8)
            target_real = Y_batch[:, :2, :, :]
            target_imag = Y_batch[:, 2:4, :, :]
            
            # Compute the loss separately and then sum them.
            loss_real = criterion(pred_real, target_real)
            loss_imag = criterion(pred_imag, target_imag)
            loss = loss_real + loss_imag
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}: Loss = {loss.item():.6f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Completed - Average Loss: {avg_loss:.6f}")

if __name__ == "__main__":
    main()
