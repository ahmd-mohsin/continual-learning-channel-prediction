import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import h5py
from dataloader import ChannelDataset
from mobilenet import MobileNetChannelPredictor

def main():
    filename = "../dataset/outputs/umi_compact_conf_8tx_2rx.mat"
    with h5py.File(filename, "r") as f:
        channel_group = f["channel_matrix"]
        real_data = np.array(channel_group["real"])
        imag_data = np.array(channel_group["imag"])
        channel_matrix = real_data + 1j * imag_data

    dataset = ChannelDataset(channel_matrix)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
    in_channels = 4
    out_features = 2 * 2 * 18 * 8
    model = MobileNetChannelPredictor(in_channels=in_channels, out_features=out_features, pretrained=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 10
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            pred = pred.view(X_batch.size(0), 4, 18, 8)
            pred_real = pred[:, 0:2, :, :]
            pred_imag = pred[:, 2:4, :, :]
            target_real = Y_batch[:, 0:2, :, :]
            target_imag = Y_batch[:, 2:4, :, :]
            loss_real = criterion(pred_real, target_real)
            loss_imag = criterion(pred_imag, target_imag)
            loss = loss_real + loss_imag
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}")

if __name__ == "__main__":
    main()
