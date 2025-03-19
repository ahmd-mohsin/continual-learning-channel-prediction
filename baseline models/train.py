import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import h5py
import sys
sys.path.append("../")
from dataloader import ChannelDataset
from mobilenet import MobileNetChannelPredictor

def main():
    filename = "/Users/muahmed/Desktop/Globecom 2025/nas-wireless/dataset/outputs/umi_compact_conf_8tx_2rx.mat"
    with h5py.File(filename, "r") as f:
        channel_group = f["channel_matrix"]
        real_data = np.array(channel_group["real"])
        imag_data = np.array(channel_group["imag"])
        channel_matrix = real_data + 1j * imag_data

    dataset = ChannelDataset(channel_matrix)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

    in_channels = 4
    out_features = 2 * 2 * 18 * 8
    model = MobileNetChannelPredictor(
        in_channels=in_channels,
        out_features=out_features,
        pretrained=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        print(f"\nEpoch {epoch+1}/{num_epochs} ---------------------------")
        for batch_idx, (X_batch, Y_batch) in enumerate(loader):
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            pred = pred.view(X_batch.size(0), 4, 18, 8)

            pred_real = pred[:, 0:2, :, :]
            print(pred_real)
            pred_imag = pred[:, 2:4, :, :]
            print(pred_imag)
            target_real = Y_batch[:, 0:2, :, :]
            print(target_real)
            target_imag = Y_batch[:, 2:4, :, :]
            print(target_imag)
            loss_real = criterion(pred_real, target_real)
            loss_imag = criterion(pred_imag, target_imag)
            loss = loss_real + loss_imag

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
                print(f"  [Batch {batch_idx+1}/{len(loader)}]  Loss: {loss.item():.6f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Completed - Average Loss: {avg_loss:.6f}")

if __name__ == "__main__":
    main()
