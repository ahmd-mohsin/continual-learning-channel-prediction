
# nas_utils.py
import torch
# import torch.nn as nn
from torch.utils.data import DataLoader
import nni
from dataloader import ChannelSequenceDataset

# from model import *
def compute_device():
    """
    Determines the best available computing device (CUDA, MPS, or CPU).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # print("Using CUDA for computation")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        # print("Using MPS (Metal Performance Shaders) for computation")
    else:
        device = torch.device("cpu")
        # print("Using CPU for computation")
    return device


device  = compute_device()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("Using device:", device)
full_dataset = ChannelSequenceDataset("/home/ahsan/Ahsan/PhD_work/nas-wireless/dataset/outputs/umi_compact_conf_2tx_2rx.", "mat", device)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
    
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
# print("train_dataset size:", len(train_dataset))
# print("test_dataset size:", len(test_dataset))

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    for x, y in loader:
        # print("test_epoch x shape:", x.shape)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

def test_epoch(model, loader, loss_fn, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            # print("test_epoch x shape:", x.shape)
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total += loss_fn(pred, y).item() * x.size(0)
            count += x.size(0)
    return total / count

def evaluate_model(model):
    # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(test_dataset, batch_size=16)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn   = torch.nn.MSELoss()

    # train & report intermediate results
    for epoch in range(5):
        train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = test_epoch(model, valid_loader, loss_fn, device)
        nni.report_intermediate_result(val_loss)

    # final result
    nni.report_final_result(val_loss)
