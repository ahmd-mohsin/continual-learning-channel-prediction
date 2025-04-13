
# nas_utils.py
import torch
# import torch.nn as nn
from torch.utils.data import DataLoader
import nni
from dataloader import ChannelSequenceDataset
from tqdm import tqdm

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
    # Wrap the training loader with tqdm and use enumerate to get batch index
    for idx, (x, y) in enumerate(tqdm(loader, desc='Training', leave=False)):
        # if idx > 2:
        #     break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        # Optionally, log or print the index and loss:
        # print(f"Batch {idx}: loss = {loss.item()}")

def test_epoch(model, loader, loss_fn, device):
    model.eval()
    total, count = 0.0, 0
    # Wrap the testing loader with tqdm and use enumerate to get batch index
    with torch.no_grad():
        for idx, (x, y) in enumerate(tqdm(loader, desc='Testing', leave=False)):
            # if idx > 2:
            #     break
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total += loss_fn(pred, y).item() * x.size(0)
            count += x.size(0)
            # Optionally, log the batch index:
            # print(f"Testing batch {idx} processed.")
    return total / count

def evaluate_model(model):
    # Assuming train_dataset and test_dataset are defined elsewhere in your project
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(test_dataset, batch_size=64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn   = torch.nn.MSELoss()

    # Wrap the epoch loop with tqdm to display a progress bar with the description "Epochs"
    for epoch in tqdm(range(5), desc='Epochs'):
        train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = test_epoch(model, valid_loader, loss_fn, device)
        nni.report_intermediate_result(val_loss)
    nni.report_final_result(val_loss)


    # import json
    # import time
    # time.sleep(2)
    # val_loss = float(0.12345)
    # nni.report_intermediate_result(val_loss)

    # time.sleep(2)
    # nni.report_final_result(val_loss)
    # return val_loss

    # import time
    # try:
    #     # Your evaluation logic here
    #     val_loss = 0.12345
        
    #     # Report intermediate result
    #     nni.report_intermediate_result(float(val_loss))
    #     time.sleep(1)  # Give time for communication
        
    #     # Report final result
    #     nni.report_final_result(float(val_loss))
    #     time.sleep(1)  # Give time for communication
        
    #     return float(val_loss)
    # except Exception as e:
    #     print(f"Error in evaluation: {e}")
    #     # Report a default value on error
    #     nni.report_final_result(1.0)
    #     return 1.0