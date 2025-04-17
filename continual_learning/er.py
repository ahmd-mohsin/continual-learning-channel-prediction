# Training the model sequentially on S1 -> S2 -> S3 using Experience Replay
from model import LSTMModel
from dataloader import ChannelSequenceDataset
import random
import torch.nn as nn
import torch
from utils import compute_device
from torch.utils.data import DataLoader, random_split, ConcatDataset, TensorDataset
from nmse import *
from dataloader import get_all_datasets
from tqdm import tqdm
snr_list = [0, 5, 10,12,14,16,18, 20,22,24,26,28, 30]
batch_size =  16
import csv


# Load datasets
data_dir = "../dataset/outputs/"  # Adjust this path to your dataset location
print("Loading datasets...")
train_S1, test_S1, train_S2, test_S2, train_S3, test_S3, train_loader_S1, test_loader_S1, train_loader_S2, test_loader_S2, train_loader_S3, test_loader_S3 = get_all_datasets(data_dir, dataset_id="all")
# get_all_datasets(data_dir)
print("Loaded datasets successfully.")
device = compute_device()


model_er = LSTMModel(input_dim=1,
            hidden_dim=32,
            output_dim=1,
            n_layers=3,
            H=16,
            W=36).to(device)
optimizer = torch.optim.Adam(model_er.parameters(), lr=1e-5)
criterion = nn.MSELoss()

num_epochs = 2
memory = []              # global memory buffer (list of (X, Y) tensors)
memory_capacity = 500    # max samples to store from each task (adjust as needed)


# Helper function to add samples to memory
def add_to_memory(dataset, num_samples):
    # Randomly select samples without replacement
    sample_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    for idx in sample_indices:
        X, Y = dataset[idx]
        # Store a copy of X, Y to avoid mutation (move to CPU for storage to free GPU if needed)
        memory.append((X.detach().cpu(), Y.detach().cpu()))
    # If global memory grows too large, truncate (FIFO or random removal)
    if len(memory) > memory_capacity * 2:  # example: if more than 2 tasks worth, trim oldest
        memory[:len(memory) - memory_capacity * 2] = []


print("Train on Task 1 (S1) normally")
# Train on Task 1 (S1) normally
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    running_loss = 0.0

    # Wrap the DataLoader with tqdm
    for X_batch, Y_batch in tqdm(train_loader_S1, desc="Training", leave=False):
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        optimizer.zero_grad()
        pred = model_er(X_batch)
        loss = criterion(pred, Y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader_S1)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Store some samples from S1 in replay memory
add_to_memory(train_S1, num_samples=memory_capacity)

# Train on Task 2 (S2) with replay of S1
# Create combined dataset of current task (S2) and memory (S1)
if memory:
    # Convert memory list of tuples into a TensorDataset for DataLoader
    X_mem = torch.stack([x for x, y in memory])
    Y_mem = torch.stack([y for x, y in memory])
    # Move memory tensors back to device for training
    X_mem = X_mem.to(device)
    Y_mem = Y_mem.to(device)
    memory_dataset = TensorDataset(X_mem, Y_mem)
    combined_train_S2 = ConcatDataset([train_S2, memory_dataset])
else:
    combined_train_S2 = train_S2  # no memory (should not happen here since memory has S1)
train_loader_combined_S2 = DataLoader(combined_train_S2, batch_size=batch_size, shuffle=True, drop_last=True)

print("Reinitialize optimizer for Task 2 training")
# Reinitialize optimizer for Task 2 training
optimizer = torch.optim.Adam(model_er.parameters(), lr=1e-5)
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    running_loss = 0.0

    # Wrap the DataLoader with tqdm for progress bar
    for X_batch, Y_batch in tqdm(train_loader_combined_S2, desc=f"Training", leave=False):
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        optimizer.zero_grad()
        pred = model_er(X_batch)
        loss = criterion(pred, Y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader_combined_S2)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
# Store samples from S2 in memory
add_to_memory(train_S2, num_samples=memory_capacity)

# Train on Task 3 (S3) with replay of S1 + S2
print("Train on Task 3 (S3) with replay of S1 + S2")
if memory:
    # Prepare combined dataset of S3 and all memory
    X_mem = torch.stack([x for x, y in memory])
    Y_mem = torch.stack([y for x, y in memory])
    X_mem = X_mem.to(device); Y_mem = Y_mem.to(device)
    memory_dataset = TensorDataset(X_mem, Y_mem)
    combined_train_S3 = ConcatDataset([train_S3, memory_dataset])
else:
    combined_train_S3 = train_S3
train_loader_combined_S3 = DataLoader(combined_train_S3, batch_size=batch_size, shuffle=True, drop_last=True)

optimizer = torch.optim.Adam(model_er.parameters(), lr=1e-5)
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    running_loss = 0.0

    # Wrap your DataLoader with tqdm
    for X_batch, Y_batch in tqdm(train_loader_combined_S3, desc="Training", leave=False):
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        optimizer.zero_grad()
        pred = model_er(X_batch)
        loss = criterion(pred, Y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader_combined_S3)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Evaluate final model on all tasks (NMSE vs SNR)
print("Experience Replay Method - NMSE on each task across SNRs:")
nmse_results_er = {}
nmse_results_er['S1'] = evaluate_nmse_vs_snr(model_er, test_loader_S1, device, snr_list)
nmse_results_er['S2'] = evaluate_nmse_vs_snr(model_er, test_loader_S2, device, snr_list)
nmse_results_er['S3'] = evaluate_nmse_vs_snr(model_er, test_loader_S3, device, snr_list)
# Print and collect rows for CSV
csv_rows = [ ['Task', 'SNR', 'NMSE'] ]
for task, nmse_vs_snr in nmse_results_er.items():
    for snr, nmse in nmse_vs_snr.items():
        print(f"Task {task}: SNR {snr}: NMSE {nmse:.4f}")
        csv_rows.append([task, snr, f"{nmse:.6f}"])

# Write to CSV
with open('er_nmse_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)

print("NMSE results saved to 'er_nmse_results.csv'")