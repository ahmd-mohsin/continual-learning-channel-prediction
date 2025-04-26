import time
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import csv
from loss import *  # If you need CustomLoss from here
from loss import NMSELoss
def compute_device():
    """
    Determines the best available computing device (CUDA, MPS, or CPU).
    """
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


def evaluate_model(model, dataloader, device, log_file="evaluation_log.csv"):
    # criterion = CustomLoss()
    criterion = NMSELoss()
    # model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for batch_idx, (X_batch, Y_batch) in enumerate(tqdm(dataloader, desc="Evaluating")):
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            # Skip if X_batch is all zeros
            # Skip batch if X_batch or Y_batch sum to zero
            # if torch.sum(X_batch) < 0.1:
            #     print(f"Skipping batch {batch_idx} because X_batch sum is less than equal to zero.")
            #     continue
            # if torch.sum(Y_batch) < 0.1:
            #     print(f"Skipping batch {batch_idx} because Y_batch sum is less than equal to zero.")
            #     continue


            predictions = model(X_batch)
            loss = criterion(predictions, Y_batch)
            
            # print("After normalized X_batch", X_batch[0], X_batch.shape)
            # print("After normalized Y_batch", Y_batch[0], Y_batch.shape)
            # print("Predictions", predictions[0], predictions.shape)
            # print("Loss", loss.item())
            total_loss += loss.item()

    avg_loss = total_loss / num_batches

    # Append evaluation loss to a CSV if desired
    # with open(log_file, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Evaluation Loss", avg_loss])

    print(f"Evaluation Completed. Average Loss: {avg_loss:.4f}")
    return avg_loss
