import time
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import csv
from loss import *  # If you need CustomLoss from here

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


def train_model(model, 
                dataloader, 
                device, 
                num_epochs=10, 
                learning_rate=1e-3, 
                log_file="training_log.csv", 
                model_save_path="best_channel_predictor.pth"):
    """
    Trains the model, logs epoch loss & time to a CSV, and saves the best model.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (DataLoader): PyTorch dataloader for training data.
        device (torch.device): CUDA, MPS, or CPU.
        num_epochs (int): Number of epochs to train.
        learning_rate (float): Optimizer learning rate.
        log_file (str): CSV file path for training logs.
        model_save_path (str): Where to save the best model checkpoint.

    Returns:
        model (torch.nn.Module): Trained model (last epoch state).
    """

    # You can replace nn.MSELoss with CustomLoss() if desired
    criterion = nn.MSELoss()  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    best_loss = float('inf')

    # We'll track the cumulative time manually
    total_time_accum = 0.0  
    overall_start = time.time()

    # Overwrite the CSV each run. If you want to resume logging, change 'w' -> 'a'
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header with columns: Epoch, Average Loss, Epoch Time (s), Total Time (s)
        writer.writerow(["Epoch", "Average Loss", "Epoch Time (s)", "Total Time (s)"])

        # ---------------------------
        # Training Loop
        # ---------------------------
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            model.train()
            total_loss = 0.0
            num_batches = len(dataloader)

            # TQDM progress bar for this epoch
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, (X_batch, Y_batch) in enumerate(progress_bar):
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = criterion(predictions, Y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                avg_loss_so_far = total_loss / (batch_idx + 1)
                progress_bar.set_postfix(avg_loss=avg_loss_so_far)

            # Average loss for this epoch
            avg_epoch_loss = total_loss / num_batches

            # Epoch timing
            epoch_end = time.time()
            epoch_time = epoch_end - epoch_start
            total_time_accum += epoch_time

            # ReduceLROnPlateau expects a metric to step on (the epoch loss)
            scheduler.step(avg_epoch_loss)

            # Checkpoint if it's the best loss so far
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, model_save_path)

            # Write epoch info to CSV
            writer.writerow([
                epoch + 1,
                f"{avg_epoch_loss:.6f}",
                f"{epoch_time:.4f}",
                f"{total_time_accum:.4f}"
            ])

        # -------------------------------------
        # After all epochs: total training time
        # -------------------------------------
        overall_end = time.time()
        overall_time = overall_end - overall_start
        formatted_time = time.strftime('%H:%M:%S', time.gmtime(overall_time))
        writer.writerow([
            "Total Training Time", 
            f"{formatted_time} ({overall_time:.2f} seconds)"
        ])

    return model


def evaluate_model(model, dataloader, device, log_file="evaluation_log.csv"):
    """
    Evaluates the model on a validation/test set using CustomLoss (if desired),
    writes the average loss to CSV, and prints the result.

    Args:
        model (torch.nn.Module): Model to evaluate.
        dataloader (DataLoader): Validation/test dataloader.
        device (torch.device): CUDA, MPS, or CPU.
        log_file (str): CSV file path to append evaluation loss.

    Returns:
        float: Average loss on the dataloader.
    """
    # Example: If you want to use a custom loss from loss.py
    # otherwise use nn.MSELoss()
    criterion = CustomLoss()

    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for batch_idx, (X_batch, Y_batch) in enumerate(tqdm(dataloader, desc="Evaluating")):
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, Y_batch)
            total_loss += loss.item()

    avg_loss = total_loss / num_batches

    # Append evaluation loss to a CSV if desired
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Evaluation Loss", avg_loss])

    print(f"Evaluation Completed. Average Loss: {avg_loss:.4f}")
    return avg_loss
