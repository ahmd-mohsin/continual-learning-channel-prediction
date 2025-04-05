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


def train_model(model, dataloader, device, num_epochs=10, learning_rate=1e-3,
                log_file="training_log.csv", model_save_path="best_channel_predictor.pth"):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    best_loss = float('inf')
    total_start_time = time.time()

    # Overwrite the CSV each run. If you want to resume logging, change 'w' -> 'a'
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Average Loss", "Epoch Time (s)"])

        # ---------------------------
        # Training Loop
        # ---------------------------
        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            model.train()
            total_loss = 0.0
            num_batches = len(dataloader)

            # TQDM progress bar for this epoch
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, (X_batch, Y_batch) in enumerate(progress_bar):
                # if batch_idx > 4:
                #     break
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device) 
                optimizer.zero_grad()

                predictions = model(X_batch)
                # print("After normalized X_batch", X_batch[0], X_batch.shape)
                # print("After normalized Y_batch", Y_batch[0], Y_batch.shape)
                # print("Predictions", predictions[0], predictions.shape)
                loss = criterion(predictions, Y_batch)
                loss.backward()
                optimizer.step()

                batch_loss = loss.item()
                total_loss += batch_loss
                avg_loss_so_far = total_loss / (batch_idx + 1)
                progress_bar.set_postfix(avg_loss=avg_loss_so_far)

            # Average loss for this epoch
            avg_epoch_loss = total_loss / num_batches
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            writer.writerow([epoch + 1, avg_epoch_loss, f"{epoch_duration:.2f}"])
            print(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds with avg loss {avg_epoch_loss:.6f}")

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

        total_end_time = time.time()
        total_training_time = total_end_time - total_start_time
        formatted_total_time = time.strftime('%H:%M:%S', time.gmtime(total_training_time))
        writer.writerow(["Total Training Time", f"{formatted_total_time} ({total_training_time:.2f} seconds)"])

    return model


def evaluate_model(model, dataloader, device, log_file="evaluation_log.csv"):
    # criterion = CustomLoss()
    criterion = nn.MSELoss()
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
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Evaluation Loss", avg_loss])

    print(f"Evaluation Completed. Average Loss: {avg_loss:.4f}")
    return avg_loss
