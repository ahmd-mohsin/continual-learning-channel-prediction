import time
import torch
import torch.optim as optim
from tqdm import tqdm
import csv
from loss import *

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


def train_model(model, dataloader, device, num_epochs=10, learning_rate=1e-3, log_file="training_log.csv", model_save_path="best_channel_predictor.pth"):
    criterion = CustomLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    best_loss = float('inf')
    start_time = time.time()

    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Average Loss"])

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
                # Optional: Debug prints
                # print("X_Batch", X_batch.shape)
                # print("Y_batch", Y_batch.shape)
                # print("predictions", predictions.shape)
                # Print min and max values for each
                # print("X_batch - Min: {}, Max: {}".format(X_batch.min().item(), X_batch.max().item()))
                # print("Y_batch - Min: {}, Max: {}".format(Y_batch.min().item(), Y_batch.max().item()))
                # print("Predictions - Min: {}, Max: {}".format(predictions.min().item(), predictions.max().item()))

                loss.backward()  
                optimizer.step()  

                batch_loss = loss.item()
                total_loss += batch_loss
                progress_bar.set_postfix(batch_loss=batch_loss)

            avg_epoch_loss = total_loss / num_batches
            writer.writerow([epoch + 1, avg_epoch_loss])

            scheduler.step(avg_epoch_loss)

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, model_save_path)

        end_time = time.time()
        total_time = end_time - start_time
        formatted_time = time.strftime('%H:%M:%S', time.gmtime(total_time))
        writer.writerow(["Total Training Time", f"{formatted_time} ({total_time:.2f} seconds)"])

    return model

def evaluate_model(model, dataloader, device, log_file="evaluation_log.csv"):
    criterion = CustomLoss()
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for batch_idx, (X_batch, Y_batch) in enumerate(tqdm(dataloader)):
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, Y_batch)
            total_loss += loss.item()

    avg_loss = total_loss / num_batches

    with open(log_file, mode='a', newline='') as file:
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
