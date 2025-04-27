import argparse
import os
import torch

# Import the new model classes from model.py:
from model import (
    GRUModel,
    LSTMModel,
    TransformerModel
)
# If you still want to use load_model for inference, you can also import it:
# from model import load_model
import csv
from utils import evaluate_model, compute_device
from dataloader import get_all_datasets
from tqdm import tqdm
from nmse import evaluate_nmse_vs_snr
from loss import NMSELoss
device = compute_device()
snr_list = [0, 5, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
batch_size = 2046

# Load datasets
print("Loading datasets...")
data_dir = "../dataset/outputs/"
train_S1, test_S1, train_S2, test_S2, train_S3, test_S3, \
    train_loader_S1, test_loader_S1, train_loader_S2, test_loader_S2, \
    train_loader_S3, test_loader_S3 = get_all_datasets(data_dir, batch_size=batch_size, dataset_id="all")
print("Loaded datasets successfully.")





def main():
    parser = argparse.ArgumentParser(description="Train or evaluate a channel-prediction model.")
    parser.add_argument("--ext", type=str, default="mat", choices=["npy", "mat"],
                        help="Dataset file extension (npy or mat)")
    parser.add_argument("--model_type", type=str, default="LSTM",
                        choices=["MLP", "CNN", "GRU", "LSTM", "TRANS"],
                        help="Which model architecture to use")
    parser.add_argument('--strategy', type=str, default='simple',
                    choices=['simple'],
                    help='Simple strategy')
    args = parser.parse_args()


    # Training the model sequentially on S1 -> S2 -> S3 using LwF (Knowledge Distillation)
    # Model instantiation
    if args.model_type == 'GRU':
        model_s1 = GRUModel(input_dim=1, hidden_dim=32, output_dim=1, n_layers=3, H=16, W=9).to(device)
    elif args.model_type == 'LSTM':
        model_s1 = LSTMModel(input_dim=1, hidden_dim=32, output_dim=1, n_layers=3, H=16, W=9).to(device)
    else:
        model_s1 = TransformerModel(dim_val=128, n_heads=4, n_encoder_layers=1,
                                n_decoder_layers=1, out_channels=2, H=16, W=9).to(device)

    optimizer = torch.optim.Adam(model_s1.parameters(), lr=1e-5)
    criterion = NMSELoss()

    num_epochs = 30
    print(f"--------------------{args.model_type}---------------------------")
    # Train on Task 1 (S1) normally (no old model yet)
    for epoch in range(num_epochs):
        running_loss = 0.0
        num_batches  = 0

        pbar = tqdm(train_loader_S1, desc=f"S1: Epoch {epoch+1}/{num_epochs}")
        # Use a simple tqdm bar for this epoch
        for X_batch, Y_batch in pbar:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            # print("=========")
            # print(X_batch.shape)
            # print(Y_batch.shape)
            optimizer.zero_grad()
            pred = model_s1(X_batch)
            task_loss = criterion(pred, Y_batch)
            task_loss.backward()
            optimizer.step()

            running_loss += task_loss.detach().item()
            num_batches  += 1


            # update the progress bar with the current batch loss
            pbar.set_postfix(loss=task_loss.detach().item())

        # Compute average loss for this epoch
        avg_loss = running_loss / num_batches
        print(f"Epoch [{epoch+1}/{num_epochs}]  Average Loss: {avg_loss}")


    if args.model_type == 'GRU':
        model_s2 = GRUModel(input_dim=1, hidden_dim=32, output_dim=1, n_layers=3, H=16, W=9).to(device)
    elif args.model_type == 'LSTM':
        model_s2 = LSTMModel(input_dim=1, hidden_dim=32, output_dim=1, n_layers=3, H=16, W=9).to(device)
    else:
        model_s2 = TransformerModel(dim_val=128, n_heads=4, n_encoder_layers=1,
                                n_decoder_layers=1, out_channels=2, H=16, W=9).to(device)
    
    
    # Train on Task 2 (S2) with LwF
    optimizer = torch.optim.Adam(model_s2.parameters(), lr=1e-5)
    criterion = NMSELoss()
    for epoch in range(num_epochs):
        running_loss = 0.0
        num_batches  = 0

        # Use tqdm with a simple description, we'll update postfix inside
        pbar = tqdm(train_loader_S2, desc=f"S2: Epoch {epoch+1}/{num_epochs}")
        for X_batch, Y_batch in pbar:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()
            pred      = model_s2(X_batch)
            task_loss = criterion(pred, Y_batch)
            task_loss.backward()
            optimizer.step()

            # accumulate for average
            running_loss += task_loss.detach().item()
            num_batches  += 1

            # update the progress bar with the current batch loss
            pbar.set_postfix(loss=task_loss.detach().item())

        # compute & print average loss for this epoch
        avg_loss = running_loss / num_batches
        print(f"Epoch [{epoch+1}/{num_epochs}]  Average Loss: {avg_loss}")

    if args.model_type == 'GRU':
        model_s3 = GRUModel(input_dim=1, hidden_dim=32, output_dim=1, n_layers=3, H=16, W=9).to(device)
    elif args.model_type == 'LSTM':
        model_s3 = LSTMModel(input_dim=1, hidden_dim=32, output_dim=1, n_layers=3, H=16, W=9).to(device)
    else:
        model_s3 = TransformerModel(dim_val=128, n_heads=4, n_encoder_layers=1,
                                n_decoder_layers=1, out_channels=2, H=16, W=9).to(device)
    
    

    # Train on Task 3 (S3) with LwF
    optimizer = torch.optim.Adam(model_s3.parameters(), lr=1e-5)
    criterion = NMSELoss()
    for epoch in range(num_epochs):
        running_loss = 0.0
        num_batches  = 0

        # Show a progress bar for this epoch
        pbar = tqdm(train_loader_S3, desc=f"S3: Epoch {epoch+1}/{num_epochs}")
        for X_batch, Y_batch in pbar:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()
            pred      = model_s3(X_batch)
            task_loss = criterion(pred, Y_batch)
            task_loss.backward()
            optimizer.step()

            # accumulate for average
            running_loss += task_loss.detach().item()
            num_batches  += 1

            # update bar with current batch loss
            pbar.set_postfix(loss=task_loss.detach().item())

        # end of epoch → compute & print average loss
        avg_loss = running_loss / num_batches
        print(f"Epoch [{epoch+1}/{num_epochs}]  Average Loss: {avg_loss}")

    # Evaluate final model on all tasks (NMSE vs SNR)
    # print("LwF Method - NMSE on each task across SNRs:")
    # nmse_results_lwf = {}
    # nmse_results_lwf['S1'] = evaluate_nmse_vs_snr(model_lwf, test_loader_S1, device, snr_list)
    # nmse_results_lwf['S2'] = evaluate_nmse_vs_snr(model_lwf, test_loader_S2, device, snr_list)
    # nmse_results_lwf['S3'] = evaluate_nmse_vs_snr(model_lwf, test_loader_S3, device, snr_list)
    # for task, nmse_vs_snr in nmse_results_lwf.items():
    #     print(f"Task {task}: " + ", ".join([f"SNR {snr}: NMSE {nmse:.4f}" 
    #                                         for snr, nmse in nmse_vs_snr.items()]))





    os.makedirs(args.strategy, exist_ok=True)
    # Evaluation

    nmse_results_ewc = {
        'model_Compact_test_compact': evaluate_nmse_vs_snr(model_s1, test_loader_S1, device, snr_list),
        'model_Compact_test_Dense': evaluate_nmse_vs_snr(model_s1, test_loader_S2, device, snr_list),
        'model_Compact_test_Standard': evaluate_nmse_vs_snr(model_s1, test_loader_S3, device, snr_list),
        'model_Dense_test_compact': evaluate_nmse_vs_snr(model_s2, test_loader_S1, device, snr_list),
        'model_Dense_test_Dense': evaluate_nmse_vs_snr(model_s2, test_loader_S2, device, snr_list),
        'model_Dense_test_Standard': evaluate_nmse_vs_snr(model_s2, test_loader_S3, device, snr_list),
        'model_Standard_test_compact': evaluate_nmse_vs_snr(model_s3, test_loader_S1, device, snr_list),
        'model_Standard_test_Dense': evaluate_nmse_vs_snr(model_s3, test_loader_S2, device, snr_list),
        'model_Standard_test_Standard': evaluate_nmse_vs_snr(model_s3, test_loader_S3, device, snr_list)
    }


    print("=== Evaluation ===")
    test_loss_csv = f"{args.strategy}/{args.strategy}_{args.model_type}_loss.csv"
    results = {
        'model_Compact_test_compact': evaluate_model(model_s1, test_loader_S1, device),
        'model_Compact_test_Dense': evaluate_model(model_s1, test_loader_S2, device),
        'model_Compact_test_Standard': evaluate_model(model_s1, test_loader_S3, device),
        'model_Dense_test_compact': evaluate_model(model_s2, test_loader_S1, device),
        'model_Dense_test_Dense': evaluate_model(model_s2, test_loader_S2, device),
        'model_Dense_test_Standard': evaluate_model(model_s2, test_loader_S3, device),
        'model_Standard_test_compact': evaluate_model(model_s3, test_loader_S1, device),
        'model_Standard_test_Dense': evaluate_model(model_s3, test_loader_S2, device),
        'model_Standard_test_Standard': evaluate_model(model_s3, test_loader_S3, device)
    }
    with open(test_loss_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Task', 'Loss'])
        for t, l in results.items():
            print(f"{t} Loss: {l}")
            writer.writerow([t, l])
    print(f"Saved losses to {test_loss_csv}")
    # nmse_results_ewc = {}
    # nmse_results_ewc['S1_Compact'] = evaluate_nmse_vs_snr(model_lwf, test_loader_S1, device, snr_list)
    # nmse_results_ewc['S2_Dense'] = evaluate_nmse_vs_snr(model_lwf, test_loader_S2, device, snr_list)
    # nmse_results_ewc['S3_Standard'] = evaluate_nmse_vs_snr(model_lwf, test_loader_S3, device, snr_list)

    csv_rows = [['Task', 'SNR', 'NMSE']]
    for task, res in nmse_results_ewc.items():
        for snr, nmse in res.items():
            print(f"Task {task} | SNR {snr:2d} → NMSE {nmse}")
            csv_rows.append([task, snr, f"{nmse}"])

    csv_path = f"{args.strategy}/{args.strategy}_{args.model_type}_nmse_results.csv"
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerows(csv_rows)

    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
