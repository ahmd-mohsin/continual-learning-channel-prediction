# import argparse
# import os
# import torch
# from model import LSTMChannelPredictor    
# import csv
# from utils import evaluate_model, compute_device
# from dataloader import get_all_datasets
# from tqdm import tqdm
# from nmse import evaluate_nmse_vs_snr
# from loss import masked_nmse    
# device = compute_device()
# snr_list = [0, 5, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
# batch_size = 2048
# seq_len    = 64

# train_S1, test_S1, train_loader_S1, test_loader_S1, \
# train_S2, test_S2, train_loader_S2, test_loader_S2, \
# train_S3, test_S3, train_loader_S3, test_loader_S3 = \
#     get_all_datasets(
#         data_dir  = "../dataset/outputs/",
#         batch_size= batch_size,
#         dataset_id= "all",
#         normalization = "log_min_max",
#         per_user      = True,
#         seq_len       = seq_len          
#     )    
# print("Loaded datasets successfully.")


# def validate_model(model, train , test_loader, device):
#     orig_ds = train.dataset  

#     mag_mean = torch.tensor(orig_ds.mag_mean, device=device, dtype=torch.float32)
#     mag_std  = torch.tensor(orig_ds.mag_std,  device=device, dtype=torch.float32)

#     criterion = masked_nmse()
#     # now run your test
#     with torch.no_grad():
#         val_loss = 0.0
#         n_batches = 0

#         for X_norm, Y_norm in test_loader:
#             X_norm = X_norm.to(device)
#             Y_norm = Y_norm.to(device)
#             pred_norm = model(X_norm)                 
#             pred = pred_norm * mag_std + mag_mean         
#             Y_true = Y_norm  * mag_std + mag_mean
#             nmse = criterion(pred , Y_true).item()
#             val_loss += nmse
#             n_batches += 1
#             val_loss /= len(test_loader_S3)
#         return val_loss

# def main():
#     parser = argparse.ArgumentParser(description="Train or evaluate a channel-prediction model.")
#     parser.add_argument("--ext", type=str, default="mat", choices=["npy", "mat"],
#                         help="Dataset file extension (npy or mat)")
#     parser.add_argument("--model_type", type=str, default="LSTM",
#                         choices=["MLP", "CNN", "GRU", "LSTM", "TRANS"],
#                         help="Which model architecture to use")
#     parser.add_argument('--strategy', type=str, default='simple',
#                     choices=['simple'],
#                     help='Simple strategy')
#     args = parser.parse_args()

#     model_s1 = LSTMChannelPredictor().to(device)
#     alpha     = 0.5

#     optimizer = torch.optim.Adam(model_s1.parameters(), lr=1e-3)
#     criterion = masked_nmse()
#     bce_loss  = torch.nn.BCEWithLogitsLoss()

#     num_epochs = 100
#     print(f"--------------------{args.model_type}---------------------------")
#     # Train on Task 1 (S1) normally (no old model yet)
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         num_batches  = 0

#         pbar = tqdm(train_loader_S1, desc=f"S1: Epoch {epoch+1}/{num_epochs}")
#         # Use a simple tqdm bar for this epoch
#         for X_batch, Y_batch in pbar:
#             # X_batch : (B, 2, R, S, T, L)   ;  Y_batch : (B, 2, R, S, T)
#             X_batch = X_batch.to(device)
#             Y_batch = Y_batch.to(device)

#             mag_true  = Y_batch[:, 0]     # channel-0 magnitude
#             mask_true = Y_batch[:, 1]     # channel-1 binary mask

#             optimizer.zero_grad()
#             mag_pred, mask_logits = model_s1(X_batch)     # forward pass

#             loss_mag  = masked_nmse(mag_pred, mag_true, mask_true)
#             loss_mask = bce_loss(mask_logits, mask_true)
#             loss      = loss_mag + alpha * loss_mask

#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#             num_batches  += 1
#             pbar.set_postfix(loss=loss_mag.item(), mask_loss=loss_mask.item())

#             # update the progress bar with the current batch loss
#             pbar.set_postfix(loss=task_loss.detach().item())

#         val_loss = validate_model(model_s1, train_S1, test_loader_S1, device)
#         # Compute average loss for this epoch
#         avg_loss = running_loss / num_batches
#         print(f"Epoch [{epoch+1}/{num_epochs}]  Average Loss: {avg_loss}")
#         print(f"Validation Loss: {val_loss}")

#     if args.model_type == 'GRU':
#         model_s2 = GRUModel(input_dim=1, hidden_dim=32, output_dim=1, n_layers=3, H=16, W=9).to(device)
#     elif args.model_type == 'LSTM':
#         model_s2 = LSTMModel(input_dim=1, hidden_dim=32, output_dim=1, n_layers=3, H=16, W=9).to(device)
#     else:
#         model_s2 = TransformerModel(dim_val=128, n_heads=4, n_encoder_layers=1,
#                                 n_decoder_layers=1, out_channels=2, H=16, W=9).to(device)
    
    
#     # Train on Task 2 (S2) with LwF
#     optimizer = torch.optim.Adam(model_s2.parameters(), lr=1e-3)
#     criterion = NMSELoss()
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         num_batches  = 0

#         # Use tqdm with a simple description, we'll update postfix inside
#         pbar = tqdm(train_loader_S2, desc=f"S2: Epoch {epoch+1}/{num_epochs}")
#         for X_batch, Y_batch in pbar:
#             X_batch = X_batch.to(device)
#             Y_batch = Y_batch.to(device)

#             optimizer.zero_grad()
#             pred      = model_s2(X_batch)
#             task_loss = criterion(pred, Y_batch)
#             task_loss.backward()
#             optimizer.step()

#             # accumulate for average
#             running_loss += task_loss.detach().item()
#             num_batches  += 1

#             # update the progress bar with the current batch loss
#             pbar.set_postfix(loss=task_loss.detach().item())

#         # compute & print average loss for this epoch
#         avg_loss = running_loss / num_batches
#         print(f"Epoch [{epoch+1}/{num_epochs}]  Average Loss: {avg_loss}")

#     if args.model_type == 'GRU':
#         model_s3 = GRUModel(input_dim=1, hidden_dim=32, output_dim=1, n_layers=3, H=16, W=9).to(device)
#     elif args.model_type == 'LSTM':
#         model_s3 = LSTMModel(input_dim=1, hidden_dim=32, output_dim=1, n_layers=3, H=16, W=9).to(device)
#     else:
#         model_s3 = TransformerModel(dim_val=128, n_heads=4, n_encoder_layers=1,
#                                 n_decoder_layers=1, out_channels=2, H=16, W=9).to(device)
    
    

#     # Train on Task 3 (S3) with LwF
#     optimizer = torch.optim.Adam(model_s3.parameters(), lr=1e-3)
#     criterion = NMSELoss()
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         num_batches  = 0

#         # Show a progress bar for this epoch
#         pbar = tqdm(train_loader_S3, desc=f"S3: Epoch {epoch+1}/{num_epochs}")
#         for X_batch, Y_batch in pbar:
#             X_batch = X_batch.to(device)
#             Y_batch = Y_batch.to(device)

#             optimizer.zero_grad()
#             pred      = model_s3(X_batch)
#             task_loss = criterion(pred, Y_batch)
#             task_loss.backward()
#             optimizer.step()

#             # accumulate for average
#             running_loss += task_loss.detach().item()
#             num_batches  += 1

#             # update bar with current batch loss
#             pbar.set_postfix(loss=task_loss.detach().item())

#         # end of epoch → compute & print average loss
#         avg_loss = running_loss / num_batches
#         print(f"Epoch [{epoch+1}/{num_epochs}]  Average Loss: {avg_loss}")





#     os.makedirs(args.strategy, exist_ok=True)
#     # Evaluation

#     nmse_results_ewc = {
#         'model_Compact_test_compact': evaluate_nmse_vs_snr(model_s1, test_loader_S1, device, snr_list),
#         'model_Compact_test_Dense': evaluate_nmse_vs_snr(model_s1, test_loader_S2, device, snr_list),
#         'model_Compact_test_Standard': evaluate_nmse_vs_snr(model_s1, test_loader_S3, device, snr_list),
#         'model_Dense_test_compact': evaluate_nmse_vs_snr(model_s2, test_loader_S1, device, snr_list),
#         'model_Dense_test_Dense': evaluate_nmse_vs_snr(model_s2, test_loader_S2, device, snr_list),
#         'model_Dense_test_Standard': evaluate_nmse_vs_snr(model_s2, test_loader_S3, device, snr_list),
#         'model_Standard_test_compact': evaluate_nmse_vs_snr(model_s3, test_loader_S1, device, snr_list),
#         'model_Standard_test_Dense': evaluate_nmse_vs_snr(model_s3, test_loader_S2, device, snr_list),
#         'model_Standard_test_Standard': evaluate_nmse_vs_snr(model_s3, test_loader_S3, device, snr_list)
#     }


#     print("=== Evaluation ===")
#     test_loss_csv = f"{args.strategy}/{args.strategy}_{args.model_type}_loss.csv"
#     results = {
#         'model_Compact_test_compact': evaluate_model(model_s1, test_loader_S1, device),
#         'model_Compact_test_Dense': evaluate_model(model_s1, test_loader_S2, device),
#         'model_Compact_test_Standard': evaluate_model(model_s1, test_loader_S3, device),
#         'model_Dense_test_compact': evaluate_model(model_s2, test_loader_S1, device),
#         'model_Dense_test_Dense': evaluate_model(model_s2, test_loader_S2, device),
#         'model_Dense_test_Standard': evaluate_model(model_s2, test_loader_S3, device),
#         'model_Standard_test_compact': evaluate_model(model_s3, test_loader_S1, device),
#         'model_Standard_test_Dense': evaluate_model(model_s3, test_loader_S2, device),
#         'model_Standard_test_Standard': evaluate_model(model_s3, test_loader_S3, device)
#     }
#     with open(test_loss_csv, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['Task', 'Loss'])
#         for t, l in results.items():
#             print(f"{t} Loss: {l}")
#             writer.writerow([t, l])
#     print(f"Saved losses to {test_loss_csv}")
#     # nmse_results_ewc = {}
#     # nmse_results_ewc['S1_Compact'] = evaluate_nmse_vs_snr(model_lwf, test_loader_S1, device, snr_list)
#     # nmse_results_ewc['S2_Dense'] = evaluate_nmse_vs_snr(model_lwf, test_loader_S2, device, snr_list)
#     # nmse_results_ewc['S3_Standard'] = evaluate_nmse_vs_snr(model_lwf, test_loader_S3, device, snr_list)

#     csv_rows = [['Task', 'SNR', 'NMSE']]
#     for task, res in nmse_results_ewc.items():
#         for snr, nmse in res.items():
#             print(f"Task {task} | SNR {snr:2d} → NMSE {nmse}")
#             csv_rows.append([task, snr, f"{nmse}"])

#     csv_path = f"{args.strategy}/{args.strategy}_{args.model_type}_nmse_results.csv"
#     with open(csv_path, 'w', newline='') as f:
#         csv.writer(f).writerows(csv_rows)

#     print(f"\nResults saved to {csv_path}")


# if __name__ == "__main__":
#     main()




# main.py  ─ single-dataset training, 3-dataset evaluation
# --------------------------------------------------------
import os, csv, argparse, torch
from tqdm import tqdm

from dataloader import get_all_datasets
from model      import LSTMChannelPredictor          # 2-channel LSTM (mag + mask)
from loss       import masked_nmse                   # masked NMSE function
from utils      import compute_device, evaluate_nmse_vs_snr_masked

# --------------- hyper-parameters -----------------------
BATCH_SIZE = 2048
SEQ_LEN    = 16
NUM_EPOCHS = 100
ALPHA      = 0.5         # weight for BCE mask loss
LR         = 1e-5
SNR_LIST   = [0,5,10,12,14,16,18,20,22,24,26,28,30]

# --------------- data loading ---------------------------
device = compute_device()
print("Loading datasets ...")
train_S1, test_S1, train_loader_S1, test_loader_S1, \
train_S2, test_S2, train_loader_S2, test_loader_S2, \
train_S3, test_S3, train_loader_S3, test_loader_S3 = \
    get_all_datasets(
        data_dir      = "../dataset/outputs/",
        batch_size    = BATCH_SIZE,
        dataset_id    = "all",
        normalization = "log_min_max",
        per_user      = True,
        seq_len       = SEQ_LEN
    )
print("Datasets ready.")

# --------------- model / loss / optim -------------------
model = LSTMChannelPredictor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
bce_loss  = torch.nn.BCEWithLogitsLoss()

# --------------- training loop (S1 only) ----------------
for epoch in range(NUM_EPOCHS):
    model.train()
    running, n = 0.0, 0
    pbar = tqdm(train_loader_S1, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for X, Y in pbar:
        X, Y = X.to(device), Y.to(device)
        mag_t, mask_t = Y[:,0], Y[:,1]

        optimizer.zero_grad()
        mag_p, mask_logits = model(X)

        loss_mag  = masked_nmse(mag_p, mag_t, mask_t)
        loss_mask = bce_loss(mask_logits, mask_t)
        loss      = loss_mag + ALPHA * loss_mask
        loss.backward()
        optimizer.step()

        running += loss.item(); n += 1
        pbar.set_postfix(mag=loss_mag.item(), bce=loss_mask.item())

    print(f"  ↳ avg train-loss: {running/n:.5f}")

# --------------- quick validation on S1 -----------------
model.eval()
val_loss = evaluate_nmse_vs_snr_masked(model,
                                       test_loader_S1,
                                       device,
                                       [0])[0] 
print(f"Validation masked-NMSE (S1, 0 dB) = {val_loss:.5f}")

# --------------- evaluation on 3 test-sets --------------
nmse_results = {
    'Compact':   evaluate_nmse_vs_snr_masked(model, test_loader_S1, device, SNR_LIST),
    'Dense':     evaluate_nmse_vs_snr_masked(model, test_loader_S2, device, SNR_LIST),
    'Standard':  evaluate_nmse_vs_snr_masked(model, test_loader_S3, device, SNR_LIST),
}

# --------------- save CSVs ------------------------------
os.makedirs("results", exist_ok=True)

with open("results/masked_nmse.csv", "w", newline="") as f:
    writer = csv.writer(f); writer.writerow(["Scenario","SNR","Masked NMSE"])
    for scen, curve in nmse_results.items():
        for snr, nmse in curve.items():
            writer.writerow([scen, snr, f"{nmse:.6e}"])
print("Saved results → results/masked_nmse.csv")
