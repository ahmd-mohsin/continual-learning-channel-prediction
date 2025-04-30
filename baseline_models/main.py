import os, csv, argparse, torch
from tqdm import tqdm

from dataloader import get_all_datasets
from model      import LSTMChannelPredictor          # 2-channel LSTM (mag + mask)
from loss       import masked_nmse                   # masked NMSE function
from utils      import compute_device, evaluate_nmse_vs_snr_masked
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

# ----K---------- hyper-parameters -----------------------
BATCH_SIZE = 2048
SEQ_LEN    = 16
NUM_EPOCHS = 100
ALPHA      = 0.2         # weight for BCE mask loss
LR         = 1e-4
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
def init_weights(m):
    if isinstance(m, (nn.Linear, nn.LSTM)):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
model.apply(init_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
bce_loss  = torch.nn.BCEWithLogitsLoss()
# new magnitude-error terms
huber_loss_fn = torch.nn.SmoothL1Loss()
l1_loss_fn   = torch.nn.L1Loss()

# sched     = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
sched = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

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

        # original masked NMSE
        loss_mag  = masked_nmse(mag_p, mag_t, mask_t)


        loss_mask = bce_loss(mask_logits, mask_t)
        # composite loss
        loss = (
            loss_mag
          + ALPHA * loss_mask
        )

        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running += loss.item(); n += 1
        pbar.set_postfix(
            nmse = loss_mag.item(),
            bce  = loss_mask.item(),
        )

    print(f"  ↳ avg train-loss: {running/n:.5f}")
    sched.step(running/n)

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
