# Training the model sequentially on S1 -> S2 -> S3 using LwF (Knowledge Distillation)
model_lwf = LSTMModel(input_dim=1,
            hidden_dim=32,
            output_dim=1,
            n_layers=3,
            H=16,
            W=36).to(device)
optimizer = torch.optim.Adam(model_lwf.parameters(), lr=1e-3)
criterion = nn.MSELoss()

num_epochs = 10
distill_lambda = 0.5  # weight for distillation loss

# Train on Task 1 (S1) normally (no old model yet)
for epoch in range(num_epochs):
    for X_batch, Y_batch in train_loader_S1:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        optimizer.zero_grad()
        pred = model_lwf(X_batch)
        loss = criterion(pred, Y_batch)
        loss.backward()
        optimizer.step()

# Save the Task 1 model as the old model for LwF
old_model = copy.deepcopy(model_lwf).to(device)
old_model.eval()
for param in old_model.parameters():
    param.requires_grad = False  # freeze old model

# Train on Task 2 (S2) with LwF
optimizer = torch.optim.Adam(model_lwf.parameters(), lr=1e-3)
for epoch in range(num_epochs):
    for X_batch, Y_batch in train_loader_S2:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        optimizer.zero_grad()
        # Forward pass new model
        pred = model_lwf(X_batch)
        # Forward pass old model (with no grad)
        with torch.no_grad():
            old_pred = old_model(X_batch)
        # Compute losses
        task_loss = criterion(pred, Y_batch)  # new task supervision loss
        distill_loss = criterion(pred, old_pred)  # L2 loss between new and old outputs
        loss = task_loss + distill_lambda * distill_loss
        loss.backward()
        optimizer.step()

# Update old_model to the model after Task 2
old_model = copy.deepcopy(model_lwf).to(device)
old_model.eval()
for param in old_model.parameters():
    param.requires_grad = False

# Train on Task 3 (S3) with LwF
optimizer = torch.optim.Adam(model_lwf.parameters(), lr=1e-3)
for epoch in range(num_epochs):
    for X_batch, Y_batch in train_loader_S3:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        optimizer.zero_grad()
        pred = model_lwf(X_batch)
        with torch.no_grad():
            old_pred = old_model(X_batch)
        task_loss = criterion(pred, Y_batch)
        distill_loss = criterion(pred, old_pred)
        loss = task_loss + distill_lambda * distill_loss
        loss.backward()
        optimizer.step()

# Evaluate final model on all tasks (NMSE vs SNR)
print("LwF Method - NMSE on each task across SNRs:")
nmse_results_lwf = {}
nmse_results_lwf['S1'] = evaluate_nmse_vs_snr(model_lwf, test_loader_S1, device, snr_list)
nmse_results_lwf['S2'] = evaluate_nmse_vs_snr(model_lwf, test_loader_S2, device, snr_list)
nmse_results_lwf['S3'] = evaluate_nmse_vs_snr(model_lwf, test_loader_S3, device, snr_list)
for task, nmse_vs_snr in nmse_results_lwf.items():
    print(f"Task {task}: " + ", ".join([f"SNR {snr}: NMSE {nmse:.4f}" 
                                        for snr, nmse in nmse_vs_snr.items()]))
