class EWC:
    """Elastic Weight Consolidation helper to store Fisher information and original parameters."""
    def __init__(self, model: nn.Module, data_loader: DataLoader, device: torch.device, sample_size=None):
        """
        Compute the Fisher information diagonal and parameter snapshot on the given data (previous task).
        Args:
            model: Trained model on the previous task (whose parameters we want to remember).
            data_loader: DataLoader for the previous task's training data.
            device: Device on which to perform computations.
            sample_size: If set, limit the number of samples used to compute Fisher (for efficiency).
        """
        self.device = device
        # Store the reference parameters (theta^* from the old task)
        self.params_snapshot = {name: p.clone().detach() for name, p in model.named_parameters()}
        # Initialize Fisher information for each parameter to zero
        self.fisher_diag = {name: torch.zeros_like(p, device=device) for name, p in model.named_parameters()}
        
        # Set model to evaluation mode and compute Fisher information
        model.eval()
        criterion = nn.MSELoss(reduction='mean')  # use MSE as proxy loss
        total_samples = len(data_loader.dataset) if sample_size is None else sample_size
        count = 0
        for X_batch, Y_batch in data_loader:
            # Limit the number of samples if sample_size is specified
            if sample_size is not None and count >= sample_size:
                break
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            model.zero_grad()
            # Forward pass
            pred = model(X_batch)
            # Compute loss (using ground truth, i.e., empirical Fisher)
            loss = criterion(pred, Y_batch)
            # Backward pass to compute gradients
            loss.backward()
            # Accumulate squared gradients
            for name, p in model.named_parameters():
                if p.grad is not None:
                    # sum of squared grad
                    self.fisher_diag[name] += (p.grad.detach() ** 2)
            count += X_batch.size(0)
        # Average the Fisher information
        for name in self.fisher_diag:
            self.fisher_diag[name] /= float(min(total_samples, count))
        model.train()
    
    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute the EWC penalty term for the given model's current parameters.
        This will be added to the loss to penalize deviation from stored parameters.
        """
        penalty = 0.0
        for name, p in model.named_parameters():
            # Fisher * (theta - theta_old)^2
            if name in self.fisher_diag:
                diff = p - self.params_snapshot[name]
                penalty += torch.sum(self.fisher_diag[name] * (diff ** 2))
        return penalty

# Training the model sequentially on S1 -> S2 -> S3 using EWC
model_ewc = LSTMModel().to(device)
optimizer = torch.optim.Adam(model_ewc.parameters(), lr=1e-3)
criterion = nn.MSELoss()

num_epochs = 10  # epochs per task (adjust as needed)
ewc_lambda = 0.4  # regularization strength for EWC (tunable hyperparameter)

# Train on Task 1 (S1) normally
for epoch in range(num_epochs):
    for X_batch, Y_batch in train_loader_S1:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        optimizer.zero_grad()
        pred = model_ewc(X_batch)
        loss = criterion(pred, Y_batch)
        loss.backward()
        optimizer.step()
    # (Optional) evaluate on S1 validation here

# After Task 1, compute Fisher info on S1 for EWC
ewc_S1 = EWC(model_ewc, train_loader_S1, device=device)

# Train on Task 2 (S2) with EWC regularization
# We reinitialize optimizer for a new task to avoid carrying momentum from previous task
optimizer = torch.optim.Adam(model_ewc.parameters(), lr=1e-3)
for epoch in range(num_epochs):
    for X_batch, Y_batch in train_loader_S2:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        optimizer.zero_grad()
        pred = model_ewc(X_batch)
        base_loss = criterion(pred, Y_batch)
        # EWC penalty from Task 1
        penalty = ewc_S1.penalty(model_ewc)
        loss = base_loss + ewc_lambda * penalty
        loss.backward()
        optimizer.step()
    # (Optional) evaluate on S1 and S2 validation here to monitor forgetting

# After Task 2, compute Fisher info on S2 and combine with S1 for EWC
ewc_S2 = EWC(model_ewc, train_loader_S2, device=device)
# We can combine EWC from S1 and S2 by summing their penalties
# (Alternatively, create a single EWC object that stores multiple tasks)

# Train on Task 3 (S3) with EWC regularization (Tasks 1 & 2 penalties)
optimizer = torch.optim.Adam(model_ewc.parameters(), lr=1e-3)
for epoch in range(num_epochs):
    for X_batch, Y_batch in train_loader_S3:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        optimizer.zero_grad()
        pred = model_ewc(X_batch)
        base_loss = criterion(pred, Y_batch)
        # Penalty from Task 1 and Task 2 EWC
        penalty = ewc_S1.penalty(model_ewc) + ewc_S2.penalty(model_ewc)
        loss = base_loss + ewc_lambda * penalty
        loss.backward()
        optimizer.step()

# Evaluate final model on all tasks (NMSE vs SNR)
snr_list = [0, 5, 10, 15, 20]  # example SNR values in dB
print("EWC Method - NMSE on each task across SNRs:")
nmse_results_ewc = {}
nmse_results_ewc['S1'] = evaluate_nmse_vs_snr(model_ewc, test_loader_S1, device, snr_list)
nmse_results_ewc['S2'] = evaluate_nmse_vs_snr(model_ewc, test_loader_S2, device, snr_list)
nmse_results_ewc['S3'] = evaluate_nmse_vs_snr(model_ewc, test_loader_S3, device, snr_list)
for task, nmse_vs_snr in nmse_results_ewc.items():
    print(f"Task {task}: " + ", ".join([f"SNR {snr}: NMSE {nmse:.4f}" 
                                        for snr, nmse in nmse_vs_snr.items()]))
