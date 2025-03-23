import torch
import torch.nn as nn
import torch.optim as optim
from effecientnet import CustomLSTMModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


input_size = 1  
hidden_size = 16
num_layers = 2
output_size = 1  

# Initialize model and move to device
model = CustomLSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

# Define loss and optimizer
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Generate Dummy Data on CUDA
batch_size = 2
input_tensor = torch.randn(batch_size, 2, 18, 8, 3000).to(device)  # (batch, 2, 18, 8, 3000)
output_tensor = torch.randn(batch_size, 2, 8, 18).to(device)  # (batch, 2, 8, 18)
print("Batch Size:", batch_size)
print("Input Shape:", input_tensor.shape)
print("Output Shape:", output_tensor.shape)
total_params = sum(p.numel() for p in model.parameters())
print("Total Parameters:", total_params)

# Training Loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    predictions = model(input_tensor)  # Forward pass
    loss = criterion(predictions, output_tensor)  # Compute loss
    
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
