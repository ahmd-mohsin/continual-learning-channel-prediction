import torch
import torch.nn as nn
import torch.optim as optim

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define LSTM Model
class CustomLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CustomLSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
        self.fc = nn.Linear(hidden_size, output_size).to(device)
    
    def forward(self, x):
        batch_size, channels, height, width, time_steps = x.shape  # (batch, 2, 18, 8, 3000)
        
        # Move input tensor to device
        x = x.to(device)
        
        # Reshape to feed into LSTM
        x = x.view(batch_size * channels * height * width, time_steps, -1)  # (batch * 2 * 18 * 8, 3000, feature_dim)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Get the last time step output
        lstm_out = lstm_out[:, -1, :]  # (batch * 2 * 18 * 8, hidden_size)
        
        # Fully connected layer to match output shape
        out = self.fc(lstm_out)  # (batch * 2 * 18 * 8, output_size)
        
        # Reshape to match desired output
        out = out.view(batch_size, channels, width, height)  # (batch, 2, 8, 18)
        
        return out


# Define model parameters
input_size = 1  # Each time step has one feature
hidden_size = 8
num_layers = 2
output_size = 1  # Predicting one value per unit

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
