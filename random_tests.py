import torch
import torch.nn as nn
import torch.optim as optim

# Generate synthetic data: y = 2x + 3 + noise
x = torch.unsqueeze(torch.linspace(-10, 10, 100), dim=1)  # Shape: [100, 1]
y = 2 * x + 3 + torch.randn(x.size()) * 2  # Adding some noise

# Define a simple linear regression model
model = nn.Linear(1, 1)  # One input, one output

# Define the loss function (Mean Squared Error) and optimizer (Stochastic Gradient Descent)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()           # Zero the gradients
    predictions = model(x)          # Forward pass: compute predictions
    loss = criterion(predictions, y)  # Compute loss
    loss.backward()                 # Backward pass: compute gradients
    optimizer.step()                # Update model parameters

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# Print learned parameters (weight and bias)
for name, param in model.named_parameters():
    print(f"{name}: {param.data}")