import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Generate random data
x = torch.unsqueeze(torch.linspace(0, 1, 100), dim=1)  # shape (100, 1)
y = 2 * x + 3 + 0.2 * torch.randn(x.size())  # Add noise

test_x = torch.unsqueeze(torch.linspace(0, 1, 100), dim=1)  # shape (100, 1)
test_y = 2 * x + 3 + 0.2 * torch.randn(x.size())  # Add noise

# Model parameter control
use_sigmoid = True    # Whether to enable sigmoid activation
use_linear2 = True    # Whether to enable second linear layer

class LinearModel(nn.Module):
    def __init__(self, use_sigmoid=False, use_linear2=False):
        super(LinearModel, self).__init__()
        self.linear1 = nn.Linear(1, 1)  # First linear layer
        self.use_linear2 = use_linear2
        if self.use_linear2:
            self.linear2 = nn.Linear(1, 1)  # Second linear layer
        self.use_sigmoid = use_sigmoid

    def forward(self, x):
        out = self.linear1(x)
        if self.use_sigmoid:
            out = torch.sigmoid(out)
        if self.use_linear2:
            out = self.linear2(out)
        return out

model = LinearModel(use_sigmoid=use_sigmoid, use_linear2=use_linear2)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

# Train the model
epochs = 1000
for epoch in range(epochs):
    # Forward propagation
    pred = model(x)
    loss = criterion(pred, y)
    
    # Backward propagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Plot results
plt.scatter(test_x.data.numpy(), test_y.data.numpy())
plt.plot(test_x.data.numpy(), model(test_x).data.numpy(), 'r-', lw=5)
plt.savefig(f'linear_fit_result_sigmoid_{use_sigmoid}_linear2_{use_linear2}.png')
plt.close()

print('Model parameters:')
for name, param in model.named_parameters():
    print(f'{name}: {param.item()}')