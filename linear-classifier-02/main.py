import torch
import matplotlib.pyplot as plt
from torch import nn, optim

"""
Linear Binary Classifier using Logistic Regression
This implementation demonstrates a simple binary classification task using PyTorch.
It generates synthetic data, trains a logistic regression model, and visualizes results.
"""

def generate_data():
    """
    Generate synthetic binary classification data
    
    Returns:
        X (torch.Tensor): Feature matrix with shape (100, 2)
        y (torch.Tensor): Label vector with shape (100,) containing 0 and 1
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Generate two classes of data points with different means
    class1 = torch.randn(50, 2) + torch.tensor([2., 2.])  # Class 0: around (2, 2)
    class2 = torch.randn(50, 2) + torch.tensor([-2., -2.])  # Class 1: around (-2, -2)
    
    # Combine data and labels
    X = torch.cat([class1, class2])
    y = torch.cat([torch.zeros(50), torch.ones(50)])

    # Combine features and labels for shuffling
    combined = torch.cat([X, y.unsqueeze(1)], dim=1)
    # Shuffle the combined dataset to randomize order
    combined = combined[torch.randperm(combined.size(0))]
    # Split back into features and labels
    X = combined[:, :2]
    y = combined[:, 2].squeeze()
    
    return X, y

class LogisticRegression(nn.Module):
    """
    Logistic Regression model for binary classification
    
    This model applies a linear transformation followed by a sigmoid activation
    to produce probabilities between 0 and 1.
    """
    def __init__(self):
        super().__init__()
        # Linear layer: 2 input features -> 1 output feature
        self.linear = nn.Linear(2, 1)
        # Sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the model
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, 2)
            
        Returns:
            torch.Tensor: Output probabilities with shape (batch_size, 1)
        """
        return self.sigmoid(self.linear(x))

def train_linear_classifier(X, y, learning_rate=0.1, epochs=1000):
    """
    Train a logistic regression classifier
    
    Args:
        X (torch.Tensor): Feature matrix with shape (n_samples, 2)
        y (torch.Tensor): Label vector with shape (n_samples,)
        learning_rate (float): Learning rate for optimization
        epochs (int): Number of training epochs
    
    Returns:
        weights (numpy.ndarray): Learned weights with shape (2,)
        bias (numpy.ndarray): Learned bias term
    """
    # Initialize model, loss function, and optimizer
    model = LogisticRegression()
    criterion = nn.BCELoss()  # Binary Cross-Entropy loss
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # Stochastic Gradient Descent

    # Ensure proper data types
    X = X.float()
    y = y.view(-1, 1).float()
    
    # Create DataLoader for batch processing
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True)

    # Training loop
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()  # Reset gradients
            outputs = model(batch_X)  # Forward pass
            loss = criterion(outputs, batch_y)  # Calculate loss
            loss.backward()  # Backward pass (compute gradients)
            optimizer.step()  # Update weights

    # Extract learned parameters
    weights = model.linear.weight.data.numpy()[0]
    bias = model.linear.bias.data.numpy()[0]
    return weights, bias

def predict(X, weights, bias):
    """
    Make predictions using learned logistic regression parameters
    
    Args:
        X (torch.Tensor or numpy.ndarray): Feature matrix with shape (n_samples, 2)
        weights (numpy.ndarray): Learned weights with shape (2,)
        bias (numpy.ndarray): Learned bias term
    
    Returns:
        list: Predicted class labels (0 or 1) for each sample
    """
    with torch.no_grad():
        # Convert to tensor if necessary
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(X, dtype=torch.float32)
        else:
            X_tensor = X
        
        # Linear combination of features and weights
        linear_model = X_tensor @ torch.tensor(weights) + bias
        # Apply threshold (0.5 after sigmoid) and convert to class labels
        y_pred = (linear_model >= 0).int().numpy()
    return y_pred.flatten().tolist()

def calculate_accuracy(y_true, y_pred):
    """
    Calculate classification accuracy
    
    Args:
        y_true (torch.Tensor or list): True labels
        y_pred (torch.Tensor or list): Predicted labels
    
    Returns:
        torch.Tensor: Accuracy score between 0 and 1
    """
    y_true_tensor = torch.as_tensor(y_true)
    y_pred_tensor = torch.as_tensor(y_pred)
    return torch.mean((y_true_tensor == y_pred_tensor).float())

def plot_results(X, y, weights, bias):
    """
    Visualize classification results and decision boundary
    
    Args:
        X (torch.Tensor): Feature matrix with shape (n_samples, 2)
        y (torch.Tensor): Label vector with shape (n_samples,)
        weights (numpy.ndarray): Learned weights with shape (2,)
        bias (numpy.ndarray): Learned bias term
    """
    X_np = X.numpy()
    y_np = y.numpy()
    
    # Plot data points for both classes
    plt.scatter(X_np[y_np == 0][:, 0], X_np[y_np == 0][:, 1], color='red', label='Class 0')
    plt.scatter(X_np[y_np == 1][:, 0], X_np[y_np == 1][:, 1], color='blue', label='Class 1')

    # Calculate and plot decision boundary
    x_min, x_max = X_np[:, 0].min() - 1, X_np[:, 0].max() + 1
    x_values = torch.tensor([x_min, x_max], dtype=torch.float32)
    # Decision boundary equation: weights[0]*x + weights[1]*y + bias = 0
    y_values = (-(weights[0] * x_values + bias) / weights[1]).numpy()
    plt.plot([x_min, x_max], y_values, 'g--', label='Decision Boundary')

    # Add labels and save plot
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Linear Binary Classifier')
    plt.legend()
    plt.savefig('decision_boundary.png')
    plt.close()

def main():
    """
    Main function to execute the linear classification pipeline
    """
    # Generate synthetic classification data
    X, y = generate_data()

    # Train logistic regression model
    weights, bias = train_linear_classifier(X, y)

    # Make predictions on training data
    y_pred = predict(X, weights, bias)

    # Calculate and print model accuracy
    accuracy = calculate_accuracy(y, y_pred)
    print(f"Model Accuracy: {accuracy:.2f} ({accuracy*100:.0f}%)")

    # Visualize results and save plot
    plot_results(X, y, weights, bias)
    print("Decision boundary plot saved as 'decision_boundary.png'")

if __name__ == "__main__":
    main()