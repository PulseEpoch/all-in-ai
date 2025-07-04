# PyTorch Operations & Functions in linear_classifier.py

## Tensor Creation
- `torch.randn(size)`: Creates tensor with values from standard normal distribution
  - Example: `torch.randn(50, 2)` - Generates 50x2 random feature matrix
- `torch.tensor(data)`: Creates tensor from data
  - Example: `torch.tensor([2., 2.])` - Creates constant shift tensor
- `torch.zeros(size)`: Creates tensor filled with zeros
  - Example: `torch.zeros(50)` - Generates labels for class 0
- `torch.ones(size)`: Creates tensor filled with ones
  - Example: `torch.ones(50)` - Generates labels for class 1

## Tensor Operations
- `torch.cat(tensors, dim)`: Concatenates tensors along specified dimension
  - Example: `torch.cat([class1, class2], dim=0)` - Combines two classes of data
- `torch.unsqueeze(dim)`: Adds singleton dimension
  - Example: `y.unsqueeze(1)` - Converts 1D labels to 2D for concatenation
- `torch.randperm(n)`: Generates random permutation of integers 0 to n-1
  - Example: `combined[torch.randperm(combined.size(0))]` - Shuffles dataset
- `torch.as_tensor(data)`: Converts data to tensor (preserves dtype if possible)
  - Example: `torch.as_tensor(y_true)` - Converts labels to tensor for accuracy calculation
- `torch.view(shape)`: Reshapes tensor
  - Example: `y.view(-1, 1)` - Reshapes labels for BCE loss compatibility
- `torch.squeeze(dim)`: Removes singleton dimensions
  - Example: `combined[:, 2].squeeze()` - Removes extra dimension from labels
- `torch.no_grad()`: Context manager to disable gradient computation
  - Used in prediction function to improve performance

## Neural Network Components
- `nn.Module`: Base class for all neural network modules
  - Subclassed by `LogisticRegression` class
- `nn.Linear(in_features, out_features)`: Linear transformation layer
  - Example: `nn.Linear(2, 1)` - 2 input features to 1 output
- `nn.Sigmoid()`: Sigmoid activation function
  - Applied after linear layer for binary classification
- `nn.BCELoss()`: Binary Cross-Entropy loss
  - Used for training binary classification model

## Optimization
- `optim.SGD(params, lr)`: Stochastic Gradient Descent optimizer
  - Example: `optim.SGD(model.parameters(), lr=0.1)` - Optimizes model parameters

## Data Utilities
- `torch.utils.data.TensorDataset(data_tensor, target_tensor)`: Creates dataset from tensors
  - Combines features and labels for batching
- `torch.utils.data.DataLoader(dataset, batch_size, shuffle)`: Creates iterable over dataset
  - Example: `DataLoader(dataset, batch_size=32, shuffle=True)` - Enables batch training