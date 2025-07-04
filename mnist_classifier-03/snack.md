# PyTorch Operations and Functions Explained

This document explains the key PyTorch operations and functions used in `main.py` for MNIST classification.

## Core PyTorch Components

### 1. `torch.nn.Module`
- Base class for all neural network modules in PyTorch
- All custom models must inherit from this class
- Provides infrastructure for parameter management and GPU acceleration

### 2. `nn.Flatten()`
- Layer that flattens multi-dimensional input tensors into a 1D tensor
- Used in the model to convert 28x28 MNIST images into 784-dimensional vectors
- Example: Converts tensor shape (batch_size, 1, 28, 28) → (batch_size, 784)

### 3. `nn.Linear(in_features, out_features)`
- Implements fully connected (dense) layer
- Applies linear transformation: `y = xA^T + b`
- Used in the model for `fc1` (784→128) and `fc2` (128→10) layers

### 4. `nn.ReLU()`
- Rectified Linear Unit activation function
- Applies element-wise operation: `ReLU(x) = max(0, x)`
- Introduces non-linearity between linear layers

### 5. `nn.CrossEntropyLoss()`
- Combination of `log_softmax()` and `nll_loss()`
- Used for multi-class classification tasks
- Automatically handles raw logits (no need for softmax layer before)

## Optimization

### 6. `torch.optim.SGD(params, lr, momentum)`
- Stochastic Gradient Descent optimizer
- `params`: Model parameters to optimize
- `lr`: Learning rate (0.01 in our code)
- `momentum`: Accelerates SGD by considering past gradients (0.5 in our code)

## Data Handling

### 7. `torchvision.datasets.MNIST`
- Built-in MNIST dataset loader
- Automatically downloads data if not available
- Applies transformations through `transform` parameter

### 8. `torch.utils.data.DataLoader`
- Wraps dataset and provides iterable over batches
- `batch_size`: Number of samples per batch (64 for training)
- `shuffle`: Randomizes order of training data

### 9. `transforms.Compose()`
- Chains multiple data transformations together
- In our code: converts PIL images to tensors and normalizes pixel values

## Training Workflow

### 10. `model.train()`
- Sets model to training mode
- Enables gradient computation for all parameters
- Activates dropout/batch normalization training behavior

### 11. `model.eval()`
- Sets model to evaluation/inference mode
- Disables gradient computation
- Uses running statistics for batch normalization

### 12. `torch.no_grad()`
- Context manager that disables gradient computation
- Reduces memory usage and speeds up inference
- Used during the testing phase

### 13. `optimizer.zero_grad()`
- Clears accumulated gradients from previous iterations
- Prevents gradient accumulation across batches

### 14. `loss.backward()`
- Computes gradients of the loss with respect to all trainable parameters
- Starts backpropagation from the loss tensor

### 15. `optimizer.step()`
- Updates model parameters using computed gradients
- Applies the optimization algorithm (SGD in our case)

## Tensor Operations

### 16. `output.argmax(dim=1)`
- Finds index of maximum value along dimension 1
- Used to get predicted class from model output logits
- `keepdim=True` preserves tensor dimensions for comparison

### 17. `target.view_as(pred)`
- Reshapes the target tensor to match the shape of the prediction tensor
- Ensures compatible tensor shapes for comparison operations
- Example: Converts tensor shape (batch_size,) → (batch_size, 1) to match prediction shape

### 18. `.item()`
- Extracts a scalar value from a 0-dimensional tensor
- Converts tensor to a standard Python float
- Used to get numerical values from loss tensors or single-value tensors
- Example: `loss.item()` converts a tensor containing a single loss value to a Python number

### 19. `torch.unsqueeze(0)`
  - Adds a new dimension at specified position
  - Used to add batch dimension to single sample for inference

These components work together to create a complete training pipeline: data loading → model definition → forward pass → loss calculation → backpropagation → parameter update → evaluation.