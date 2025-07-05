# PyTorch Operations and Functions in MNIST Distillation

This document lists and explains PyTorch operations and functions used in the distillation model:

## Core Components
- **nn.Module**: Base class for all neural network modules in PyTorch. Custom models must inherit from this class.
- **nn.Conv2d**: 2D convolutional layer used for extracting image features. Parameters include input channels, output channels, and kernel size.
- **nn.MaxPool2d**: 2D max pooling layer for downsampling feature maps. Parameters include kernel size and stride.
- **nn.Linear**: Fully connected linear layer that applies a linear transformation to the input. Parameters include input and output feature sizes.
- **nn.ReLU**: Rectified Linear Unit activation function that introduces non-linearity by setting negative values to zero.
- **nn.Flatten**: Layer that flattens multi-dimensional input tensors into a 1D tensor for input to linear layers.

## Loss Functions
- **nn.Softmax**: Applies softmax function along a specified dimension to convert logits into probabilities.
- **nn.LogSoftmax**: Applies log softmax function, useful for calculating cross-entropy loss.
- **nn.KLDivLoss**: Kullback-Leibler divergence loss that measures the difference between two probability distributions. Used for knowledge distillation.
- **nn.CrossEntropyLoss**: Combines LogSoftmax and NLLLoss for classification tasks. Computes loss between input logits and target labels.
- **F.cross_entropy**: Functional API for cross-entropy loss. Combines log softmax and negative log likelihood loss in a single function call.

## Tensor Operations
- **x.view**: Reshapes tensor dimensions without changing data. Similar to NumPy's reshape function. Used to flatten tensors before linear layers.
- **torch.save**: Saves a serialized object to disk. Used to save model state dictionaries for later use.
- **torch.device**: Represents the device (CPU/GPU) on which tensors and models are allocated.
- **torch.no_grad()**: Context manager that disables gradient computation, reducing memory usage and speeding up inference.
- **torch.cuda.is_available()**: Checks if CUDA is available for GPU acceleration.

## Optimizers
- **optim.SGD**: Stochastic Gradient Descent optimizer. Parameters include model parameters, learning rate, and momentum. Used to update model weights during training.

## Training Utilities
- **model.train()**: Sets the model to training mode, enabling dropout and batch normalization if present.
- **model.eval()**: Sets the model to evaluation mode, disabling dropout and batch normalization. Used during inference/testing.
- **optimizer.zero_grad()**: Resets gradients of all model parameters to zero to prevent accumulation from previous batches.
- **loss.backward()**: Computes gradients of the loss with respect to all trainable parameters using automatic differentiation.
- **optimizer.step()**: Updates model parameters using the computed gradients.

## Data Loading
- **DataLoader**: Provides an iterable over a dataset. Supports batching, shuffling, and parallel loading of data.
- **datasets.MNIST**: MNIST dataset class for loading handwritten digit images and corresponding labels. Supports automatic downloading and transformation of data.