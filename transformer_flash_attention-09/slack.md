# PyTorch Ops and Functions Explanation

## torch.no_grad()
- **Purpose**: Context manager that disables gradient computation, used during inference to save memory and speed up computation.
- **Example**: 
  ```python
  with torch.no_grad():
      output = model(input)
  ```

## torch.multinomial()
- **Purpose**: Performs multinomial sampling on the input probability distribution and returns the sampled indices.
- **Parameters**: 
  - input: Probability tensor
  - num_samples: Number of samples to draw
- **Example**: 
  ```python
  probs = torch.tensor([0.1, 0.9])
  samples = torch.multinomial(probs, num_samples=1)
  ```

## torch.cat()
- **Purpose**: Concatenates tensors along a specified dimension.
- **Parameters**: 
  - tensors: Sequence of tensors
  - dim: Dimension to concatenate along
- **Example**: 
  ```python
  a = torch.tensor([1, 2])
  b = torch.tensor([3, 4])
  c = torch.cat([a, b], dim=0)  # Result: tensor([1, 2, 3, 4])
  ```

## torch.nn.functional.softmax()
- **Purpose**: Applies the softmax function to convert logits into a probability distribution.
- **Parameters**: 
  - input: Input tensor
  - dim: Dimension along which to compute softmax
- **Example**: 
  ```python
  logits = torch.tensor([1.0, 2.0, 3.0])
  probs = torch.nn.functional.softmax(logits, dim=0)
  ```

## torch.ones_like()
- **Purpose**: Creates a tensor filled with ones that has the same shape as the input tensor.
- **Parameters**: 
  - input: Input tensor
- **Example**: 
  ```python
  x = torch.tensor([[0, 1], [2, 3]])
  y = torch.ones_like(x)  # Result: tensor([[1, 1], [1, 1]])
  ```