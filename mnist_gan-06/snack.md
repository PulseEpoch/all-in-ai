# Explanation of PyTorch Operations and Functions Used in MNIST GAN

## Core Modules and Basic Components
- **`torch.nn.Module`**: The base class for all neural network modules in PyTorch. Both Generator and Discriminator inherit from this class to implement network functionality.
- **`torch.nn.Sequential`**: A container that wraps multiple network layers in sequence, used to simplify network definition. In both Generator and Discriminator, the network body is constructed via `self.main = nn.Sequential(...)`.

## Network Layer Components
### Convolutional and Transposed Convolutional Layers
- **`nn.Conv2d`**: 2D convolutional layer used to extract features from input images. Used in the Discriminator for downsampling, e.g., `nn.Conv2d(image_channels, hidden_dim, 4, 2, 1, bias=False)` downsamples input images from 28x28 to 14x14.
- **`nn.ConvTranspose2d`**: Transposed convolutional layer (deconvolution) used for upsampling to generate images. Used in the Generator to gradually increase feature map size, e.g., `nn.ConvTranspose2d(latent_dim, hidden_dim * 4, 4, 1, 0, bias=False)` upsamples a 1x1 latent vector to a 4x4 feature map.

### Normalization Layers
- **`nn.BatchNorm2d`**: Batch normalization layer that accelerates network training and improves stability by normalizing input data. Used with convolutional layers in the middle layers of both Generator and Discriminator, e.g., `nn.BatchNorm2d(hidden_dim * 2)`.

### Activation Functions
- **`nn.ReLU(True)`**: ReLU activation function that introduces non-linear transformations. The `inplace=True` parameter saves memory. Used in the Generator for middle layer feature transformations.
- **`nn.LeakyReLU(0.2, inplace=True)`**: Leaky ReLU that allows small gradients to pass through negative input regions, mitigating the dying neuron problem. Used in all middle layers of the Discriminator.
- **`nn.Tanh()`**: Tanh activation function with output range [-1, 1]. Used in the Generator's output layer to normalize generated images to this range.
- **`nn.Sigmoid()`**: Sigmoid activation function with output range [0, 1]. Used in the Discriminator's output layer to convert results to probability values.

## Loss Functions and Optimizers
- **`nn.BCELoss()`**: Binary cross-entropy loss function used to calculate loss between real labels and predicted probabilities. Used in GAN for both Discriminator (distinguishing real/generated images) and Generator (fooling the Discriminator) loss calculations.
- **`optim.Adam`**: Adam optimizer that combines momentum and adaptive learning rate strategies to accelerate model convergence. Generator and Discriminator each use separate Adam optimizers with learning rate `lr=0.0002` and momentum parameters `betas=(0.5, 0.999)`.

## Data Processing and Loading
- **`transforms.Compose`**: Container for combining multiple data preprocessing operations. Used in MNIST data preprocessing to chain `ToTensor()` and `Normalize()`.
- **`transforms.ToTensor()`**: Converts PIL images to PyTorch tensors and normalizes pixel values to [0, 1].
- **`transforms.Normalize((0.5,), (0.5,))`**: Normalizes tensors, adjusting pixel values from [0, 1] to [-1, 1] (mean=0.5, standard deviation=0.5).
- **`datasets.MNIST`**: MNIST handwritten digit dataset loader that automatically downloads and loads training data.
- **`DataLoader`**: Iterator for batch loading data, supporting data shuffling (`shuffle=True`) and multi-threaded loading.

## Tensor Operations and Device Configuration
- **`torch.device('cuda' if torch.cuda.is_available() else 'mps')`**: Automatically selects the computing device (prioritizing GPU, then MPS, defaulting to CPU).
- **`torch.randn`**: Generates random tensors following a standard normal distribution, used for Generator input latent vectors and fixed noise (`fixed_noise`).
- **`torch.ones`/`torch.zeros`**: Generates all-1 or all-0 tensors, used to define labels for real/fake images (`real_labels`/`fake_labels`).
- **`torch.manual_seed(42)`**: Sets random seed to ensure reproducible experimental results.
- **`torch.no_grad()`**: Context manager that disables gradient computation, used in inference phase (e.g., when generating samples) for efficiency.

## Model Saving and Image Generation
- **`torch.save`**: Saves model weights to files. After training, saves Generator and Discriminator parameters to `generator_weights.pth` and `discriminator_weights.pth`.
- **`save_image`**: Imported from `torchvision.utils`, saves tensors as image files. Generates and saves 16 sample images to the `generated_images` directory after each epoch.