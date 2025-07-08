# PyTorch Operations in VAE Image Compression

## Encoder Components
- `nn.Conv2d`: Used for spatial feature extraction and downsampling. Example: `nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)`
- `nn.ReLU`: Activation function introducing non-linearity between convolutional layers
- `nn.Linear`: Projects high-dimensional features to latent space dimensions

## Decoder Components
- `nn.Linear`: Expands latent vector dimensions before upsampling
- `nn.ConvTranspose2d`: Performs upsampling to reconstruct image dimensions
- `nn.Sigmoid`: Normalizes output values to [0,1] range for image pixel values

## Reparameterization Trick
- `torch.mean`: Computes mean of latent distribution
- `torch.var`: Computes variance of latent distribution
- `torch.randn_like`: Generates random noise for stochastic sampling

## Loss Functions
- `nn.BCELoss`: Binary Cross-Entropy loss for image reconstruction
- Custom KL Divergence: Calculated using `torch.log`, `torch.var`, and `torch.mean` to regularize latent space

## Data Handling
- `DataLoader`: Efficiently batches and shuffles training data
- `ToTensor`: Converts PIL images to PyTorch tensors with normalized values