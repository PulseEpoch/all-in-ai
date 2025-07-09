import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
from PIL import Image as PILImage
from datasets import Image
import numpy as np
from vae_model import VAE
import matplotlib.pyplot as plt
import os
# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),  # PIL image to tensor
])

def train_vae(args):
    # Set device
    device = torch.device(args.device)

    # Load Hugging Face MNIST dataset
    dataset = load_dataset('mnist')
    dataset = dataset.cast_column('image', Image())
    train_dataset = dataset['train'].map(
        lambda x: {'image': transform(x['image'])})
    test_dataset = dataset['test'].map(
        lambda x: {'image': transform(x['image'])})

    train_loader = DataLoader(train_dataset.with_format(
        "torch"), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset.with_format(
        "torch"), batch_size=args.batch_size, shuffle=False)

    # Initialize model and optimizer
    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Loss function
    def loss_function(recon_x, x, mu, logvar):
        bce_loss = torch.nn.BCELoss(reduction='sum')
        BCE = bce_loss(recon_x.view(-1, 784), x.view(-1, 784))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            data = batch['image'].to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(
            f'Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset)}')

    # Save model
    torch.save(model.state_dict(), 'vae_model.pth')
    return model


def compress_image(model, image, device):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        mu, logvar = model.encode(image)
        z = model.reparameterize(mu, logvar)
    return z.cpu().numpy()


def decompress_image(model, z, device):
    model.eval()
    with torch.no_grad():
        z = torch.tensor(z).to(device)
        recon_image = model.decode(z)
    return recon_image.squeeze().cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE Image Compression')
    parser.add_argument('--latent_dim', type=int, default=16, help='Latent dimension size')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training (cpu or cuda)')
    args = parser.parse_args()

    # Create save directory
    os.makedirs('results', exist_ok=True)

    # Train model
    model = train_vae(args)

    # Demonstrate compression and decompression
    device = torch.device(args.device)
    test_dataset = load_dataset('mnist', split='test')
    num_images = 9
    plt.figure(figsize=(15, 15))
    
    for i in range(num_images):
        test_image = transforms.ToTensor()(test_dataset[i]['image'])
        
        # Compress
        z = compress_image(model, test_image, device)
        
        # Decompress
        recon_image = decompress_image(model, z, device)
        
        # Display decompressed image
        plt.subplot(3, 3, i+1)
        plt.imshow(recon_image, cmap='gray')
        plt.title(f'Decompressed Image {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/compressed_9images.png')
    print('9 compressed images saved to results/compressed_9images.png')

    # Save results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(test_image.squeeze(), cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(recon_image, cmap='gray')
    plt.title('Decompressed Image')
    plt.savefig('results/compression_demo.png')
    print('Compression demo saved to results/compression_demo.png')
