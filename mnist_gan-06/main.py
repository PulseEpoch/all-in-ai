import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os

# Create directory to save results
os.makedirs('generated_images', exist_ok=True)

# Hyperparameter settings
latent_dim = 100
hidden_dim = 128
image_channels = 1
image_size = 28
batch_size = 128
epochs = 50
lr = 0.0002
beta1 = 0.5

# Data preprocessing and loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(
    root='../data', train=True, download=True, transform=transform
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Generator network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, hidden_dim * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(True),
            # Output: (hidden_dim*4) x 4 x 4

            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(True),
            # Output: (hidden_dim*2) x 7 x 7

            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            # Output: hidden_dim x 14 x 14

            nn.ConvTranspose2d(hidden_dim, image_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: image_channels x 28 x 28
        )

    def forward(self, input):
        return self.main(input)

# Discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: image_channels x 28 x 28
            nn.Conv2d(image_channels, hidden_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: hidden_dim x 14 x 14

            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (hidden_dim*2) x 7 x 7

            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (hidden_dim*4) x 4 x 4

            nn.Conv2d(hidden_dim * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Output: 1 x 1 x 1
        )

    def forward(self, input):
        return self.main(input)

# Initialize models, loss function and optimizers
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Fixed noise for generating samples
torch.manual_seed(42)
fixed_noise = torch.randn(16, latent_dim, 1, 1, device=device)

# Training loop
for epoch in range(epochs):
    for i, (real_images, _) in enumerate(dataloader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)

        # Labels
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # Train Discriminator
        discriminator.zero_grad()

        # Real images
        outputs = discriminator(real_images).view(-1, 1)
        d_loss_real = criterion(outputs, real_labels)
        d_loss_real.backward()
        d_real_acc = outputs.mean().item()

        # Generated images
        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_images = generator(noise)
        outputs = discriminator(fake_images.detach()).view(-1, 1)
        d_loss_fake = criterion(outputs, fake_labels)
        d_loss_fake.backward()
        d_fake_acc = outputs.mean().item()

        # Optimize Discriminator
        d_loss = d_loss_real + d_loss_fake
        optimizer_D.step()

        # Train Generator
        generator.zero_grad()
        outputs = discriminator(fake_images).view(-1, 1)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_G.step()

        # Print progress
        if i % 100 == 0:
            print(f'[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] '
                  f'[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}] '
                  f'[D real: {d_real_acc:.4f}] [D fake: {d_fake_acc:.4f}]')

    # Save generated images every epoch
    with torch.no_grad():
        fake = generator(fixed_noise).detach().cpu()
    save_image(fake, f'generated_images/epoch_{epoch}.png', nrow=4, normalize=True)

# Save model weights
torch.save(generator.state_dict(), 'generator_weights.pth')
torch.save(discriminator.state_dict(), 'discriminator_weights.pth')

print('Training completed! Generated images are saved in the generated_images folder, and model weights have been saved.')