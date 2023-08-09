import torch.nn as nn
import torch


class Discriminator(nn.Module):
    """
    Discriminator model for searching VAE latent space
    Args:
        latent_size (int): size of the latent space
    """
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.fc1 = nn.Linear(latent_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 16)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the discriminator
        Args:
            x (torch.Tensor): input tensor
        Returns:
            torch.Tensor: prediction (0.0 to 1.0)
        """
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        return self.sigmoid(self.fc4(h3))


def reparameterize(mu, logvar):
    """
    Reparameterization trick for sampling VAE latent space
    Args:
        mu (torch.Tensor): tensor of mu values
        logvar (torch.Tensor): tensor of logvar values
    Returns:
        torch.Tensor: tensor of sampled values
    """

    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)
