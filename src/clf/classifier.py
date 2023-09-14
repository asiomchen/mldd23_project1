import torch.nn as nn
import torch

class NLPClassifier(nn.Module):
    """
    Classifier model for searching VAE latent space
    Args:
        latent_size (int): size of the latent space
    """
    def __init__(self, latent_size, use_sigmoid=True, fc1_size=256, fc2_size=256):
        super().__init__()
        self.latent_size = latent_size
        self.fc1 = nn.Linear(latent_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, 16)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.use_sigmoid = use_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the classifier
        Args:
            x (torch.Tensor): input tensor of size (latent space)
        Returns:
            torch.Tensor: prediction (0.0 to 1.0) if use_sigmoid=True, else logits
        """
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        if self.use_sigmoid:
            return self.sigmoid(self.fc4(h3))
        else:
            return self.fc4(h3)

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
