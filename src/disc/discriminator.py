import torch.nn as nn
import torch
from sklearn.ensemble import RandomForestClassifier

class Discriminator(nn.Module):
    """
    Discriminator model for searching VAE latent space
    Args:
        latent_size (int): size of the latent space
    """
    def __init__(self, latent_size, use_sigmoid=True):
        super().__init__()
        self.latent_size = latent_size
        self.fc1 = nn.Linear(latent_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 16)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.use_sigmoid = use_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the discriminator
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


class RandomForestWrapper:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def forward(self, X):
        y = self.model.predict_proba(X)[0]
        return y 