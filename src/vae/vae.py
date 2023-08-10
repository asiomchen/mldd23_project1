import torch
import torch.nn as nn


class VAEEncoder(nn.Module):
    """
    Encoder for VAE
    Args:
        input_size (int): size of input
        output_size (int): size of latent space
    """
    def __init__(self, input_size, output_size):
        super(VAEEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc41 = nn.Linear(512, output_size)
        self.fc42 = nn.Linear(512, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input tensor of shape (batch_size, input_size)
        Returns:
            mu (torch.Tensor): mean of latent space
            logvar (torch.Tensor): log variance of latent space
        """
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        mu = self.fc41(h3)
        logvar = self.fc42(h3)
        return mu, logvar


class VAEDecoder(nn.Module):
    """
    Decoder for VAE
    Args:
        input_size (int): size of latent space
        output_size (int): size of output
    """
    def __init__(self, input_size, output_size):
        super(VAEDecoder, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input tensor of shape (batch_size, input_size)
        Returns:
            out (torch.Tensor): reconstructed x
        """
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        out = self.fc4(h3)
        return self.sigmoid(out)


class VAE(nn.Module):
    """
    VAE
    Args:
        input_size (int): size of input
        latent_size (int): size of latent space
    """
    def __init__(self, input_size, latent_size):
        super(VAE, self).__init__()
        self.encoder = VAEEncoder(input_size, latent_size)
        self.decoder = VAEDecoder(latent_size, input_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input tensor of shape (batch_size, input_size)
        Returns:
            out (torch.Tensor): reconstructed x
            mu (torch.Tensor): mean of latent space
            logvar (torch.Tensor): log variance of latent space
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


class VAELoss(nn.Module):
    """
    Calculates reconstruction loss and KL divergence loss
    """
    def __init__(self):
        super(VAELoss, self).__init__()

    def forward(self, recon_x, x, mu, logvar):
        """
        Args:
            recon_x (torch.Tensor): reconstructed x
            x (torch.Tensor): original x
            mu (torch.Tensor): latent space mu
            logvar (torch.Tensor): latent space log variance
        Returns:
            BCE (torch.Tensor): binary cross entropy loss (VAE recon loss)
            KLD (torch.Tensor): KL divergence loss
        """
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE, KLD
