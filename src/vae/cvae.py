import torch
import torch.nn as nn


class CVAEEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CVAEEncoder, self).__init__()
        input_size = input_size + 1  # add one for activity label
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc41 = nn.Linear(128, hidden_size)
        self.fc42 = nn.Linear(128, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        y = y.reshape(-1, 1)  # reshape to (batch_size, 1)
        x = torch.cat((x, y), dim=1)
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        mu = self.fc41(h3)
        logvar = self.fc42(h3)
        return mu, logvar


class CVAEDecoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(CVAEDecoder, self).__init__()
        input_size = hidden_size + 1  # add one for the label
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 1024)
        self.fc4 = nn.Linear(1024, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        y = y.reshape(-1, 1)  # reshape to (batch_size, 1), unsqueeze should also work
        x = torch.cat((x, y), dim=1)
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        out = self.fc4(h3)
        return self.sigmoid(out)


class CVAE(nn.Module):
    def __init__(self, fp_size, latent_size):
        super(CVAE, self).__init__()
        self.encoder = CVAEEncoder(fp_size, latent_size)
        self.decoder = CVAEDecoder(latent_size, fp_size)

    def forward(self, x, y):
        mu, logvar = self.encoder(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, y), mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)


class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()

    def forward(self, recon_x, x, mu, logvar):
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD


def sample_CVAE(model, n_samples, device, active=True):
    if active:
        y = 1
    else:
        y = 0
    z = torch.randn(n_samples, model.decoder.fc1.in_features - 1).to(device)
    y = torch.tensor([y] * n_samples).to(device)
    return model.decoder(z, y).detach().cpu().numpy()
