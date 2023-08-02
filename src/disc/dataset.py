from torch.utils.data import Dataset
import pandas as pd
import torch


class DiscrDataset(Dataset):
    def __init__(self, mu_path, logvar_path):
        super().__init__()
        self.mu, self.activity = self.load_mu_n_labels(mu_path)
        self.logvar = self.load_logvar(logvar_path)

    def __getitem__(self, idx):
        encoding = reparameterize(self.mu[idx], self.logvar[idx])
        activity = self.activity[idx]
        return encoding, activity

    def __len__(self):
        return len(self.mu)

    @staticmethod
    def load_mu_n_labels(path):
        df = pd.read_parquet(path)
        labels = df.label
        df = df.drop(columns=['label'])
        tensor = torch.tensor(df.to_numpy())
        return tensor, labels

    @staticmethod
    def load_logvar(path):
        df = pd.read_parquet(path)
        tensor = torch.tensor(df.to_numpy())
        return tensor


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)
