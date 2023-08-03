from torch.utils.data import Dataset
from src.disc.discriminator import reparameterize
import pandas as pd
import torch


class DiscDataset(Dataset):
    """
    Dataset for the discriminator model
    Args:
        mu_path (str): path to the mu parquet file containing the mu values and activity labels in 'label' column
        logvar_path (str): path to the logvar parquet file
    """

    def __init__(self, mu_path, logvar_path):
        super().__init__()
        self.mu, self.activity = self.load_mu_n_labels(mu_path)
        self.logvar = self.load_logvar(logvar_path)

    def __getitem__(self, idx):
        encoding = reparameterize(self.mu[idx], self.logvar[idx])
        activity = self.activity[idx].float()
        return encoding, activity

    def __len__(self):
        return len(self.mu)

    @staticmethod
    def load_mu_n_labels(path):
        df = pd.read_parquet(path)
        labels = torch.tensor(df.label.to_numpy())
        df = df.drop(columns=['label'])
        tensor = torch.tensor(df.to_numpy())
        return tensor, labels

    @staticmethod
    def load_logvar(path):
        df = pd.read_parquet(path)
        tensor = torch.tensor(df.to_numpy())
        return tensor
