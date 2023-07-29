import torch
import numpy as np
from torch.utils.data import Dataset

class PredictionDataset(Dataset):
    def __init__(self, df, fp_len):
        """
        Dataset class for handling GRU evaluation data.
        Args:
            df (pd.DataFrame): dataframe containing  fingerprints,
                               must contain ['fps'] column
        """
        self.fps = df['fps']
        self.fps = self.prepare_X(self.fps)
        self.fp_len = fp_len

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, idx):
        """
        Get item from dataset.
        Args:
            idx (int): index of item to get
        Returns:
            X (torch.Tensor): reconstructed fingerprint
        """
        raw_X = self.fps[idx]
        X = np.array(raw_X, dtype=int)
        X_reconstructed = self.reconstruct_fp(X)
        return torch.from_numpy(X_reconstructed).float()

    def reconstruct_fp(self, fp):
        fp_rec = np.zeros(self.fp_len)
        fp_rec[fp] = 1
        return fp_rec

    @staticmethod
    def prepare_X(fps):
        fps = fps.apply(eval).apply(lambda x: np.array(x, dtype=int))
        return fps.values