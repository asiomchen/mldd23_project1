# Dataset class for handling VAE training data

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class VAEDataset(Dataset):
    def __init__(self, data_path, fp_len):
        self.fps = pd.read_parquet(data_path, columns=['fps'])
        self.fp_len = fp_len

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, idx):
        raw_X = self.fps.iloc[idx]
        X_prepared = self.prepare_X(raw_X).values[0]
        X = np.array(X_prepared, dtype=int)
        X_reconstructed = self.reconstruct_fp(X, self.fp_len)
        return torch.from_numpy(X_reconstructed).float()

    def reconstruct_fp(self, fp):
        fp_rec = np.zeros(self.fp_len)
        fp_rec[fp] = 1
        return fp_rec

    @staticmethod
    def prepare_X(fps):
        fps = fps.apply(eval).apply(lambda x: np.array(x, dtype=int))
        return fps

    @staticmethod
    def prepare_y(activity):
        return np.array(activity.values)
