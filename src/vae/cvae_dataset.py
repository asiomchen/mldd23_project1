# Dataset class for handling cVAE training data

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class VAEDataset(Dataset):
    def __init__(self, data_path):
        self.fps = pd.read_parquet(data_path, columns=['fps'])
        self.activity = pd.read_parquet(data_path, columns=['Class'])
    def __len__(self):
        return len(self.fps)
    def __getitem__(self, idx):
        activity = self.activity.iloc[idx].values[0]
        activity = np.array(activity)
        raw_X = self.fps.iloc[idx]
        X_prepared = self.prepare_X(raw_X).values[0]
        X = np.array(X_prepared, dtype=int)
        X_reconstructed = self.reconstruct_fp(X)
        return torch.from_numpy(X_reconstructed).float(), torch.from_numpy(activity).float()

    @staticmethod
    def prepare_X(fps):
        fps = fps.apply(eval).apply(lambda x: np.array(x, dtype=int))
        return fps
    
    @staticmethod
    def prepare_y(activity):
        return np.array(activity.values)

    @staticmethod
    def reconstruct_fp(fp, length=4860):
        fp_rec = np.zeros(length)
        fp_rec[fp] = 1
        return fp_rec
