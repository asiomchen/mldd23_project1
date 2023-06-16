import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class GRUDataset(Dataset):
    def __init__(self, smiles_fp, selfies, vectorizer):
        self.smiles_fp = pd.read_parquet(smiles_fp)
        self.selfies = pd.read_parquet(selfies)
        self.selfies = self.prepare_y(self.selfies)
        self.vectorizer = vectorizer
    def __len__(self):
        return len(self.smiles_fp)
    def __getitem__(self, idx):
        raw_selfie = self.selfies[idx][0]
        vectorized_selfie = self.vectorizer.vectorize(raw_selfie)
        raw_X = self.smiles_fp.fps[idx]
        X = np.array(raw_X, dtype=int)
        X_reconstructed = self.reconstruct_fp(X)
        return torch.from_numpy(X_reconstructed).float(), torch.from_numpy(vectorized_selfie).float()

    @staticmethod
    def prepare_X(smiles_fp):
        fps = smiles_fp.fps.apply(eval).apply(lambda x: np.array(x, dtype=int))
        return fps
    @staticmethod
    def prepare_y(selfies):
        return selfies.values

    @staticmethod
    def reconstruct_fp(fp, length=4860):
        fp_rec = np.zeros(length)
        fp_rec[fp] = 1
        return fp_rec