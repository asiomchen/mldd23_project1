import torch
import selfies as sf
import torch.nn as nn
import torch.utils.data as data_utils
import torch.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import rdkit.Chem as Chem


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
        

class GRUDataset_v2(Dataset):
    def __init__(self, data_path, vectorizer):
        self.smiles = pd.read_parquet(data_path)['SMILES']
        self.fps = pd.read_parquet(data_path)['fps']
        self.fps = self.prepare_X(self.fps)
        self.smiles = self.prepare_y(self.smiles)
        self.vectorizer = vectorizer
    def __len__(self):
        return len(self.fps)
    def __getitem__(self, idx):
        raw_smile = self.smiles[idx]
        print(raw_smile)
        randomized_smile = self.randomize_smiles(raw_smile)
        print(randomized_smile)
        raw_selfie = sf.encoder(randomized_smile)
        vectorized_selfie = self.vectorizer.vectorize(raw_selfie)
        raw_X = self.fps[idx]
        X = np.array(raw_X, dtype=int)
        X_reconstructed = self.reconstruct_fp(X)
        return torch.from_numpy(X_reconstructed).float(), torch.from_numpy(vectorized_selfie).float()

    def randomize_smiles(self, smiles):
        m = Chem.MolFromSmiles(smiles)
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m,ans)
        return Chem.MolToSmiles(nm, canonical=True, isomericSmiles=False)
    
    @staticmethod
    def prepare_X(fps):
        fps = fps.apply(eval).apply(lambda x: np.array(x, dtype=int))
        return fps
    @staticmethod
    def prepare_y(selfies):
        return selfies.values

    @staticmethod
    def reconstruct_fp(fp, length=4860):
        fp_rec = np.zeros(length)
        fp_rec[fp] = 1
        return fp_rec
