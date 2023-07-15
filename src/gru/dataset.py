# Dataset class for handling GRU data

import torch
import selfies as sf
import numpy as np
from torch.utils.data import Dataset
import rdkit.Chem as Chem

class GRUDataset(Dataset):
    def __init__(self, df, vectorizer):
        """
        Dataset class for handling GRU training data.
        Args:
            df (pd.DataFrame): dataframe containing SMILES and fingerprints,
                               SMILES must be contained in ['smiles'] column as strings,
                               fingerprints must be contained in ['fps'] column as lists
                               of integers (dense vectors).
            vectorizer: SELFIES vectorizer instantiated from vectorizer.py
        """
        self.smiles = df['smiles']
        self.fps = df['fps']
        self.fps = self.prepare_X(self.fps)
        self.smiles = self.prepare_y(self.smiles)
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, idx):
        """
        Get item from dataset.
        Args:
            idx (int): index of item to get
        Returns:
            X (torch.Tensor): reconstructed fingerprint
            y (torch.Tensor): vectorized SELFIES
        """
        try:
            raw_smile = self.smiles[idx]
            randomized_smile = self.randomize_smiles(raw_smile)
            raw_selfie = sf.encoder(randomized_smile, strict=False)
            vectorized_selfie = self.vectorizer.vectorize(raw_selfie)
        except:
            print('Exception occured when vectorizing selfie of idx', idx)
            raw_smile = self.smiles[idx]
            raw_selfie = sf.encoder(raw_smile, strict=False)
            vectorized_selfie = self.vectorizer.vectorize(raw_selfie)
        raw_X = self.fps[idx]
        X = np.array(raw_X, dtype=int)
        X_reconstructed = self.reconstruct_fp(X)
        return torch.from_numpy(X_reconstructed).float(), torch.from_numpy(vectorized_selfie).float()

    def randomize_smiles(self, smiles):
        """
        Randomize SMILES string.
        Args:
            smiles: SMILES string
        Returns:
            randomized SMILES string
        """
        m = Chem.MolFromSmiles(smiles)
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m, ans)
        return Chem.MolToSmiles(nm, canonical=True, isomericSmiles=False)

    @staticmethod
    def prepare_X(fps):
        fps = fps.apply(eval).apply(lambda x: np.array(x, dtype=int))
        return fps.values

    @staticmethod
    def prepare_y(selfies):
        return selfies.values

    @staticmethod
    def reconstruct_fp(fp, length=4860):
        fp_rec = np.zeros(length)
        fp_rec[fp] = 1
        return fp_rec

class PredictionDataset(Dataset):
    def __init__(self, df, vectorizer):
        """
        Dataset class for handling GRU evaluation data.
        Args:
            df (pd.DataFrame): dataframe containing  fingerprints,
                               must contain ['fps'] column
            vectorizer: SELFIES vectorizer instantiated from vectorizer.py
        """
        self.fps = df['fps']
        self.fps = self.prepare_X(self.fps)

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

    @staticmethod
    def prepare_X(fps):
        fps = fps.apply(eval).apply(lambda x: np.array(x, dtype=int))
        return fps.values

    @staticmethod
    def reconstruct_fp(fp, length=4860):
        fp_rec = np.zeros(length)
        fp_rec[fp] = 1
        return fp_rec