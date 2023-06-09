from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
import selfies as sf
import re

class SELFIESVectorizer:
    
    def __init__(self, pad_to_len=None):
        self.alphabet = self.read_alphabet()
        self.char2idx = {s: i for i, s in enumerate(self.alphabet)}
        self.idx2char = {i: s for i, s in enumerate(self.alphabet)}
        self.pad_to_len = pad_to_len
    
    def vectorize(self, selfie, no_special=False):
        
        # Vectorize a list of SELFIES strings to a numpy array of shape (len(smiles), embed, len(charset))
        
        if no_special:
            splited = self.split_selfi(selfie)
        elif self.pad_to_len is None:
            splited = ['[start]'] + self.split_selfi(selfie) + ['[end]']
        else:
            splited = ['[start]'] + self.split_selfi(selfie) + ['[end]'] + ['[nop]'] * (self.pad_to_len - len(self.split_selfi(selfie)) - 2)
        X = np.zeros((len(splited), len(self.alphabet)))
        for i in range(len(splited)):
            X[i, self.char2idx[splited[i]]] = 1
        return X
    
    def devectorize(self, ohe, remove_special=False):
        
        #D evectorize a numpy array of shape (len(smiles), len(charset)) to a list of SELFIES strings
        
        selfie_str = ''
        for j in range(ohe.shape[0]):
            idx = np.argmax(ohe[j, :])
            if remove_special and (self.idx2char[idx] == '[start]' or self.idx2char[idx] == '[end]'):
                continue
            selfie_str += self.idx2char[idx]
        return selfie_str
    
    def idxize(self, selfie, no_special=False):
        if no_special:
            splited = self.split_selfi(selfie)
        else:
            splited = ['[start]'] + self.split_selfi(selfie) + ['[end]'] + ['[nop]'] * (self.pad_to_len - len(self.split_selfi(selfie)) - 2)
        return np.array([self.char2idx[s] for s in splited])
    
    def deidxize(self, idx, no_special=False):
        if no_special:
            selfie = []
            for i in idx:
                char = self.idx2char[i]
                if char not in ['[end]', '[nop]', '[start]']:
                    selfie.append(char)
            return "".join(selfie)
        else:    
            return "".join([self.idx2char[i] for i in idx])
    
    def split_selfi(self, selfie):
        pattern = r'(\[[^\[\]]*\])'
        return re.findall(pattern, selfie)
        
        # Read alphabet of permitted SELFIES tokens from file
        
    def read_alphabet(self):
        alphabet = pd.read_csv('data/GRU_data/alphabet.txt', header=None).values.flatten()
        return alphabet
