import pandas as pd
import numpy as np
import re

class SELFIESVectorizer:
    def __init__(self, pad_to_len=None):
        """
        SELFIES vectorizer
        Args:
            pad_to_len (int):size of the padding
        """
        self.alphabet = self.read_alphabet()
        self.char2idx = {s: i for i, s in enumerate(self.alphabet)}
        self.idx2char = {i: s for i, s in enumerate(self.alphabet)}
        self.pad_to_len = pad_to_len
    
    def vectorize(self, selfie, no_special=False):
        """
        Vectorize a list of SELFIES strings to a numpy array of shape (len(selfies), len(charset))
        Args:
            selfie (string):list of SELFIES strings
            no_special (bool):remove special tokens
        Returns:
            X (numpy.ndarray): vectorized SELFIES strings
        """
        if no_special:
            splited = self.split_selfi(selfie)
        elif self.pad_to_len is None:
            splited = ['[start]'] + self.split_selfi(selfie) + ['[end]']
        else:
            splited = ['[start]'] + self.split_selfi(selfie) + ['[end]'] \
                      + ['[nop]'] * (self.pad_to_len - len(self.split_selfi(selfie)) - 2)
        X = np.zeros((len(splited), len(self.alphabet)))
        for i in range(len(splited)):
            X[i, self.char2idx[splited[i]]] = 1
        return X
    
    def devectorize(self, ohe, remove_special=False):
        """
        Devectorize a numpy array of shape (len(selfies), len(charset)) to a list of SELFIES strings
        Args:
            ohe (numpy.ndarray): one-hot encoded sequence as numpy array
            remove_special (bool): remove special tokens
        Returns:
            selfie_str (string): SELFIES string
        """
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
            splited = ['[start]'] + self.split_selfi(selfie) + ['[end]'] \
                      + ['[nop]'] * (self.pad_to_len - len(self.split_selfi(selfie)) - 2)
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

    @staticmethod
    def split_selfi(selfie):
        pattern = r'(\[[^\[\]]*\])'
        return re.findall(pattern, selfie)
        
        # Read alphabet of permitted SELFIES tokens from file

    @staticmethod
    def read_alphabet():
        alphabet = pd.read_csv('data/alphabet.txt', header=None).values.flatten()
        return alphabet
