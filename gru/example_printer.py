import torch
import numpy as np
import selfies as sf
from rdkit import Chem
import random

from vectorizer import SELFIESVectorizer, determine_alphabet

class ExamplePrinter():
    def __init__(self, data_path, test_loader, num_examples=3):
        self.y_path = data_path
        self.alphabet = determine_alphabet(self.y_path) 
        self.vectorizer = SELFIESVectorizer(self.alphabet, pad_to_len = 128)
        self.dataloader = test_loader
        self.num_examples = num_examples
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __call__(self, model):
        model.eval()
        x, y = next(iter(self.dataloader))
        x = x.to(self.device)
        preds = model(x, y, teacher_forcing=False)
        preds_indices = torch.argmax(preds.cpu(), dim=2)
        preds_indices = preds_indices.numpy()
        y_indices = torch.argmax(y.cpu(), dim=2)
        y_indices = y_indices.numpy()
        
        pred_selfies = []
        pred_smiles = []
        for molecule in preds_indices:
            selfie = self.vectorizer.deidxize(molecule, no_special=True)
            pred_selfies.append(selfie)
            pred_smiles.append(sf.decoder(selfie))
            
        y_selfies = []
        y_smiles = []
        for molecule in y_indices:
            selfie = self.vectorizer.deidxize(molecule, no_special=True)
            y_selfies.append(selfie)
            y_smiles.append(sf.decoder(selfie))
        
        for j in range(self.num_examples):
            print('Predicted SELFIE:')
            print(pred_selfies[j], '\n')
            print('True SELFIE:')
            print(y_selfies[j])
            print('-'*60)
        
        selected = random.sample(y_smiles, self.num_examples)
        
        return selected
            