import torch
import rdkit.Chem as Chem
from rdkit.Chem.QED import qed
import selfies as sf

def mean_batch_QED(batch, vectorizer):
    total_QED = 0
    batch_size = batch.shape[0]
    for seq in batch:
        total_QED += QED(seq, vectorizer)
    mean_QED = total_QED/batch_size
    return mean_QED

def QED(seq, vectorizer):
    selfies = vectorizer.devectorize(seq, remove_special=True)
    try:    
        smiles = sf.decoder(selfies)
    except:
        smiles = 'C'
    m = Chem.MolFromSmiles(smiles)
    return qed(m)