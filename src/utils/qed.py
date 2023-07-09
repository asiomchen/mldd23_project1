import torch
import rdkit.Chem as Chem
from rdkit.Chem.QED import qed
import selfies as sf

def mean_batch_QED(batch)
    total_QED = 0
    batch_size = batch.shape[0]
    for seq in batch:
        selfies = vectorizer.devectorize(seq)
        smiles = sf.decoder(selfies)
        m = Chem.MolFromSmiles(smiles)
        total_QED += qed(m)
    mean_QED = total_QED/batch_size
    return mean_QED