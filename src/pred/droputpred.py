import numpy as np
import pandas as pd
from rdkit import Chem
from src.pred.filter import get_largest_ring
import torch
from src.utils.vectorizer import SELFIESVectorizer
import selfies as sf
from rdkit.Chem import Draw
import rdkit.Chem.QED as QED


def extended_QED(mol, penalty=4, max_ring_size=7):
    qed = QED.qed(mol)
    largest_ring = get_largest_ring(mol)
    if largest_ring > 7:
        qed *= 0.5 * np.exp((7 - largest_ring) / penalty)
    return qed


def predict_with_dropout(model,
                         latent_vectors: np.array,
                         n_iter: int = 100,
                         device: str = 'cuda',
                         return_imgs=True):
    """
    Generate molecules from latent vectors with dropout.
    Args:
        model: EncoderDecoderv3 model.
        latent_vectors: numpy array of latent vectors. Shape = (n_samples, latent_size).
        n_iter: number of iterations.
        device: device to use for prediction. Can be 'cpu' or 'cuda'.
        return_imgs: if True, returns grid image of the generated molecules.

    Returns:
        pd.DataFrame: Dataframe containing smiles and scores.
    """
    vectorizer = SELFIESVectorizer(pad_to_len=128)
    device = torch.device(device)
    dataframes = []
    for n in range(n_iter):
        df = pd.DataFrame(columns=['smiles', 'score'])
        latent_tensor = torch.Tensor(latent_vectors).to(device)
        model = model.to(device)
        preds, _ = model(latent_tensor, None, omit_encoder=True)
        preds = preds.detach().cpu().numpy()
        preds = [vectorizer.devectorize(pred, remove_special=True) for pred in preds]
        smiles = [sf.decoder(x) for x in preds]
        mols = [Chem.MolFromSmiles(x) for x in smiles]
        scores = [extended_QED(mol) for mol in mols]
        df['smiles'] = smiles
        df['score'] = scores
        dataframes.append(df)

    best_smiles = []
    best_scores = []
    for n in range(len(latent_vectors)):
        scores = np.array([df['score'][n] for df in dataframes])
        best_idx = np.argmax(scores)
        best_smile = dataframes[best_idx]['smiles'][n]
        best_score = dataframes[best_idx]['score'][n]
        best_smiles.append(best_smile)
        best_scores.append(best_score)

    best_results = pd.DataFrame(columns=['smiles', 'score'])
    best_results['smiles'] = best_smiles
    best_results['score'] = best_scores
    best_results['mol'] = best_results['smiles'].apply(Chem.MolFromSmiles)

    if return_imgs:
        img = Draw.MolsToGridImage(best_results.mol.to_list(), molsPerRow=3,
                                   subImgSize=(300, 300))
        return best_results, img
    else:
        return best_results
