import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from src.pred.filter import molecule_score
from src.utils.vectorizer import SELFIESVectorizer
import selfies as sf


def predict_with_dropout(model,
                         latent_vectors: np.array,
                         n_iter: int = 100,
                         device: str = 'cuda'):
    """
    Generate molecules from latent vectors with dropout.
    Args:
        model (torch.nn.Module): EncoderDecoderv3 model.
        latent_vectors (np.array): numpy array of latent vectors. Shape = (n_samples, latent_size).
        n_iter (int): number of iterations.
        device: device to use for prediction. Can be 'cpu' or 'cuda'.

    Returns:
        pd.DataFrame: Dataframe containing smiles and scores.
    """
    vectorizer = SELFIESVectorizer(pad_to_len=128)
    device = torch.device(device)
    dataframes = []
    with torch.no_grad():
        for n in range(n_iter):
            df = pd.DataFrame(columns=['smiles', 'score'])
            latent_tensor = torch.Tensor(latent_vectors).to(device)
            model = model.to(device)
            preds, _ = model(latent_tensor, None, omit_encoder=True)
            preds = preds.detach().cpu().numpy()
            preds = [vectorizer.devectorize(pred, remove_special=True) for pred in preds]
            smiles = [sf.decoder(x) for x in preds]
            mols = [Chem.MolFromSmiles(x) for x in smiles]
            scores = [molecule_score(mol) for mol in mols]
            df['smiles'] = smiles
            df['molecule_score'] = scores
            dataframes.append(df)

    best_smiles = []
    best_scores = []
    for n in range(len(latent_vectors)):
        scores = np.array([df['molecule_score'][n] for df in dataframes])
        best_idx = np.argmax(scores)
        best_smile = dataframes[best_idx]['smiles'][n]
        best_score = dataframes[best_idx]['molecule_score'][n]
        best_smiles.append(best_smile)
        best_scores.append(best_score)

    best_results = pd.DataFrame(columns=['smiles', 'molecule_score'])
    best_results['smiles'] = best_smiles

    return best_results
