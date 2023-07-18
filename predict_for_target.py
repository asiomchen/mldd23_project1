from src.gru.generator import EncoderDecoder
from src.gru.dataset import PredictionDataset
from src.utils.vectorizer import SELFIESVectorizer
import selfies as sf
from torch.utils.data import DataLoader
import rdkit.Chem as Chem
import rdkit.Chem.Draw as Draw
from rdkit.Chem import QED
import os
import torch
import pandas as pd
import argparse
from src.utils.data import closest_in_train
import configparser


def main():
    vectorizer = SELFIESVectorizer(pad_to_len=128)
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--Target", type=str, help="Target name (5ht1a, 5ht7, beta2, d2, h1)", default='5ht1a')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read('pred_config.ini')
    batch_size = int(config['PRED']['batch_size'])
    encoding_size = int(config['PRED']['encoding_size'])
    hidden_size = int(config['PRED']['hidden_size'])
    num_layers = int(config['PRED']['num_layers'])
    dropout = float(config['PRED']['dropout'])
    model_path = str(config['PRED']['model_path'])

    name = args.Target

    model = EncoderDecoder(
        fp_size=4860,
        encoding_size=encoding_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        teacher_ratio=0).to(device)

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    print(f'Loaded model from {model_path}')

    data_path = f'data/pred_data/{name}_fp.parquet'
    df = pd.read_parquet(data_path)
    print('Getting predictions...')
    predictions = get_predictions(model, df, vectorizer, batch_size=batch_size)
    print('Filtering out non-druglike molecules...')
    predictions_druglike, qeds, fps = filter_out_nondruglike(predictions, 0.7, df)

    print(f'Calculating tanimoto scores for {len(predictions_druglike)} molecules...')
    tanimoto_scores = []
    for mol in predictions_druglike:
        tanimoto = closest_in_train(mol)
        tanimoto_scores.append(tanimoto)

    if not os.path.isdir(f'imgs/{name}'):
        os.mkdir(f'imgs/{name}')

    # save data as csv
    data_to_save = pd.DataFrame({'smiles': [Chem.MolToSmiles(m) for m in predictions_druglike],
                                 'fp': fps,
                                 'qed': qeds,
                                 'tanimoto': tanimoto_scores
                                 })
    data_to_save.to_csv(f'imgs/{name}/{name}.csv', index=False)
    print(f'Saved data to imgs/{name}/{name}.csv')

    i = 0
    while i < 1000:
        mol4 = predictions_druglike[i:(i + 4)]
        qed4 = qeds[i:(i + 4)]
        qed4 = ['{:.2f}'.format(x) for x in qed4]
        tan4 = tanimoto_scores[i:(i + 4)]
        tan4 = ['{:.2f}'.format(x) for x in tan4]
        img = Draw.MolsToGridImage(mol4, molsPerRow=2, subImgSize=(400, 400),
                                   legends=[f'QED: {qed}, Tan: {tan}' for qed, tan in zip(qed4, tan4)],
                                   returnPNG=False
                                   )
        img.save(f'imgs/{name}/{i}.png')
        i += 4
    print(f'Saved images to imgs/{name}/')


def get_predictions(model, df, vectorizer, batch_size=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = PredictionDataset(df, vectorizer)
    loader = DataLoader(dataset, shuffle=False, batch_size=batch_size, drop_last=True)
    preds_smiles = []
    with torch.no_grad():
        model.eval()
        for X in loader:
            X = X.to(device)
            preds = model(X, None, teacher_forcing=False)
            preds = preds.detach().cpu().numpy()

            # sf.set_semantic_constraints("hypervalent")
            for seq in preds:
                selfie = vectorizer.devectorize(seq, remove_special=True)
                try:
                    preds_smiles.append(sf.decoder(selfie))
                except:
                    preds_smiles.append('C')
    return preds_smiles


def filter_out_nondruglike(predictions, threshold, df):
    raw_mols = [Chem.MolFromSmiles(s) for s in predictions]
    raw_qeds = [QED.qed(m) for m in raw_mols]
    raw_fps = df['fps'].tolist()
    filtered_mols = []
    filtered_qeds = []
    filtered_fps = []
    for i, value in enumerate(raw_qeds):
        mol = raw_mols[i]
        fp = raw_fps[i]
        ri = mol.GetRingInfo()
        largest_ring_size = max((len(r) for r in ri.AtomRings()), default=0)
        if value > threshold and largest_ring_size < 8:
            filtered_mols.append(mol)
            filtered_qeds.append(value)
            filtered_fps.append(fp)
    return filtered_mols, filtered_qeds, filtered_fps


if __name__ == '__main__':
    main()
