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
    """
    Main function for predicting molecules from a trained model.

    Model parameters are loaded from pred_config.ini file.
    The script scans the results folder for parquet generated earlier using the generate.py script.
    For each parquet file, the script generates predictions and saves them in a new directory.

    Returns: None
    """
    vectorizer = SELFIESVectorizer(pad_to_len=128)
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--Target", type=str, help="Target name (5ht1a, 5ht7, beta2, d2, h1)", default='5ht1a')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read('pred_config.ini')
    QED_threshold = float(config['FILTER']['QED_threshold'])
    max_ring_size = int(config['FILTER']['max_ring_size'])
    batch_size = int(config['MODEL']['batch_size'])
    fp_len = int(config['MODEL']['fp_len'])
    encoding_size = int(config['MODEL']['encoding_size'])
    hidden_size = int(config['MODEL']['hidden_size'])
    num_layers = int(config['MODEL']['num_layers'])
    dropout = float(config['MODEL']['dropout'])
    model_path = str(config['MODEL']['model_path'])
    target = args.Target

    model = EncoderDecoder(
        fp_size=4860,
        encoding_size=encoding_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        teacher_ratio=0).to(device)

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    print(f'Loaded model from {model_path}')

    files = os.listdir('results')

    for file in files:
        f_name, f_type = file.split('.')
        already_processed = os.path.isdir(f'results/{f_name}')
        if f_type != 'parquet' or already_processed:
            continue
        os.mkdir(f'results/{f_name}')
        with open(f'results/{f_name}/config.ini', 'w') as configfile:
            config.write(configfile)

        df = pd.read_parquet(f'results/{file}')

        print(f'Getting predictions for file {f_name}...')
        predictions = get_predictions(model, df, vectorizer, fp_len=fp_len, batch_size=batch_size)
        print('Filtering out non-druglike molecules...')
        predictions_druglike, qeds, fps = filter_out_nondruglike(predictions, QED_threshold, max_ring_size, df)
        tanimoto_scores = [closest_in_train(mol) for mol in predictions_druglike]

        # save data as csv
        data_to_save = pd.DataFrame({'smiles': [Chem.MolToSmiles(m) for m in predictions_druglike],
                                     'fp': fps,
                                     'qed': qeds,
                                     'tanimoto': tanimoto_scores
                                     })
        data_to_save.to_csv(f'results/{f_name}/{target}.csv', index=False)
        print(f'Saved data to results/{f_name}/{target}.csv')

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
            img.save(f'results/{f_name}/imgs/{i}.png')
            i += 4
        print(f'results/{f_name}/imgs/{i}.png')


def get_predictions(model, df, vectorizer, fp_len, batch_size=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = PredictionDataset(df, fp_len=fp_len)
    loader = DataLoader(dataset, shuffle=False, batch_size=batch_size, drop_last=True)
    preds_smiles = []
    with torch.no_grad():
        model.eval()
        for X in loader:
            X = X.to(device)
            preds = model(X, None, teacher_forcing=False)
            preds = preds.detach().cpu().numpy()
            for seq in preds:
                selfie = vectorizer.devectorize(seq, remove_special=True)
                try:
                    preds_smiles.append(sf.decoder(selfie))
                except sf.DecoderError:
                    sf.set_semantic_constraints("hypervalent")
                    preds_smiles.append(sf.decoder(selfie))
                    sf.set_semantic_constraints()
    return preds_smiles


def filter_out_nondruglike(predictions, threshold, max_ring_size, df):
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
        if value > threshold and largest_ring_size <= max_ring_size:
            filtered_mols.append(mol)
            filtered_qeds.append(value)
            filtered_fps.append(fp)
    return filtered_mols, filtered_qeds, filtered_fps


if __name__ == '__main__':
    main()
