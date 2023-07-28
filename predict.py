from src.gru.generator import EncoderDecoder
from src.gru.dataset import PredictionDataset
from src.utils.vectorizer import SELFIESVectorizer
import selfies as sf
from torch.utils.data import DataLoader
import rdkit.Chem as Chem
import rdkit.Chem.Draw as Draw
from src.pred.filter import molecule_filter
import os
import torch
import pandas as pd
import configparser
import argparse
from tqdm import tqdm
from time import time


def main():
    """
    Predicting molecules using the trained model.

    Model parameters are loaded from pred_config.ini file.
    The script scans the results folder for parquet files generated earlier using generate.py.
    For each parquet file, the script generates predictions and saves them in a new directory.

    Returns: None
    """

    # setup
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='pred_config.ini', help='Path to config file')
    config_path = parser.parse_args().config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    # load config
    config = configparser.ConfigParser()
    config.read(config_path)
    batch_size = int(config['MODEL']['batch_size'])
    fp_len = int(config['MODEL']['fp_len'])
    encoding_size = int(config['MODEL']['encoding_size'])
    hidden_size = int(config['MODEL']['hidden_size'])
    num_layers = int(config['MODEL']['num_layers'])
    dropout = float(config['MODEL']['dropout'])
    model_path = str(config['MODEL']['model_path'])
    progress_bar = config['SCRIPT'].getboolean('progress_bar')

    # load model
    model = EncoderDecoder(
        fp_size=4860,
        encoding_size=encoding_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        teacher_ratio=0
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    print(f'Loaded model from {model_path}')

    # get list of files and dirs in results folder
    if os.path.isdir('results') and os.listdir('results') is not None:
        dir_list = os.listdir('results')
        files = [name for name in dir_list if name.split('.')[-1] == 'parquet']
    else:
        files = []

    # iterate over parquet files
    for name in files:
        start_time = time()
        f_name, f_type = name.split('.')
        f_name += '_done'

        # load data
        data = pd.read_parquet(f'results/{name}')

        # get predictions
        print(f'Getting predictions for file {name}...')
        predictions = get_predictions(model,
                                      data,
                                      fp_len=fp_len,
                                      batch_size=batch_size,
                                      progress_bar=progress_bar
                                      )

        # pred out non-druglike molecules
        print('Filtering out non-druglike molecules...')
        df = pd.DataFrame({'smiles': predictions})
        df['mols'] = df.smiles.apply(Chem.MolFromSmiles)
        df['raw_fps'] = data.fps

        druglike_df = molecule_filter(df, config=config)
        output = druglike_df.drop(columns=['mols'])

        # save data as csv
        os.mkdir(f'results/{f_name}')

        with open(f'results/{f_name}/config.ini', 'w') as configfile:
            config.write(configfile)

        output.to_csv(f'results/{f_name}/{f_name}.csv', index=False)
        print(f'Saved data to results/{f_name}.csv')

        os.rename(f'results/{name}', f'results/{f_name}/{name}')

        # save images
        os.mkdir(f'results/{f_name}/imgs')
        for i, mol in enumerate(druglike_df.mols):
            Draw.MolToFile(mol, f'results/{f_name}/imgs/{i}.png', size=(300, 300))
        time_elapsed = time() - start_time
        print(f'Finished in {(time_elapsed / 60):.2f} minutes')


def get_predictions(model, df, fp_len=4860, batch_size=512, progress_bar=False):
    """
    Generates predictions for a given model and dataframe.
    Args:
        model (EncoderDecoder): Model to use for predictions.
        df  (pd.DataFrame): Dataframe containing fingerprints in 'fps' column.
        fp_len: Fingerprint length.
        batch_size (int): Batch size.
        progress_bar (bool): Whether to show progress bar.
    Returns:
        list: List of predicted molecules in SMILES notation.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = PredictionDataset(df, fp_len=fp_len)
    vectorizer = SELFIESVectorizer(pad_to_len=128)
    loader = DataLoader(dataset, shuffle=False, batch_size=batch_size, drop_last=True)
    preds_smiles = []
    with torch.no_grad():
        for X in tqdm(loader, disable=not progress_bar):
            X = X.to(device)
            preds = model(X, None, teacher_forcing=False)
            preds = preds.detach().cpu().numpy()
            for seq in preds:
                selfie = vectorizer.devectorize(seq, remove_special=True)
                try:
                    preds_smiles.append(sf.decoder(selfie))
                except sf.DecoderError:
                    preds_smiles.append('C')  # dummy SMILES
                    print('DecoderError raised, appending dummy SMILES')
    return preds_smiles

if __name__ == '__main__':
    main()
