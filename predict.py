from src.gru.generator import EncoderDecoder
from src.gru.dataset import PredictionDataset
from src.utils.vectorizer import SELFIESVectorizer
from torch.utils.data import DataLoader
import rdkit.Chem.Draw as Draw
from src.pred.filter import molecule_filter
import multiprocessing as mp
import selfies as sf
import os
import torch
import pandas as pd
import configparser
import argparse
from tqdm import tqdm
import time


def predict(file_name, is_verbose=True):
    """
    Predicting molecules using the trained model.

    Model parameters are loaded from pred_config.ini file.
    The script scans the results folder for parquet files generated earlier using generate.py.
    For each parquet file, the script generates predictions and saves them in a new directory.

    Returns: None
    """

    # setup
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        '-c',
                        type=str,
                        default='pred_config.ini',
                        help='Path to config file')
    config_path = parser.parse_args().config

    name, _ = file_name.split('.')
    name += '_processed'

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
    use_cuda = config['SCRIPT'].getboolean('cuda')
    progress_bar = config['SCRIPT'].getboolean('progress_bar') if is_verbose else False
    device = 'cuda' if use_cuda else 'cpu'
    print(f'Using {device} device') if is_verbose else None

    # load model
    model = EncoderDecoder(
        fp_size=4860,
        encoding_size=encoding_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        teacher_ratio=0,
        use_cuda=use_cuda,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    print(f'Loaded model from {model_path}') if is_verbose else None

    # load data
    data = pd.read_parquet(f'results/{file_name}')

    # get predictions
    print(f'Getting predictions for file {file_name}...') if is_verbose else None
    predictions = get_predictions(model,
                                  data,
                                  fp_len=fp_len,
                                  batch_size=batch_size,
                                  progress_bar=progress_bar,
                                  use_cuda=use_cuda
                                  )

    # pred out non-druglike molecules
    print('Filtering out non-druglike molecules...') if is_verbose else None
    df = pd.DataFrame({'smiles': predictions, 'fps': data.fps})
    druglike_df = molecule_filter(df, config=config)
    output = druglike_df.drop(columns=['mols'])

    # save data as csv
    os.mkdir(f'results/{name}')

    with open(f'results/{name}/config.ini', 'w') as configfile:
        config.write(configfile)

    output.to_csv(f'results/{name}/{name}.csv', index=False)
    print(f'Saved data to results/{name}/{name}.csv') if is_verbose else None

    os.rename(f'results/{file_name}', f'results/{name}/{file_name}')

    # save images
    os.mkdir(f'results/{name}/imgs')
    for i, mol in enumerate(druglike_df.mols):
        Draw.MolToFile(mol, f'results/{name}/imgs/{i}.png', size=(300, 300))
    time_elapsed = time.time() - start_time
    print(f'{name} processed in {(time_elapsed / 60):.2f} minutes')


def get_predictions(model, df, fp_len=4860, batch_size=512, progress_bar=False, use_cuda=False):
    """
    Generates predictions for a given model and dataframe.
    Args:
        model (EncoderDecoder): Model to use for predictions.
        df  (pd.DataFrame): Dataframe containing fingerprints in 'fps' column.
        fp_len: Fingerprint length.
        batch_size (int): Batch size.
        progress_bar (bool): Whether to show progress bar.
        use_cuda (bool): Whether to use CUDA for predictions.
    Returns:
        list: List of predicted molecules in SMILES notation.
    """
    device = 'cuda' if use_cuda else 'cpu'
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
    return preds_smiles


if __name__ == '__main__':
    """
    Multiprocessing support and queue handling
    """
    cpus = mp.cpu_count()
    print("Number of cpus: ", cpus)
    queue = []

    # get list of files and dirs in results folder
    if os.path.isdir('results'):
        dir_list = os.listdir('results')
        queue = [name for name in dir_list if name.split('.')[-1] == 'parquet']
    else:
        print('No data files found in results directory')

    if len(queue) > cpus:
        queue = queue[:cpus]
    # launch a process for each file in queue
    procs = []
    verbose = True
    for i, name in enumerate(queue):
        print(f'Processing file {name}')
        if i == 0:
            verbose = True
        else:
            verbose = False
        proc = mp.Process(target=predict, args=(name, verbose))
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()
