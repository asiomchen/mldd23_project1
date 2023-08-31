import argparse
import configparser
import multiprocessing as mp
import os
import queue
import time

import pandas as pd
import rdkit.Chem.Draw as Draw
import selfies as sf
import torch

from src.gru.generator import EncoderDecoderV3
from src.pred.filter import molecule_filter
from src.utils.vectorizer import SELFIESVectorizer


def predict(file_name, is_verbose=True):
    """
    Predicting molecules using the trained model.

    Model parameters are loaded from pred_config.ini file.
    The script scans the results folder for parquet files generated earlier using generate.py.
    For each parquet file, the script generates predictions and saves them in a new directory.

    Args:
        file_name (str): Name of the parquet file to process.
        is_verbose (bool): Whether to print progress.
    Returns: None
    """

    # setup
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        '-c',
                        type=str,
                        default='config_files/pred_config.ini',
                        help='Path to config file')
    config_path = parser.parse_args().config

    name, _ = file_name.split('.')
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    name += '_' + timestamp

    # load config
    config = configparser.ConfigParser()
    config.read(config_path)
    fp_len = int(config['MODEL']['fp_len'])
    encoding_size = int(config['MODEL']['encoding_size'])
    hidden_size = int(config['MODEL']['hidden_size'])
    num_layers = int(config['MODEL']['num_layers'])
    dropout = float(config['MODEL']['dropout'])
    model_path = str(config['MODEL']['model_path'])
    use_cuda = config['SCRIPT'].getboolean('cuda')
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device') if is_verbose else None

    # load model
    model = EncoderDecoderV3(
        fp_size=fp_len,
        encoding_size=encoding_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        teacher_ratio=0,
        use_cuda=use_cuda,
        output_size=42,
        random_seed=42
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    print(f'Loaded model from {model_path}') if is_verbose else None

    # load data
    query_df = pd.read_parquet(f'results/{file_name}').sample(10000) # TODO: remove sample
    input_tensor = torch.tensor(query_df.to_numpy(), dtype=torch.float32)

    # get predictions
    print(f'Getting predictions for file {file_name}...') if is_verbose else None
    df = get_predictions(model,
                         input_tensor,
                         use_cuda=use_cuda,
                         verbose=is_verbose
                         )

    # pred out non-druglike molecules
    print('Filtering out non-druglike molecules...') if is_verbose else None
    druglike_df = molecule_filter(df, config=config)

    if len(druglike_df) == 0:
        print('None of predicted molecules meets the filter criteria')
        return None

    # save data as csv
    os.mkdir(f'results/{name}')

    with open(f'results/{name}/config.ini', 'w') as configfile:
        config.write(configfile)

    druglike_df.to_csv(f'results/{name}/{name}.csv',
                       columns=['smiles', 'qed'],
                       index=False)

    print(f'Saved data to results/{name}/{name}.csv') if is_verbose else None

    # save images
    os.mkdir(f'results/{name}/imgs')
    for i, mol in enumerate(druglike_df.mols):
        Draw.MolToFile(mol, f'results/{name}/imgs/{i}.png', size=(300, 300))
    time_elapsed = time.time() - start_time
    print(f'{name} processed in {(time_elapsed / 60):.2f} minutes')


def get_predictions(model,
                    input_tensor,
                    use_cuda=False,
                    verbose=True,
                    ):
    """
    Generates predictions for a given model and dataframe.
    Args:
        model (nn.Module): Model to use for predictions.
        input_tensor (torch.Tensor) Tensor containing batched latent vectors.
        use_cuda (bool): Whether to use CUDA for predictions.
        verbose (bool): Whether to print progress.
    Returns:
        output (pd.DataFrame): prediction df containing 'smiles' and 'fp' columns
    """
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    vectorizer = SELFIESVectorizer(pad_to_len=128)
    preds_smiles = []
    with torch.no_grad():
        X = input_tensor.to(device)
        preds, _ = model(X, None, teacher_forcing=False, encode_first=False)
        preds = preds.cpu().numpy()
        for seq in preds:
            selfie = vectorizer.devectorize(seq, remove_special=True)
            try:
                preds_smiles.append(sf.decoder(selfie))
            except sf.DecoderError:
                preds_smiles.append('C')  # dummy SMILES
    print('Predictions complete') if verbose else None
    output = pd.DataFrame({'smiles': preds_smiles})
    return output


if __name__ == '__main__':
    """
    Multiprocessing support and queue handling
    """
    cpus = mp.cpu_count()
    print("Number of cpus: ", cpus)

    # get list of files and dirs in results folder
    if not os.path.isdir('results'):
        os.mkdir('results')

    dir_list = os.listdir('results')
    files = [name for name in dir_list if name.split('.')[-1] == 'parquet']
    if not files:
        print('No .parquet files found')

    # prepare a process for each file and add to queue
    queue = queue.Queue()
    verbose = True
    for i, name in enumerate(files):
        print(f'Processing file {name}')
        # make only the first process verbose
        if i == 0:
            verbose = True
        else:
            verbose = False
        proc = mp.Process(target=predict, args=(name, verbose))
        queue.put(proc)

    # handle the queue
    processes = []
    while True:
        if queue.empty():
            break
        if len(mp.active_children()) < cpus:
            proc = queue.get()
            proc.start()
            processes.append(proc)
        time.sleep(10)

    # complete the processes
    for proc in processes:
        proc.join()
