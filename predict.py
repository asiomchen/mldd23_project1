import argparse
import configparser
import multiprocessing as mp
import os
import queue
import time

import pandas as pd
import rdkit.Chem.Draw as Draw
import rdkit.Chem as Chem
import selfies as sf
import torch

from src.generator.generator import EncoderDecoderV3
from src.pred.filter import molecule_filter
from src.utils.vectorizer import SELFIESVectorizer


def predict(file_path, is_verbose=True):
    """
    Predicting molecules using the trained model.

    Args:
        file_path (str): Path to the file containing latent vectors.
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

    # get file name
    dir_name, file_name = file_path.split('/')
    name, _ = file_name.split('.')
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    name = '_'.join([dir_name, name, timestamp])

    # load config
    config = configparser.ConfigParser()
    config.read(config_path)
    model_path = str(config['MODEL']['model_path'])
    use_cuda = config['SCRIPT'].getboolean('cuda')
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device') if is_verbose else None

    if model_path.lower() == 'auto':
        model_config_path = f'models/{dir_name}/hyperparameters.ini'
        config.read(model_config_path)

    # load model
    model = EncoderDecoderV3(
        fp_size=int(config['MODEL']['fp_len']),
        encoding_size=int(config['MODEL']['encoding_size']),
        hidden_size=int(config['MODEL']['hidden_size']),
        num_layers=int(config['MODEL']['num_layers']),
        dropout=0.0,
        teacher_ratio=0.0,
        use_cuda=use_cuda,
        output_size=42,
        random_seed=42,
        fc1_size=int(config['MODEL']['fc1_size']),
        fc2_size=int(config['MODEL']['fc2_size']),
        fc3_size=int(config['MODEL']['fc3_size'])
    ).to(device)

    if model_path.lower() == 'auto':
        info = configparser.ConfigParser()
        info.read(f'data/encoded_data/{dir_name}/info.ini')
        model_path = info['INFO']['model_path']
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    elif model_path.lower() != 'auto':
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        print(f'Loaded model from {model_path}') if is_verbose else None

    # load data
    query_df = pd.read_parquet(f'data/encoded_data/{dir_name}/{file_name}').sample(1000)
    target_smiles = query_df['smiles'].values
    for col in ['smiles', 'label']:
        if col in query_df.columns:
            query_df = query_df.drop(columns=[col])
    input_tensor = torch.tensor(query_df.to_numpy(), dtype=torch.float32)

    # get predictions
    print(f'Getting predictions for file {file_name}...') if is_verbose else None
    df = get_predictions(model,
                         input_tensor,
                         use_cuda=use_cuda,
                         verbose=is_verbose
                         )
    df['target_smiles'] = target_smiles

    # pred out non-druglike molecules
    print('Filtering out non-druglike molecules...') if is_verbose else None

    manager = mp.Manager()
    return_list = manager.list()

    cpus = mp.cpu_count()
    print("Number of cpus: ", cpus)

    q = queue.Queue()

    # prepare a process for each file and add to queue
    chunk_size = len(df) // cpus + 1 if len(df) % cpus != 0 else len(df) // cpus
    n_chunks = int(len(df) / chunk_size) + 1 if len(df) % chunk_size != 0 else int(len(df) / chunk_size)
    chunks = [df[i * chunk_size:(i + 1) * chunk_size] for i in range(n_chunks)]
    for chunk in chunks:
        proc = mp.Process(target=molecule_filter, args=(chunk, config, return_list))
        q.put(proc)

    # handle the queue
    processes = []
    while True:
        if q.empty():
            print("(mp) Queue handled successfully")
            break
        if len(mp.active_children()) < cpus:
            proc = q.get()
            print("(mp) Starting:", proc.name)
            proc.start()
            processes.append(proc)
        time.sleep(1)

    # complete the processes
    for proc in processes:
        proc.join()

    druglike_df = pd.concat(return_list)

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

    print(f'Saved data to results/{name}') if is_verbose else None

    # save images
    os.mkdir(f'results/{name}/imgs')

    if 'target_smiles' in druglike_df.columns:
        if 'tanimoto' in druglike_df.columns:
            for i, tuple in enumerate(zip(druglike_df.mols, druglike_df.target_smiles, druglike_df.tanimoto)):
                mol, target_smile, tanimoto = tuple
                target_mol = Chem.MolFromSmiles(target_smile)
                img = Draw.MolsToGridImage([mol, target_mol],
                                           legends=[f'pred', f'target\nclosest in train: {tanimoto}'],
                                           molsPerRow=2,
                                           subImgSize=(300, 300)
                                           )
                img.save(f'results/{name}/imgs/{i}.png')
        else:
            for i, tuple in enumerate(zip(druglike_df.mols, druglike_df.target_smiles)):
                mol, target_smile = tuple
                target_mol = Chem.MolFromSmiles(target_smile)
                img = Draw.MolsToGridImage([mol, target_mol],
                                           legends=['pred', 'target'],
                                           molsPerRow=2,
                                           subImgSize=(300, 300)
                                           )
                img.save(f'results/{name}/imgs/{i}.png')
    else:
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
        preds, _ = model(X, None, teacher_forcing=False, omit_encoder=True)
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

    # get list of files and dirs in data/encoded_data directory
    if not os.path.isdir('data/encoded_data'):
        os.mkdir('data/encoded_data')

    encoded_data_directories = os.listdir('data/encoded_data')
    parquet_files = []
    for dir_name in encoded_data_directories:
        dir_list = os.listdir(f'data/encoded_data/{dir_name}')
        files = [f'{dir_name}/{name}' for name in dir_list if name.split('.')[-1] == 'parquet']
        parquet_files.extend(files)
    if not parquet_files:
        print('No .parquet files found')

    for file in parquet_files:
        predict(file, is_verbose=True)
