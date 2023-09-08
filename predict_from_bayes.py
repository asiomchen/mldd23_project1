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

from src.gen.generator import EncoderDecoderV3
from src.pred.filter import molecule_filter
from src.utils.vectorizer import SELFIESVectorizer


def predict(file_path, model_path, config_path, is_verbose=True):
    """
    Predicting molecules using the trained model.

    Args:
        file_path (str): Path to the file containing latent vectors.
        model_path (str): Path to the model weights.
        is_verbose (bool): Whether to print progress.
    Returns: None
    """

    # setup
    start_time = time.time()

    # get file name
    name = file_path.split('/')[-1].split('.')[0]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    name = '_'.join([name, timestamp])
    use_cuda = False
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device') if is_verbose else None

    model_config_path = model_path.replace('epoch_80.pt', 'hyperparameters.ini')
    config = configparser.ConfigParser()
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

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    print(f'Loaded model from {model_path}') if is_verbose else None

    config.read(config_path)

    # load data
    query_df = pd.read_parquet(file_path)
    for col in ['smiles', 'label', 'score']:
        if col in query_df.columns:
            query_df = query_df.drop(columns=[col])
    input_tensor = torch.tensor(query_df.to_numpy(), dtype=torch.float32)

    # get predictions
    print(f'Getting predictions for file {file_path}...') if is_verbose else None
    df = get_predictions(model,
                         input_tensor,
                         use_cuda=use_cuda,
                         verbose=is_verbose
                         )

    # pred out non-druglike molecules
    print('Filtering out non-druglike molecules...') if is_verbose else None

    manager = mp.Manager()
    return_list = manager.list()

    cpus = mp.cpu_count()
    print("Number of cpus: ", cpus)

    q = queue.Queue()

    # prepare a process for each file and add to queue
    if len(df) < 5000:
        chunk_size = 20
    else:
        chunk_size = 1000
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
            proc.start()
            if q.qsize() % 5 == 0:
                print('(mp) Processes in queue: ', q.qsize())
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        '-c',
                        type=str,
                        default='config_files/pred_config.ini',
                        help='Path to config file')
    parser.add_argument('-d',
                        '--data',
                        type=str,
                        help='Path to data file')
    parser.add_argument('-m',
                        '--model_path',
                        type=str,
                        help='Path to model weights')
    args = parser.parse_args()
    predict(file_path=args.data, model_path=args.model_path, config_path=args.config)

