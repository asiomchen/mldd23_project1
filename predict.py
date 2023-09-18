import argparse
import configparser
import multiprocessing as mp
import os
import queue
import time
import pandas as pd
import rdkit.Chem.Draw as Draw
import rdkit.Chem as Chem
from src.pred.droputpred import predict_with_dropout
import torch
from src.gen.generator import EncoderDecoderV3
from src.pred.filter import molecule_filter


def predict(file_path, model_path, config_path):
    """
    Predicting molecules using the trained model.

    Args:
        file_path (str): Path to the file containing latent vectors.
        model_path (str): Path to the model weights.
        config_path: Path to the config file.
    Returns: None
    """

    # setup
    start_time = time.time()

    config = configparser.ConfigParser()
    config.read(config_path)
    use_cuda = config['SCRIPT'].getboolean('use_cuda')
    verbosity = config['SCRIPT'].getboolean('verbosity')
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'

    # get file name
    name = file_path.split('/')[-1].split('.')[0]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    name = '_'.join([name, timestamp])

    print(f'Using {device} device') if verbosity > 0 else None

    model_epoch = model_path.split('/')[-1]
    model_config_path = model_path.replace(model_epoch, 'hyperparameters.ini')
    model_config = configparser.ConfigParser()
    model_config.read(model_config_path)

    # load model
    model = EncoderDecoderV3(
        fp_size=int(model_config['MODEL']['fp_len']),
        encoding_size=int(model_config['MODEL']['encoding_size']),
        hidden_size=int(model_config['MODEL']['hidden_size']),
        num_layers=int(model_config['MODEL']['num_layers']),
        dropout=float(model_config['MODEL']['dropout']),
        teacher_ratio=0.0,
        use_cuda=use_cuda,
        output_size=42,
        random_seed=42,
        fc1_size=int(model_config['MODEL']['fc1_size']),
        fc2_size=int(model_config['MODEL']['fc2_size']),
        fc3_size=int(model_config['MODEL']['fc3_size']),
        encoder_activation=model_config['MODEL']['encoder_activation']
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f'Loaded model from {model_path}') if verbosity > 1 else None

    # load data
    if file_path.endswith('.csv'):
        query_df = pd.read_csv(file_path)
    elif file_path.endswith('.parquet'):
        query_df = pd.read_parquet(file_path)
    else:
        raise ValueError('Data file format not supported (must be .csv or .parquet)')

    for col in ['smiles', 'label', 'score', 'activity', 'norm']:
        if col in query_df.columns:
            query_df = query_df.drop(columns=[col])
    input_vector = query_df.to_numpy()
    print(f'Loaded data from {file_path}') if verbosity > 1 else None

    # get predictions
    print(f'Getting predictions for file {file_path}...') if verbosity > 1 else None
    df = predict_with_dropout(model,
                              input_vector,
                              n_iter=int(config['SCRIPT']['n_iter']),
                              device=device,)

    # pred out non-druglike molecules
    print('Filtering out non-druglike molecules...') if verbosity > 1 else None

    # multiprocessing
    manager = mp.Manager()
    return_list = manager.list()
    cpu_count = mp.cpu_count()

    if config['SCRIPT']['n_workers'] == '-1':
        cpus = cpu_count
    else:
        n_workers = int(config['SCRIPT']['n_workers'])
        cpus = n_workers if n_workers < cpu_count else cpu_count

    print("Number of workers:", cpus) if verbosity > 0 else None
    q = queue.Queue()

    # prepare a process for each file and add to queue
    if len(df) < 10:
        chunk_size = len(df)
    elif len(df) < 500:
        chunk_size = 20
    else:
        chunk_size = 100
    n_chunks = int(len(df) / chunk_size) + 1 if len(df) % chunk_size != 0 else int(len(df) / chunk_size)
    chunks = [df[i * chunk_size:(i + 1) * chunk_size] for i in range(n_chunks)]
    for chunk in chunks:
        proc = mp.Process(target=molecule_filter, args=(chunk, config, return_list))
        q.put(proc)

    # handle the queue
    processes = []
    while True:
        if q.empty():
            print("(mp) Queue handled successfully") if verbosity > 0 else None
            break
        if len(mp.active_children()) < cpus:
            proc = q.get()
            proc.start()
            if q.qsize() % 5 == 0:
                print('(mp) Processes in queue: ', q.qsize()) if verbosity > 0 else None
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

    print(f'Saved data to results/{name}') if verbosity > 0 else None

    # save images
    os.mkdir(f'results/{name}/imgs')
    for i, smiles in enumerate(druglike_df['smiles']):
        mol = Chem.MolFromSmiles(smiles)
        Draw.MolToFile(mol, f'results/{name}/imgs/{i}.png', size=(300, 300))

    time_elapsed = time.time() - start_time
    print(f'{name} processed in {(time_elapsed / 60):.2f} minutes')


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

