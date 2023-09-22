import argparse
import configparser
import os
import time
import pandas as pd
import rdkit.Chem.Draw as Draw
import rdkit.Chem as Chem
from src.pred.droputpred import predict_with_dropout
import torch
from src.gen.generator import EncoderDecoderV3
from src.pred.props import get_properties

def main(file_path, model_path, config_path):
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
        output_size=31,
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

    # get properties
    print(f'Getting properties...') if verbosity > 1 else None
    df = get_properties(df)

    # save data as csv
    os.mkdir(f'results/{name}')
    with open(f'results/{name}/config.ini', 'w') as configfile:
        config.write(configfile)
    df.to_csv(f'results/{name}/{name}.csv',
                       index=False)

    print(f'Saved data to results/{name}') if verbosity > 0 else None

    # save images
    os.mkdir(f'results/{name}/imgs')
    for i, smiles in enumerate(df['smiles']):
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
                        '--data_path',
                        type=str,
                        help='Path to data file')
    parser.add_argument('-m',
                        '--model_path',
                        type=str,
                        help='Path to model weights')
    args = parser.parse_args()
    main(file_path=args.data_path, model_path=args.model_path, config_path=args.config)
