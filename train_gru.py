# import packages
from src.gen.train import train
from src.gen.dataset import GRUDataset
from src.gen.generator import EncoderDecoderV3
from src.utils.vectorizer import SELFIESVectorizer
from src.utils.split import scaffold_split
import torch
from torch.utils.data import DataLoader
import os
import pandas as pd
import configparser
import argparse


def main():
    """
    Training script for model with variational encoder and GRU decoder
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='gru_config0.ini',
                        help='Path to config file')
    config_path = parser.parse_args().config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    vectorizer = SELFIESVectorizer(pad_to_len=128)

    NUM_WORKERS = 3
    train_size = 0.9
    val_size = round(1 - train_size, 1)
    train_percent = int(train_size * 100)
    val_percent = int(val_size * 100)

    config = configparser.ConfigParser()
    config.read(config_path)
    run_name = str(config['RUN']['run_name'])
    batch_size = int(config['RUN']['batch_size'])
    data_path = str(config['RUN']['data_path'])
    smiles_enum = config.getboolean('RUN', 'smiles_enum')
    model_type = str(config['MODEL']['model_type'])
    encoding_size = int(config['MODEL']['encoding_size'])
    hidden_size = int(config['MODEL']['hidden_size'])
    num_layers = int(config['MODEL']['num_layers'])
    dropout = float(config['MODEL']['dropout'])
    teacher_ratio = float(config['MODEL']['teacher_ratio'])
    fp_len = int(config['MODEL']['fp_len'])
    encoder_path = str(config['MODEL']['encoder_path'])
    checkpoint_path = str(config['MODEL']['checkpoint_path'])
    fc1_size = int(config['MODEL']['fc1_size'])
    fc2_size = int(config['MODEL']['fc2_size'])
    fc3_size = int(config['MODEL']['fc3_size'])
    encoder_activation = str(config['MODEL']['encoder_activation'])

    dataset = pd.read_parquet(data_path).sample(100000)

    # create a directory for this model if not there
    if not os.path.isdir(f'models/{run_name}'):
        os.mkdir(f'models/{run_name}')

    with open(f'models/{run_name}/hyperparameters.ini', 'w') as configfile:
        config.write(configfile)

    # if train_dataset not generated, perform scaffold split
    if (not os.path.isfile(data_path.split('.')[0] + f'_train_{train_percent}.parquet')
            or not os.path.isfile(data_path.split('.')[0] + f'_val_{val_percent}.parquet')):
        train_df, val_df = scaffold_split(dataset, train_size, seed=42, shuffle=True)
        train_df.to_parquet(data_path.split('.')[0] + f'_train_{train_percent}.parquet')
        val_df.to_parquet(data_path.split('.')[0] + f'_val_{val_percent}.parquet')
        print("Scaffold split complete")
    else:
        train_df = pd.read_parquet(data_path.split('.')[0] + f'_train_{train_percent}.parquet')
        val_df = pd.read_parquet(data_path.split('.')[0] + f'_val_{val_percent}.parquet')
    scoring_df = val_df.sample(frac=0.1, random_state=42)

    train_dataset = GRUDataset(train_df, vectorizer, fp_len, smiles_enum=smiles_enum)
    val_dataset = GRUDataset(val_df, vectorizer, fp_len, smiles_enum=False)
    scoring_dataset = GRUDataset(scoring_df, vectorizer, fp_len, smiles_enum=False)

    print("Dataset size:", len(dataset))
    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))
    print("Scoring size:", len(scoring_dataset))

    val_batch_size = batch_size if batch_size < len(val_dataset) else len(val_dataset)
    scoring_batch_size = batch_size if batch_size < len(scoring_dataset) else len(scoring_dataset)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                              drop_last=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=val_batch_size,
                            drop_last=True, num_workers=NUM_WORKERS)
    scoring_loader = DataLoader(scoring_dataset, shuffle=False, batch_size=scoring_batch_size,
                                drop_last=True, num_workers=NUM_WORKERS)

    # Init model
    if model_type == 'EncoderDecoderV3':
        model = EncoderDecoderV3(
            fp_size=fp_len,
            encoding_size=encoding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            teacher_ratio=teacher_ratio,
            output_size=31,  # alphabet length
            fc1_size=fc1_size,
            fc2_size=fc2_size,
            fc3_size=fc3_size,
            encoder_activation=encoder_activation
        ).to(device)

    else:
        raise ValueError('Invalid model type')

    if checkpoint_path.lower() != 'none':
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    elif encoder_path.lower() != 'none':
        model.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    _ = train(config, model, train_loader, val_loader, scoring_loader)
    return None


if __name__ == '__main__':
    main()
