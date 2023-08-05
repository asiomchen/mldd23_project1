# import packages
from src.gru.dataset import GRUDataset
from src.gru.generator import EncoderDecoder
from src.utils.vectorizer import SELFIESVectorizer
from src.utils.split import scaffold_split
from torch.utils.data import DataLoader
from src.gru.train import train_rl
import torch
import os
import pandas as pd
import configparser
import argparse


def main():
    NUM_WORKERS = 3
    train_size = 0.8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    vectorizer = SELFIESVectorizer(pad_to_len=128)

    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='config_files/rl_config.ini',
                        help='Path to config file')
    config_path = parser.parse_args().config

    config = configparser.ConfigParser()
    config.read(config_path)

    run_name = str(config['RUN']['run_name'])
    batch_size = int(config['RUN']['batch_size'])
    data_path = str(config['RUN']['data_path'])
    encoding_size = int(config['MODEL']['encoding_size'])
    hidden_size = int(config['MODEL']['hidden_size'])
    num_layers = int(config['MODEL']['num_layers'])
    dropout = float(config['MODEL']['dropout'])
    fp_len = int(config['MODEL']['fp_len'])
    teacher_ratio = float(config['MODEL']['teacher_ratio'])
    encoder_path = str(config['MODEL']['encoder_path'])
    encoder_nograd = config.getboolean('MODEL', 'encoder_nograd')
    checkpoint_path = str(config['MODEL']['checkpoint_path'])

    # create a directory for this model if not there
    if not os.path.isdir(f'./models/{run_name}'):
        os.mkdir(f'./models/{run_name}')

    dataset = pd.read_parquet(data_path)

    # if train_dataset not generated, perform scaffold split
    if not os.path.isfile(data_path.split('.')[0] + '_train.parquet'):
        train_df, val_df = scaffold_split(dataset, train_size)
        train_df.to_parquet(data_path.split('.')[0] + '_train.parquet')
        val_df.to_parquet(data_path.split('.')[0] + '_val.parquet')
        print("Scaffold split complete")
    else:
        train_df = pd.read_parquet(data_path.split('.')[0] + '_train.parquet')
        val_df = pd.read_parquet(data_path.split('.')[0] + '_val.parquet')

    train_dataset = GRUDataset(train_df, vectorizer, fp_len)
    val_dataset = GRUDataset(val_df, vectorizer, fp_len)

    print("Dataset size:", len(dataset))
    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                              drop_last=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size,
                            drop_last=True, num_workers=NUM_WORKERS)

    # Init model
    model = EncoderDecoder(
        fp_size=fp_len,
        encoding_size=encoding_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        teacher_ratio=teacher_ratio,
        encoder_nograd=encoder_nograd
    ).to(device)

    if checkpoint_path != 'None':
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    elif encoder_path != 'None':
        model.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    _ = train_rl(config, model, train_loader, val_loader)

    return None


if __name__ == "__main__":
    main()
