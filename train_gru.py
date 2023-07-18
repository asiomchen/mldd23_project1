# import packages
from src.gru.train import train_gru
from src.gru.dataset import GRUDataset
from src.gru.generator import EncoderDecoder
from src.utils.vectorizer import SELFIESVectorizer
from src.utils.split import scaffold_split
import torch
from torch.utils.data import DataLoader
import os
import pandas as pd
import configparser


def main():
    """
    Training script for model with variational encoder and GRU decoder
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    vectorizer = SELFIESVectorizer(pad_to_len=128)

    NUM_WORKERS = 3
    train_size = 0.8

    config = configparser.ConfigParser()
    config.read('gru_config.ini')
    run_name = config['GRU']['run_name']
    batch_size = int(config['GRU']['batch_size'])
    encoding_size = int(config['GRU']['encoding_size'])
    hidden_size = int(config['GRU']['hidden_size'])
    num_layers = int(config['GRU']['num_layers'])
    dropout = float(config['GRU']['dropout'])
    teacher_ratio = float(config['GRU']['teacher_ratio'])
    encoder_path = str(config['GRU']['encoder_path'])
    data_path = str(config['GRU']['data_path'])

    dataset = pd.read_parquet(data_path)

    # create a directory for this model if not there
    if not os.path.isdir(f'models/{run_name}'):
        os.mkdir(f'models/{run_name}')

    with open(f'models/{run_name}/hyperparameters.ini', 'w') as configfile:
        config.write(configfile)

    # if train_dataset not generated, perform scaffold split
    if not os.path.isfile(data_path.split('.')[0] + '_train.parquet'):
        train_df, val_df = scaffold_split(dataset, train_size)
        train_df.to_parquet(data_path.split('.')[0] + '_train.parquet')
        val_df.to_parquet(data_path.split('.')[0] + '_val.parquet')
        print("Scaffold split complete")
    else:
        train_df = pd.read_parquet(data_path.split('.')[0] + '_train.parquet')
        val_df = pd.read_parquet(data_path.split('.')[0] + '_val.parquet')

    train_dataset = GRUDataset(train_df, vectorizer)
    val_dataset = GRUDataset(val_df, vectorizer)

    print("Dataset size:", len(dataset))
    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                              drop_last=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size,
                            drop_last=True, num_workers=NUM_WORKERS)

    # Init model
    model = EncoderDecoder(
        fp_size=4860,
        encoding_size=encoding_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        teacher_ratio=teacher_ratio,
    ).to(device)

    #  Load model parameters
    #model.load_state_dict(torch.load('models/fixed_cce_3_layers/epoch_175.pt'))
    model.encoder.load_state_dict(torch.load(encoder_path))
    model = train_gru(config, model, train_loader, val_loader)
    return None


if __name__ == '__main__':
    main()
