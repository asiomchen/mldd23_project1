import torch
import pandas as pd
from src.gen.dataset import VAEDataset
from src.gen.generator import EncoderDecoderV3
import torch.utils.data as D
import numpy as np
from tqdm import tqdm
import argparse
import os
import configparser


def main(encoder_path, data_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_name = (data_path.split('/')[-1].split('_')[0] +
                 '_epoch_' + encoder_path.split('/')[-1].split('_')[-1].split('.')[0])
    model_name = encoder_path.split('/')[-2]
    df = pd.read_parquet(data_path)
    dataset = VAEDataset(df, fp_len=4860)
    dataloader = D.DataLoader(dataset, batch_size=1024, shuffle=False)

    if not os.path.exists(f'data/encoded_data'):
        os.mkdir(f'data/encoded_data')

    config = configparser.ConfigParser()
    config.read(f'models/{model_name}/hyperparameters.ini')
    model = EncoderDecoderV3(fp_size=config.getint('MODEL', 'fp_len'),
                             hidden_size=config.getint('MODEL', 'hidden_size'),
                             encoding_size=config.getint('MODEL', 'encoding_size'),
                             num_layers=config.getint('MODEL', 'num_layers'),
                             dropout=0.0,
                             output_size=42,
                             teacher_ratio=0.0,
                             random_seed=42,
                             fc1_size=config.getint('MODEL', 'fc1_size'),
                             fc2_size=config.getint('MODEL', 'fc2_size'),
                             fc3_size=config.getint('MODEL', 'fc3_size')
                             ).to(device)
    model.load_state_dict(torch.load(encoder_path, map_location=device))
    model = model.encoder

    with torch.no_grad():
        mu_list = []
        logvar_list = []
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            mu, logvar = model(batch)
            mu_list.append(mu.cpu().numpy())
            logvar_list.append(logvar.cpu().numpy())
        mus = pd.DataFrame(np.concatenate(mu_list, axis=0))
        logvars = pd.DataFrame(np.concatenate(logvar_list, axis=0))
        mus.columns = mus.columns.astype(str)
        logvars.columns = logvars.columns.astype(str)
        mus['label'] = df['Class']
        mus['smiles'] = df['smiles']
        if not os.path.exists(f'data/encoded_data/{model_name}'):
            os.mkdir(f'data/encoded_data/{model_name}')
        mus.to_parquet(f'data/encoded_data/{model_name}/mu_{data_name}.parquet', index=False)
        # logvars.to_parquet(f'data/encoded_data/{model_name}/logvar_{data_name}.parquet', index=False)

        out_config = configparser.ConfigParser()
        out_config['INFO'] = {'model_path': encoder_path,
                                  'encoding_size': config.getint('MODEL', 'encoding_size')}
        if not os.path.exists(f'data/encoded_data/{model_name}/info.ini'):
            with open(f'data/encoded_data/{model_name}/info.ini', 'x') as file:
                out_config.write(file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--model_path',
                        type=str,
                        help='Path to encoder model')
    parser.add_argument('-d',
                        '--data_path',
                        type=str,
                        help='Path to data',
                        default='all')
    encoder_path = parser.parse_args().model_path
    data_path = parser.parse_args().data_path
    if data_path == 'all':
        paths = [
            'data/activity_data/5ht1a_klek_100nM.parquet',
            'data/activity_data/5ht7_klek_100nM.parquet',
            'data/activity_data/beta2_klek_100nM.parquet',
            'data/activity_data/d2_klek_100nM.parquet',
            'data/activity_data/h1_klek_100nM.parquet'
                 ]
    else:
        paths = [data_path]
    for data_path in paths:
        main(encoder_path, data_path)
