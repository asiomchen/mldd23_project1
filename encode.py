import torch
import pandas as pd
from src.vae.vae import VAEEncoder
from src.vae.vae_dataset import VAEDataset
import torch.utils.data as D
import numpy as np
from tqdm import tqdm
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e',
                        '--encoder_path',
                        type=str,
                        help='Path to encoder model')
    parser.add_argument('-d',
                        '--data_path',
                        type=str,
                        help='Path to data')
    data_path = parser.parse_args().data_path
    encoder_path = parser.parse_args().encoder_path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_name = (data_path.split('/')[-1].split('_')[0] +
                 '_epoch_' + encoder_path.split('/')[-1].split('_')[-1].split('.')[0])
    model_name = encoder_path.split('/')[-2]
    df = pd.read_parquet(data_path)
    dataset = VAEDataset(df, fp_len=4860)
    dataloader = D.DataLoader(dataset, batch_size=1024, shuffle=False)

    model = VAEEncoder(input_size=4860, output_size=16).to(device)
    model.load_state_dict(torch.load(encoder_path, map_location=device))

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
        logvars['label'] = df['Class']
        if not os.path.exists(f'data/encoded_data/{model_name}'):
            os.mkdir(f'data/encoded_data/{model_name}')
        mus.to_parquet(f'data/encoded_data/{model_name}/mu_{data_name}.parquet', index=False)
        logvars.to_parquet(f'data/encoded_data/{model_name}/logvar_{data_name}.parquet', index=False)


if __name__ == '__main__':
    main()
