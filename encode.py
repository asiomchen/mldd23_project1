import torch
import pandas as pd
from src.vae.vae import VAEEncoder
from src.vae.vae_dataset import VAEDataset
import torch.utils.data as D
import numpy as np
from tqdm import tqdm
import argparse


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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder_path = 'models/VAE_64/encoder_epoch_100.pt'
    data_path = 'data/activity_data/d2_klek_100nM.parquet'

    data_name = data_path.split('/')[-1].split('_')[0]
    model_name = encoder_path.split('/')[-2]
    df = pd.read_parquet(data_path)
    dataset = VAEDataset(df, fp_len=4860)
    dataloader = D.DataLoader(dataset, batch_size=1024, shuffle=False)

    model = VAEEncoder(input_size=4860, output_size=64).to(device)
    model.load_state_dict(torch.load(encoder_path, map_location=device))

    with torch.no_grad():
        encoded_list = []
        for i, batch in enumerate(tqdm(dataloader)):
            batch = batch.to(device)
            z = model(batch)[0].cpu().numpy()
            encoded_list.append(z)
        encoded = pd.DataFrame(np.concatenate(encoded_list, axis=0))
    encoded.to_parquet(f'data/activity_data/{data_name}_encoded_with_{model_name}.parquet', idx=False)


if __name__ == '__main__':
    main()
