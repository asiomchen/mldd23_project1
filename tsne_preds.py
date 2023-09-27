import argparse
import configparser
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.utils.data as Data
from sklearn.manifold import TSNE
from tqdm import tqdm

from src.gen.dataset import VAEDataset
from src.gen.generator import EncoderDecoderV3
from src.utils.modelinit import initialize_model


def main(model_path, activity_data, predicted_data, seed):
    config = configparser.ConfigParser()
    random.seed(seed)
    templist = model_path.split('/')
    config_path = '/'.join(templist[:-1]) + '/hyperparameters.ini'
    config.read(config_path)
    model_name = model_path.split('/')[1]
    epoch = model_path.split('_')[-1].split('.')[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = initialize_model(config_path, dropout=False, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    preds = pd.read_csv(predicted_data).drop(columns=['norm', 'score']).sample(1000)
    preds_numpy = preds.to_numpy()
    activity_data = pd.read_parquet(activity_data)
    activity_encoded, _ = encode(activity_data, model, device)

    random_state = seed
    cat = np.concatenate((preds_numpy, activity_encoded), axis=0)
    print('Running t-SNE...')
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=40, n_jobs=-1)

    results = tsne.fit_transform(cat)

    all_df = pd.DataFrame((results[-len(activity_encoded):]), columns=['x', 'y'])
    all_df['activity'] = ['D2 active' if x == 1 else 'D2 inactive' for x in activity_data['activity']]
    drugs_df = pd.DataFrame((results[:-len(activity_encoded)]), columns=['x', 'y'])

    sns.set(rc={'figure.figsize': (10, 8)})
    sns.set_style("white")
    with sns.color_palette("Paired"):
        sns.scatterplot(
            data=all_df,
            x="x", y="y", hue="activity", s=5
        )
    sns.scatterplot(
        data=drugs_df,
        x="x", y="y", c='crimson', s=10, label='Predictions'
    )
    plt.xlabel('t-SNE dim 1')
    plt.ylabel('t-SNE dim 2')
    plt.savefig(f'plots/{model_name}_epoch_{epoch}_pred_tsne.png')
    print('Images saved to plots/')

def encode(df, model, device):
    """
    Encodes the fingerprints of the molecules in the dataframe using VAE encoder.
    Args:
        df (pd.DataFrame): dataframe containing 'fps' column with Klekota&Roth fingerprints
            in the form of a list of integers (dense representation)
        model (EncoderDecoderV3): model to be used for encoding
        device (torch.device): device to be used for encoding
    Returns:
        mus (np.ndarray): array of means of the latent space
        logvars (np.ndarray): array of logvars of the latent space
    """
    dataset = VAEDataset(df, fp_len=model.fp_size)
    dataloader = Data.DataLoader(dataset, batch_size=1024, shuffle=False)
    mus = []
    logvars = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            X = batch.to(device)
            mu, logvar = model.encoder(X)
            mus.append(mu.cpu().numpy())
            logvars.append(logvar.cpu().numpy())

        mus = np.concatenate(mus, axis=0)
        logvars = np.concatenate(logvars, axis=0)
    return mus, logvars


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-m', type=str, required=True)
    parser.add_argument('--activity_data', '-d', type=str, required=True)
    parser.add_argument('--predicted_data', '-p', type=str, required=True)
    parser.add_argument('--random_seed', '-r', type=int, default=42)
    args = parser.parse_args()
    main(args.model_path, args.activity_data, args.predicted_data, args.random_seed)
