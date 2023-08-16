import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from src.vae.vae import VAE
from src.vae.vae_dataset import VAEDataset
import torch.utils.data as Data
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import jaccard_score
import os
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name',
                    type=str,
                    required=True,
                    action='append',
                    nargs='+',
                    help='Names of the models to be used for encoding (as in models/ directory).'
                    )
name_list = parser.parse_args().name
epoch_list = ['25', '50', '75', '100', '125']


def encode(df, model, device):
    dataset = VAEDataset(df, fp_len=4860)
    dataloader = Data.DataLoader(dataset, batch_size=1024, shuffle=False)
    fps = []
    reconstructions = []
    mus = []
    logvars = []
    with torch.no_grad():
        total_jaccard = 0
        for batch in tqdm(dataloader):
            X = batch.to(device)
            fps.append(X.cpu().numpy())
            recon, mu, logvar = model(X)
            reconstructions.append(recon.cpu().numpy())
            mus.append(mu.cpu().numpy())
            logvars.append(logvar.cpu().numpy())
            X = X.cpu().numpy()
            recon = np.round(recon.cpu().numpy())
            total_jaccard += jaccard_score(X, recon, average='micro')
        mus = np.concatenate(mus, axis=0)
        logvars = np.concatenate(logvars, axis=0)
        fps = np.concatenate(fps, axis=0)
        mean_distance = total_jaccard / len(dataloader)
    return mus, logvars, fps, mean_distance


def transform(mu_values, pca_model):
    pca_results = pca_model.transform(mu_values)
    pca_results = np.ascontiguousarray(pca_results)
    pca_results = pca_results.reshape(2, len(pca_results))
    return pca_results


def read_data(path):
    df = pd.read_parquet(path)
    df = df[df['Ki'] < 100]
    if len(df) > 1000:
        df = df.sample(1000)
    return df


for name in name_list:
    for epoch in epoch_list:

        model_path = 'models/' + f'{name}/vae_epoch_{epoch}.pt'
        latent_size = int(name.split('_')[1])

        model = VAE(input_size=4860, latent_size=latent_size).to(device)
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

        df = pd.read_parquet('data/train_data/combined_dataset.parquet').sample(100000)
        mus, _, _, _ = encode(df, model, device)
        pca = PCA(n_components=2)
        pca.fit(mus)
        all_results = transform(mus, pca)

        ht1a_encoded = encode(read_data('data/activity_data/5ht1a_klek_100nM.parquet'), model, device)
        ht7_encoded = encode(read_data('data/activity_data/5ht7_klek_100nM.parquet'), model, device)
        d2_encoded = encode(read_data('data/activity_data/d2_klek_100nM.parquet'), model, device)
        beta2_encoded = encode(read_data('data/activity_data/beta2_klek_100nM.parquet'), model, device)
        h1_encoded = encode(read_data('data/activity_data/h1_klek_100nM.parquet'), model, device)

        mean_jaccard = (ht1a_encoded[3] +
                        ht7_encoded[3] +
                        d2_encoded[3] +
                        beta2_encoded[3] +
                        h1_encoded[3]) / 5

        ht1a_results = transform(ht1a_encoded[0], pca)
        ht7_results = transform(ht7_encoded[0], pca)
        d2_results = transform(d2_encoded[0], pca)
        beta2_results = transform(beta2_encoded[0], pca)
        h1_results = transform(h1_encoded[0], pca)

        marker_size = 10
        plt.clf()
        fig = plt.figure(figsize=(12, 8))
        plt.scatter(*all_results, marker='o', label='train', s=12, c='lightgrey')
        plt.scatter(*ht1a_results, marker='o', label='5HT1A', s=marker_size)
        plt.scatter(*ht7_results, marker='o', label='5HT7', s=marker_size)
        plt.scatter(*d2_results, marker='o', label='D2', s=marker_size)
        plt.scatter(*beta2_results, marker='o', label='Beta2', s=marker_size)
        plt.scatter(*h1_results, marker='o', label='H1', s=marker_size)
        plt.title(f'{name} latent space projection at epoch {epoch}. Recon_tanimoto: {mean_jaccard:.3f}')
        plt.xlabel('PCA-1')
        plt.ylabel('PCA-2')
        plt.xlim([-3, 3])
        plt.ylim([-3, 3])
        plt.legend(loc='upper left')
        if not os.path.exists(f'results/{name}'):
            os.mkdir(f'results/{name}')
        fig.savefig(f'results/{name}/{name}_epoch_{epoch}.png', dpi=300)
        print(f'Saved successfully to results/{name}/{name}_epoch_{epoch}.png')
