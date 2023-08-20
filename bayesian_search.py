from bayes_opt import BayesianOptimization
from src.disc.discriminator import Discriminator
import torch
import pandas as pd
import argparse
import numpy as np
from tqdm import tqdm

device = torch.device('cpu')

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--n_samples', type=int, default=10)
parser.add_argument('-p', '--init_points', type=int, default=3)
parser.add_argument('-i', '--n_iter', type=int, default=8)

n_samples = parser.parse_args().n_samples
init_points = parser.parse_args().init_points
n_iter = parser.parse_args().n_iter

latent_size = 64
model = Discriminator(latent_size=latent_size, use_sigmoid=False).to(device)
model.load_state_dict(torch.load('models/discr_d2_mandarynka_epoch_100/epoch_150.pt', map_location=device))

args = {str(n): 0.0 for n in range(latent_size)}
pbounds = {str(p): (-5, 5) for p in range(latent_size)}


def foo(**args):
    dict = {**args}
    input_tensor = torch.tensor(list(dict.values()))
    input_tensor = input_tensor.to(torch.float32)
    pred = model(input_tensor)
    output = pred.cpu().detach().numpy()[0]
    return output


def sample_latent(n_samples, init_points, n_iter, pbounds, random_state=42):
    optimizer = BayesianOptimization(
        f=foo,
        pbounds=pbounds,
        random_state=random_state,
        verbose=0
    )
    vector_list = []
    for _ in tqdm(range(n_samples)):
        optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter
        )
        vector = np.array(list(optimizer.max['params'].values()))
        vector_list.append(vector)
    vector_list = np.array(vector_list)
    return vector_list


samples = pd.DataFrame(sample_latent(n_samples, init_points, n_iter, pbounds))
samples.columns = [str(n) for n in range(latent_size)]
samples.to_parquet('samples.parquet')
