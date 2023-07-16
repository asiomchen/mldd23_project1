from src.vae import vae, vae_dataset
from src.vae.train import train_VAE
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch
import os
import configparser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

config = configparser.ConfigParser()
config.read('vae_config.ini')
run_name = config['VAE']['run_name']
batch_size = int(config['VAE']['batch_size'])
input_size = int(config['VAE']['input_size'])
latent_size = int(config['VAE']['latent_size'])
learning_rate = float(config['VAE']['learning_rate'])
epochs = float(config['VAE']['epochs'])

test_size = 0.8

full_path = './data/GRU_data/combined_dataset.parquet'

# load data

dataset = vae_dataset.VAEDataset(full_path)

# create a directory for this model if not there

if not os.path.isdir(f'./models/{run_name}'):
    os.mkdir(f'./models/{run_name}')

train_dataset, val_dataset = data.random_split(dataset, [test_size, 1-test_size])
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=2)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, drop_last=True, num_workers=2)

# init model

model = vae.VAE(input_size=input_size, latent_size=latent_size).to(device)

# train model

vae = train_VAE(config, model, train_loader, val_loader, plot_loss=False)
