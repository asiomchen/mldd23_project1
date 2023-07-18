from src.vae.vae_dataset import VAEDataset
from src.vae.train import train_vae
from src.vae.vae import VAE
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch
import os
import configparser


def main():
    """
    Training script for the VAE model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    config = configparser.ConfigParser()
    config.read('vae_config.ini')
    run_name = config['VAE']['run_name']
    batch_size = int(config['VAE']['batch_size'])
    input_size = int(config['VAE']['input_size'])
    latent_size = int(config['VAE']['latent_size'])
    full_path = str(config['VAE']['data_path'])

    test_size = 0.8

    # load data
    dataset = VAEDataset(full_path, fp_len=input_size)

    # create a directory for this model if not there
    if not os.path.isdir(f'./models/{run_name}'):
        os.mkdir(f'./models/{run_name}')

    learning_rate = float(config['VAE']['learning_rate'])
    epochs = float(config['VAE']['epochs'])
    train_dataset, val_dataset = data.random_split(dataset, [test_size, 1 - test_size])
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=2)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, drop_last=True, num_workers=2)

    # init model
    model = VAE(input_size=input_size, latent_size=latent_size).to(device)

    # train model
    vae = train_vae(config, model, train_loader, val_loader)

    return None


if __name__ == '__main__':
    main()
