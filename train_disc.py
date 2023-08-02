import src.disc.dataset as dataset
import src.disc.discriminator as discriminator
import src.disc.train as train
import torch.utils.data as data
from torch.utils.data import DataLoader
import configparser
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c',
                    '--config',
                    type=str,
                    default='disc_config.ini',
                    help='Path to config file')
config_path = parser.parse_args().config
config = configparser.ConfigParser()
config.read(config_path)
run_name = str(config['DISC']['run_name'])
batch_size = int(config['DISC']['batch_size'])
latent_size = int(config['DISC']['latent_size'])
mu_path = str(config['DISC']['mu_path'])
logvar_path = str(config['DISC']['logvar_path'])

dataset = dataset.DiscrDataset(mu_path, logvar_path)
train_dataset, val_dataset = data.random_split(dataset, [0.8, 0.2])
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = discriminator.Discriminator(512)

_ = train.train_discr(config, model, train_dataloader, val_dataloader)