# import packages
from src.gru.train import train
from src.gru.dataset import GRUDataset
from src.gru.generator import EncoderDecoder
from src.utils.vectorizer import SELFIESVectorizer
from src.utils.split import scaffold_split
import torch
from torch.utils.data import DataLoader
import os
import pandas as pd
import configparser

# -------------------------------------------------------
NUM_WORKERS = 3
train_size = 0.8

config = configparser.ConfigParser()
config.read('gru_config.ini')
run_name = config['GRU']['run_name']
batch_size = int(config['GRU']['batch_size'])
fp_size = int(config['GRU']['fp_size'])
encoding_size = int(config['GRU']['encoding_size'])
hidden_size = int(config['GRU']['hidden_size'])
num_layers = int(config['GRU']['num_layers'])
dropout = float(config['GRU']['dropout'])
teacher_ratio = float(config['GRU']['teacher_ratio'])

# --------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
vectorizer = SELFIESVectorizer(pad_to_len=128)
data_path = 'data/GRU_data/combined_dataset.parquet'
dataset = pd.read_parquet(data_path)

# create a directory for this model if not there

if not os.path.isdir(f'./models/{run_name}'):
    os.mkdir(f'./models/{run_name}')

# if train_dataset not generated, perform scaffold split

if not os.path.isfile(f'data/GRU_data/train_dataset.parquet'):
    train_df, val_df = scaffold_split(dataset, train_size)
    train_df.to_parquet(f'data/GRU_data/train_dataset.parquet')
    val_df.to_parquet(f'data/GRU_data/val_dataset.parquet')
    print("Scaffold split complete")
else:
    train_df = pd.read_parquet(f'data/GRU_data/train_dataset.parquet')
    val_df = pd.read_parquet(f'data/GRU_data/val_dataset.parquet')

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

#model.load_state_dict(torch.load('models/fixed_cce_3_layers/epoch_175.pt'))
model.encoder.load_state_dict(torch.load('models/VAEEncoder_epoch_100.pt'))
model = train(config, model, train_loader, val_loader)
