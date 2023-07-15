# import packages
from src.gru.dataset import GRUDataset
from src.gru.generator import EncoderDecoder
from src.gru.cce import CCE
from src.utils.vectorizer import SELFIESVectorizer
from src.utils.split import scaffold_split
from torch.utils.data import DataLoader
import torch
import time
import os
import pandas as pd
import configparser

NUM_WORKERS = 3
train_size = 0.8

config = configparser.ConfigParser()
config.read('rl_config.ini')
run_name = config['RL']['run_name']
batch_size = int(config['RL']['batch_size'])
fp_size = int(config['RL']['fp_size'])
encoding_size = int(config['RL']['encoding_size'])
hidden_size = int(config['RL']['hidden_size'])
num_layers = int(config['RL']['num_layers'])
dropout = float(config['RL']['dropout'])
teacher_ratio = float(config['RL']['teacher_ratio'])

# number of samples per batch when using RL
n_samples = int(config['RL']['n_samples'])

run_name = 'rl_' + run_name
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
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
rl_loader = DataLoader(train_dataset, shuffle=True, batch_size=1, num_workers=NUM_WORKERS)

# Init model
model = EncoderDecoder(
    fp_size=4860,
    encoding_size=encoding_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    dropout=dropout,
    teacher_ratio=teacher_ratio).to(device)

model.encoder.load_state_dict(torch.load('models/VAEEncoder_epoch_100.pt'))


def train(config, model, train_loader, val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run_name = config['RL']['run_name']
    learn_rate = float(config['RL']['learn_rate'])
    epochs = int(config['RL']['epochs'])

    # Define dataframe for logging progress
    epochs_range = range(1, epochs + 1)
    metrics = pd.DataFrame(columns=['epoch', 'val_loss', 'rl_loss', 'total_reward'])

    # Define loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    criterion = CCE()

    print("Starting RL Training of GRU")
    print(f"Device: {device}")

    # Start training loop
    for epoch in epochs_range:
        start_time = time.time()
        print(f'Epoch: {epoch}')
        epoch_loss = 0
        epoch_rl_loss = 0
        epoch_total_reward = 0
        model.train()
        for batch_idx, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output, rl_loss, total_reward = model(X, y, teacher_forcing=True, reinforcement=True)
            loss = criterion(y, output)
            epoch_loss += loss.item()
            epoch_rl_loss += rl_loss
            epoch_total_reward += total_reward

            (loss + rl_loss).backward() #TODO: check values of loss and rl_loss
            optimizer.step()

        epoch_rl_loss = epoch_rl_loss/len(train_loader)
        epoch_total_reward = epoch_total_reward/len(train_loader)
        val_loss = evaluate(model, val_loader)
        metrics_dict = {'epoch': epoch,
                        'val_loss': val_loss,
                        'rl_loss': epoch_rl_loss,
                        'total_reward': epoch_total_reward,
                        }

        # Update metrics df
        metrics.loc[len(metrics)] = metrics_dict
        if epoch % 10 == 0:
            save_path = f"./models/{run_name}/epoch_{epoch}.pt"
            torch.save(model.state_dict(), save_path)

        metrics.to_csv(f"./models/{run_name}/metrics.csv", index=False)
        with open(f"./models/{run_name}/hyperparameters.csv", 'w') as file:
            for key, value in config.items():
                file.write('%s:%s\n' % (key, value))

        end_time = time.time()
        loop_time = (end_time - start_time) / 60  # in minutes
        print(f'Executed in {loop_time} minutes')

    # wandb.finish()
    return model


def evaluate(model, val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    criterion = CCE()
    epoch_loss = 0
    for batch_idx, (X, y) in enumerate(val_loader):
        X = X.to(device)
        y = y.to(device)
        output = model(X, y, teacher_forcing=False)
        loss = criterion(y, output)
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(val_loader)
    return avg_loss


model = train(config, model, train_loader, val_loader)
