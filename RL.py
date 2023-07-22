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


def main():
    NUM_WORKERS = 3
    train_size = 0.8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    vectorizer = SELFIESVectorizer(pad_to_len=128)
    data_path = 'data/train_data/combined_dataset.parquet'
    dataset = pd.read_parquet(data_path)

    config = configparser.ConfigParser()
    config.read('rl_config.ini')
    run_name = config['RL']['run_name']
    run_name = 'rl_' + run_name
    batch_size = int(config['RL']['batch_size'])
    encoding_size = int(config['RL']['encoding_size'])
    hidden_size = int(config['RL']['hidden_size'])
    num_layers = int(config['RL']['num_layers'])
    dropout = float(config['RL']['dropout'])
    fp_len = int(config['RL']['fp_len'])
    teacher_ratio = float(config['RL']['teacher_ratio'])
    data_path = str(config['RL']['data_path'])
    encoder_path = str(config['RL']['encoder_path'])

    # create a directory for this model if not there
    if not os.path.isdir(f'./models/{run_name}'):
        os.mkdir(f'./models/{run_name}')

    # if train_dataset not generated, perform scaffold split
    if not os.path.isfile(data_path.split('.')[0] + '_train.parquet'):
        train_df, val_df = scaffold_split(dataset, train_size)
        train_df.to_parquet(data_path.split('.')[0] + '_train.parquet')
        val_df.to_parquet(data_path.split('.')[0] + '_val.parquet')
        print("Scaffold split complete")
    else:
        train_df = pd.read_parquet(data_path.split('.')[0] + '_train.parquet')
        val_df = pd.read_parquet(data_path.split('.')[0] + '_val.parquet')

    train_dataset = GRUDataset(train_df, vectorizer, fp_len)
    val_dataset = GRUDataset(val_df, vectorizer, fp_len)

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
        teacher_ratio=teacher_ratio).to(device)

    model.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    # model.load_state_dict(torch.load('models/fixed_cce_3_layers/epoch_100.pt'))
    _ = train(config, model, train_loader, val_loader)

    return None


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
            if epoch == 1:
                output = model(X, y, teacher_forcing=True, reinforcement=False)
                rl_loss = 0
            else:
                output, rl_loss, total_reward = model(X, y, teacher_forcing=True, reinforcement=True)
                epoch_rl_loss += rl_loss.item()
                epoch_total_reward += total_reward.item()
            loss = criterion(y, output)
            epoch_loss += loss.item()
            # print('loss: ', loss.item(), 'rl_loss: ', rl_loss.item())
            (loss + rl_loss).backward()
            optimizer.step()

        epoch_rl_loss = epoch_rl_loss / len(train_loader)
        epoch_total_reward = epoch_total_reward / len(train_loader)
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


if __name__ == "__main__":
    main()
