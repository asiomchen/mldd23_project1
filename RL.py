# import packages
from src.gru.example_printer import ExamplePrinter
from src.gru.dataset import GRUDataset
from src.gru.vae_gru import VAEEncoder, DecoderNet, EncoderDecoder
from src.gru.cce import CCE
from src.utils.vectorizer import SELFIESVectorizer
from src.utils.split import scaffold_split
from torch.utils.data import DataLoader
from src.utils.qed import QED, mean_batch_QED
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
import time
import os
import pandas as pd
import random
import numpy as np
#import wandb

#-------------------------------------------------------

run_name = 'reinforcement'
train_size = 0.8
batch_size = 128
EPOCHS = 200
NUM_WORKERS = 3

# Set hyperparameters
encoding_size = 512
hidden_size = 512
num_layers = 3
learn_rate = 0.0001
dropout = 0.2 # dropout must be equal 0 if num_layers = 1
teacher_ratio = 0.5

#--------------------------------------------------------

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
    train_df = pd.read_parquet(f'data/GRU_data/train_dataset.parquet')[:1000]
    val_df = pd.read_parquet(f'data/GRU_data/val_dataset.parquet')[:1000]
    
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

n_samples = 10

# Init model
model = EncoderDecoder(
    fp_size=4860,
    encoding_size=encoding_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    dropout=dropout,
    teacher_ratio = teacher_ratio).to(device)

model.encoder.load_state_dict(torch.load('models/VAEEncoder_epoch_100.pt'))

# wandb config and init
config = dict()
config['learning rate'] = learn_rate
config['encoding size'] = model.encoding_size
config['num epochs'] = EPOCHS
config['Trainable parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
config['hidden size'] = model.hidden_size
config['Number of layers'] = num_layers
config['Dropout'] = model.decoder.dropout
config['Batch size'] = batch_size
config['teacher_ratio'] = teacher_ratio
#wandb.init(project="gmum-servers", config=config, dir='./tmp')

def train(model, train_loader, val_loader, rl_loader, vectorizer, epochs):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define dataframe for logging progess
    epochs_range = range(1,EPOCHS+1)
    metrics = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'rl_loss']);
    
    # Init example printer
    printer = ExamplePrinter('data/GRU_data/val_dataset.parquet', val_loader, num_examples=25)
    
    # Define loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    criterion = CCE()
    #sheduler = ReduceLROnPlateau(optimizer, 'min', patience=25, verbose=True)

    print("Starting RL Training of GRU")
    print(f"Device: {device}")

    # Start training loop
    for epoch in epochs_range:
        start_time = time.time()
        print(f'Epoch: {epoch}')
        epoch_loss = 0
        model.train()
        for batch_idx, (X,y) in enumerate(train_loader):

            # Supervised learning
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(X, y, teacher_forcing=True)
            loss = criterion(y, output)
            epoch_loss += loss.item()

            # Reinforcement learning
            rl_loss = 0
            total_reward = 0
            gamma = 0.97

            for _ in range(n_samples):

                # Sampling new trajectory
                X, y = next(iter(rl_loader))
                X = X.to(device)
                y = y.to(device)
                trajectory = model(X, y, teacher_forcing=False)
                trajectory = trajectory.detach().cpu().numpy().squeeze(0)
                reward = QED(trajectory, vectorizer)
                total_reward += reward

                # "Following" the trajectory and accumulating the loss
                discounted_reward = reward
                p = 1
                print(trajectory.shape)
                argmax_trajectory = np.argmax(trajectory, axis=1)
                hidden = model.decoder.init_hidden(batch_size)
                while argmax_trajectory[p] != 40:
                    out, hidden = model.decoder(trajectory[p], hidden)
                    log_probs = F.log_softmax(out, dim=1)
                    top_i = trajectory[p+1]
                    rl_loss -= (log_probs[0, top_i]*discounted_reward)
                    discounted_reward = discounted_reward * gamma
                    p += 1

            # Doing backward pass and parameters update
            rl_loss = rl_loss / n_samples
            total_reward = total_reward / n_samples

            (loss + rl_loss).backward()
            optimizer.step()
            
            
        # calculate loss and log to wandb
        avg_loss = epoch_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, epoch)
        metrics_dict = {'epoch': epoch,
                        'train_loss': avg_loss,
                        'val_loss': val_loss,
                        'rl_loss': rl_loss}
        #wandb.log(metrics_dict)
        #sheduler.step(val_loss)
        
        # Update metrics df
        metrics.loc[len(metrics)] = metrics_dict
        if (epoch % 25 == 0):
            save_path = f"./models/{run_name}/epoch_{epoch}.pt"
            torch.save(model.state_dict(),save_path)
        
        metrics.to_csv(f"./models/{run_name}/metrics.csv", index=False)
        with open(f"./models/{run_name}/hyperparameters.csv", 'w') as file:
            for key, value in config.items(): 
                file.write('%s:%s\n' % (key, value))
        new_samples = printer(model)
        samples.append(new_samples)
        end_time = time.time()
        loop_time = (end_time - start_time)/60 # in minutes
        print(f'Executed in {loop_time} minutes')

    #wandb.finish()
    return model

def evaluate(model, val_loader, epoch):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    criterion = CCE()
    score = 0
    qed_exec_time = 0
    epoch_loss = 0
    for batch_idx, (X, y) in enumerate(val_loader):
        X = X.to(device)
        y = y.to(device)
        output = model(X, y, teacher_forcing=False)
        loss = criterion(y, output)
        epoch_loss += loss.item()
    if (epoch % 10 == 0):
        output = output.cpu().detach().numpy()
        score += mean_batch_QED(output, vectorizer)
    avg_loss = epoch_loss / len(val_loader)
    if (epoch % 10 == 0):
        print(f'Mean QED = {score/len(val_loader)}')
    return avg_loss

model = train(model, train_loader, val_loader, rl_loader, vectorizer, EPOCHS)