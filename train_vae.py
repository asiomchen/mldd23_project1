from vae import vae, vae_dataset
import numpy as np
import pandas as pd
import torch
import os
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

#-----------------------------------------------------------------

latent_size = 512
input_size = 4860
learning_rate = 0.0001

batch_size = 32

test_size = 0.8

EPOCHS = 500

full_path = './GRU_data/combined_dataset.parquet'

run_name = 'vae_500epochs'

#-----------------------------------------------------------------


# load data

dataset = vae_dataset.VAEDataset(full_path)

# create a directory for this model if not there

if not os.path.isdir(f'./models/{run_name}'):
    os.mkdir(f'./models/{run_name}')

import torch.utils.data as data
train_dataset, val_dataset = data.random_split(dataset, [test_size, 1-test_size])

from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=2)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, drop_last=True, num_workers=2)

# init model

model = vae.VAE(input_size=input_size, latent_size=latent_size).to(device)

from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

def train_VAE(model, train_loader, val_loader, learning_rate, epochs, plot_loss=False):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = vae.VAELoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    sheduler = ReduceLROnPlateau(optimizer, 'min', patience=20, verbose=True)
    
    # Define dataframe for logging progess
    metrics = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss']);
    
    if plot_loss:
        fig, ax = plt.subplots()
        dh = display(fig, display_id=True)

    for epoch in range(1, epochs+1):
        print(f'Epoch: {epoch}')
        epoch_loss = 0
        start_time = time.time()
        
        for fp in train_loader:
            fp = fp.to(device)
            encoded, mu, logvar = model(fp)
            loss = criterion(encoded, fp, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # calculate loss and log to wandb
        avg_loss = epoch_loss / len(train_loader)
        val_loss = evaluate(model, val_loader)
        metrics_dict = {'epoch': epoch,
                        'train_loss': avg_loss,
                        'val_loss': val_loss}
        
        sheduler.step(avg_loss)
        
        # Update metrics df
        metrics.loc[len(metrics)] = metrics_dict
        save_path = f"./models/{run_name}/epoch_{epoch}.pt"
        torch.save(model.state_dict(),save_path)
        
        metrics.to_csv(f"./models/{run_name}/metrics.csv")
        
        if (epoch%50 == 0):
            torch.save(model.state_dict(), f'./models/CVAE_full_epoch_{epoch}.pt')
        
        if plot_loss:
            ax.clear()
            ax.plot(losses)
            ax.set_title(f'Epoch {epoch}, Loss: {loss.item():.2f}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_yscale('log')
            dh.update(fig)
         
        end_time = time.time()
        loop_time = (end_time - start_time)/60 # in minutes
        print(f'Executed in {loop_time} minutes')
         
    return model

def evaluate(model, val_loader):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = vae.VAELoss()
    epoch_loss = 0
    for fp in val_loader:
        fp = fp.to(device)
        encoded, mu, logvar = model(fp)
        loss = criterion(encoded, fp, mu, logvar)
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(val_loader)
    return avg_loss

vae = train_VAE(model, train_loader, val_loader, learning_rate, 
                          epochs=EPOCHS, plot_loss=False)
