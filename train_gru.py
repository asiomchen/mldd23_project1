# import packages
from gru.example_printer import ExamplePrinter
from gru.dataset import GRUDataset
from gru.gru_v3 import EncoderNet, DecoderNet, EncoderDecoder
from gru.cce import CCE, ConsciousCrossEntropy
from vectorizer import SELFIESVectorizer, determine_alphabet
from split import scaffold_split
from torch.utils.data import DataLoader
import time

import os
import torch
#import wandb
import pandas as pd
import random

#-------------------------------------------------------

run_name = '10_epochs'

train_size = 0.8

batch_size = 64

EPOCHS = 10

# Set hyperparameters
encoding_size = 512
hidden_size = 512
num_layers = 1
learn_rate = 0.0005
dropout = 0 # dropout must be equal 0 if num_layers = 1
teacher_ratio = 0.5

#--------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
alphabet = pd.read_csv('./GRU_data/alphabet.txt', header=None).values.flatten()
vectorizer = SELFIESVectorizer(alphabet, pad_to_len=128)
data_path = './GRU_data/combned_dataset.parquet'
dataset = pd.read_parquet(data_path)

# create a directory for this model if not there

if not os.path.isdir(f'./models/{run_name}'):
    os.mkdir(f'./models/{run_name}')

# if train_dataset not generated, perform scaffold split

if not os.path.isdir(f'./models/train_dataset.parquet'):
    train_df, val_df = scaffold_split(dataset, train_size)
    train_df.to_parquet(f'./models/{run_name}/train_dataset.parquet')
    val_df.to_parquet(f'./models/{run_name}/val_dataset.parquet')
    print("Scaffold split complete")
else:
    train_df = pd.read_parquet(f'./models/{run_name}/train_dataset.parquet')
    val_df = pd.read_parquet(f'./models/{run_name}/val_dataset.parquet')
    
train_dataset = GRUDataset(train_df, vectorizer)
val_dataset = GRUDataset(val_df, vectorizer)

print("Dataset size:", len(dataset))
print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, drop_last=True)

# Init model
model = EncoderDecoder(
    fp_size=4860,
    encoding_size=encoding_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    dropout=dropout,
    teacher_ratio = teacher_ratio)

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

model = nn.DataParallel(model).to(device)

def train(model, train_loader, val_loader, vectorizer, epochs):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define dataframe for logging progess
    epochs_range = range(1,EPOCHS+1)
    metrics = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss']);
    metrics['epoch'] = epochs_range

    # Init example printer
    printer = ExamplePrinter('./models/{run_name}/val_dataset.parquet', val_loader, num_examples=25)

    # Define loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    criterion = CCE()

    print("Starting Training of GRU")
    print(f"Device: {device}")
    samples = []

    # Start training loop
    for epoch in epochs_range:
        start_time = time.time()
        print(f'Epoch: {epoch}')
        epoch_loss = 0
        model.train()
        for batch_idx, (X,y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(X, y, teacher_forcing=True).to(device)
            loss = criterion(y, output)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # calculate loss and log to wandb
        avg_loss = epoch_loss / len(train_loader)
        val_loss = evaluate(model, val_loader)
        metrics_dict = {'epoch': epoch,
                        'train_loss': avg_loss,
                        'val_loss': val_loss}
        #wandb.log(metrics_dict)

        # Update metrics df
        metrics.loc[len(metrics)] = metrics_dict
        if (epoch % 1 == 0):
            save_path = f"./models/{run_name}/epoch_{epoch}.pt"
            torch.save(model.state_dict(),save_path)
        
        metrics.to_csv(f"./models/{run_name}/metrics.csv")
        new_samples = printer(model)
        samples.append(new_samples)
        end_time = time.time()
        loop_time = (end_time - start_time)/60 # in minutes
        print(f'Executed in {loop_time} minutes')

    #wandb.finish()
    return model, samples

def evaluate(model, val_loader):
    model.eval()
    criterion = CCE()
    epoch_loss = 0
    for batch_idx, (X, y) in enumerate(val_loader):
        X = X.to(device)
        y = y.to(device)
        output = model(X, y, teacher_forcing=False).to(device)
        loss = criterion(y, output)
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(val_loader)
    return avg_loss

model, samples = train(model, train_loader, val_loader, vectorizer, EPOCHS)