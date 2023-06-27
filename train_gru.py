# import packages
from gru import example_printer
from gru.dataset import GRUDataset
from gru.gru_v3 import EncoderNet, DecoderNet, EncoderDecoder
from vectorizer import SELFIESVectorizer, determine_alphabet
from gru.cce import CCE, ConsciousCrossEntropy
from split import scaffold_split
from tqdm import tqdm

import torch
import wandb
import pandas as pd
import random

# weights and biases
!wandb login 18463128757ca947acdea58348412cc9a098efd4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

alphabet = pd.read_csv('./GRU_data/alphabet.txt', header=None).values.flatten()
vectorizer = SELFIESVectorizer(alphabet, pad_to_len=128)

data_path = './GRU_data/combned_dataset.parquet'
dataset = pd.read_parquet(data_path)

train_size = 0.9

train_df, val_df = scaffold_split(dataset, train_size)

train_dataset = GRUDataset(train_df, vectorizer)
val_dataset = GRUDataset(val_df, vectorizer)

print("Dataset size:", len(dataset))
print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))

from torch.utils.data import DataLoader
batch_size = 512
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, drop_last=True)

run_name = 'v3_server_test'

train_df.to_parquet(f'./models/{run_name}/train_dataset.parquet')
val_df.to_parquet(f'./models/{run_name}/val_dataset.parquet')

# Set hyperparameters
encoding_size = 512
hidden_size = 512
num_layers = 1
learn_rate = 0.0005
dropout = 0 # dropout must be equal 0 if num_layers = 1
teacher_ratio = 0.5

# Init model
model = EncoderDecoder(
    fp_size=4860,
    encoding_size=encoding_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    dropout=dropout,
    teacher_ratio = teacher_ratio).to(device)

import matplotlib.pyplot as plt

def train(model, train_loader, val_loader, vectorizer, device):

    EPOCHS = 5

    # Define dataframe for training progess display
    epochs_range = range(1,EPOCHS+1)
    metrics = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss']);
    metrics['epoch'] = epochs_range
    
    # Init example printer
    printer = example_printer.ExamplePrinter(data_path, val_loader, num_examples=25)

    # Define pyplot for plotting metrics
    fig, ax = plt.subplots(ncols=2, figsize=(7, 3), layout="constrained")
    dh = display(fig, display_id=True)

    # Define loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    criterion = CCE()

    # wandb config and init
    config = dict()
    config['learning rate'] = learn_rate
    config['encoding size'] = model.encoding_size
    config['criterion'] = criterion
    config['optimizer'] = optimizer
    config['num epochs'] = EPOCHS
    config['Trainable parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config['hidden size'] = model.hidden_size
    config['Number of layers'] = num_layers
    config['Dropout'] = model.decoder.dropout
    config['Batch size'] = batch_size
    config['teacher_ratio'] = teacher_ratio
    wandb.init(project="encoded-token-concat", config=config)

    print("Starting Training of GRU")
    print(f"Device: {device}")
    
    samples = []
    
    # Start training loop
    for epoch in epochs_range:
        print(f'Epoch: {epoch}')
        epoch_loss = 0
        model.train()
        for batch_idx, (X,y) in enumerate(tqdm(train_loader)):
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
        wandb.log(metrics_dict)

        # Update metrics df
        metrics.loc[len(metrics)] = metrics_dict
        
        save_path = f"./models/{run_name}/epoch_{epoch}.pt"
        torch.save(model.state_dict(),save_path)
        
        new_samples = printer(model)
        samples.append(new_samples)
        
    plt.close()
    wandb.finish()
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

model, samples = train(model, train_loader, val_loader, vectorizer, device)
