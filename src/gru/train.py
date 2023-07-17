import torch
import pandas as pd
import time
from src.gru.cce import CCE


def train_gru(config, model, train_loader, val_loader):
    """
    Training loopr fo GRU model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epochs = int(config['GRU']['epochs'])
    run_name = config['GRU']['run_name']
    learn_rate = float(config['GRU']['learn_rate'])

    # Define dataframe for logging progess
    epochs_range = range(1, epochs + 1)
    metrics = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss'])

    # Define loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    criterion = CCE()

    print("Starting Training of GRU")
    print(f"Device: {device}")

    # Start training loop
    for epoch in epochs_range:
        start_time = time.time()
        print(f'Epoch: {epoch}')
        epoch_loss = 0
        model.train()
        for batch_idx, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(X, y, teacher_forcing=True, reinforcement=False)
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
        # wandb.log(metrics_dict)

        # Update metrics df
        metrics.loc[len(metrics)] = metrics_dict
        if epoch % 25 == 0:
            save_path = f"./models/{run_name}/epoch_{epoch}.pt"
            torch.save(model.state_dict(), save_path)

        metrics.to_csv(f"./models/{run_name}/metrics.csv", index=False)
        end_time = time.time()
        loop_time = (end_time - start_time) / 60  # in minutes
        print(f'Executed in {loop_time} minutes')


def evaluate(model, val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    criterion = CCE()
    epoch_loss = 0
    for batch_idx, (X, y) in enumerate(val_loader):
        X = X.to(device)
        y = y.to(device)
        output = model(X, y, teacher_forcing=False, reinforcement=False)
        loss = criterion(y, output)
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(val_loader)
    return avg_loss
