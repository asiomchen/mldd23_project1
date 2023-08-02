import torch
import time
import pandas as pd


def train_discr(config, model, train_loader, val_loader):
    """
    Training loop for discriminator model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epochs = int(config['VAE']['epochs'])
    run_name = config['VAE']['run_name']
    learning_rate = float(config['VAE']['learning_rate'])

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    # Define dataframe for logging progress
    metrics = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss'])

    for epoch in range(1, epochs + 1):
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

        # Update metrics df
        metrics.loc[len(metrics)] = metrics_dict

        if epoch % 25 == 0:
            save_path = f"./models/{run_name}/epoch_{epoch}.pt"
            torch.save(model.encoder.state_dict(), save_path)

        metrics.to_csv(f"./models/{run_name}/metrics.csv")

        end_time = time.time()
        loop_time = (end_time - start_time) / 60  # in minutes
        print(f'Executed in {loop_time} minutes')

    return model


def evaluate(model, val_loader):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.BCELoss()
    epoch_loss = 0
    for X, y in val_loader:
        X = X.to(device)
        y = y.to(device)
        y_pred = model(X)
        loss = criterion(y_pred, y)
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(val_loader)
    return avg_loss
