import torch
import time
import pandas as pd


def train_disc(config, model, train_loader, val_loader):
    """
    Training loop for discriminator model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epochs = int(config['DISC']['epochs'])
    start_epoch = int(config['DISC']['start_epoch'])
    run_name = config['DISC']['run_name']
    learning_rate = float(config['DISC']['learning_rate'])

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    # Define dataframe for logging progress
    metrics = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'accuracy'])

    for epoch in range(start_epoch, epochs + start_epoch):
        print(f'Epoch: {epoch}')
        epoch_loss = 0
        start_time = time.time()

        for X, y in train_loader:
            X = X.to(device)
            y = y.unsqueeze(1).to(device)
            y_pred = model(X)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # calculate loss and log to wandb
        avg_loss = epoch_loss / len(train_loader)
        val_loss, acc = evaluate(model, val_loader)
        metrics_dict = {'epoch': epoch,
                        'train_loss': avg_loss,
                        'val_loss': val_loss,
                        'accuracy': acc}

        # Update metrics df
        metrics.loc[len(metrics)] = metrics_dict

        if epoch % 100 == 0:
            save_path = f"./models/{run_name}/epoch_{epoch}.pt"
            torch.save(model.state_dict(), save_path)

        metrics.to_csv(f"./models/{run_name}/metrics.csv")

        end_time = time.time()
        loop_time = (end_time - start_time) / 60  # in minutes
        print(f'Executed in {loop_time} minutes')

    return model


def evaluate(model, val_loader):
    model.eval()
    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = torch.nn.BCELoss()
        epoch_loss = 0
        epoch_acc = 0
        for X, y in val_loader:
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X).squeeze(1)
            loss = criterion(y_pred, y)
            epoch_loss += loss.item()
            epoch_acc += accuracy(y_pred.cpu(), y.cpu())
        avg_loss = epoch_loss / len(val_loader)
        avg_accuracy = epoch_acc / len(val_loader)
    return avg_loss, avg_accuracy


def accuracy(y_pred: torch.tensor, y: torch.tensor):
    batch_size = y.shape[0]
    y_pred = torch.round(y_pred).bool()
    y = y.bool()
    acc = torch.sum(y_pred == y).item() / batch_size
    return acc
