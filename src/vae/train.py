from torch.optim.lr_scheduler import ReduceLROnPlateau
import IPython.display as display
import torch
import src.vae as vae
import pandas as pd
import matplotlib.pyplot as plt
import time


def train_VAE(config, model, train_loader, val_loader, plot_loss=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epochs = int(config['GRU']['epochs'])
    run_name = config['GRU']['run_name']
    learning_rate = float(config['GRU']['learn_rate'])

    criterion = vae.VAELoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=20, verbose=True)

    # Define dataframe for logging progess
    metrics = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss']);

    if plot_loss:
        fig, ax = plt.subplots()
        dh = display(fig, display_id=True)

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

        scheduler.step(avg_loss)

        # Update metrics df
        metrics.loc[len(metrics)] = metrics_dict
        save_path = f"./models/{run_name}/epoch_{epoch}.pt"
        torch.save(model.state_dict(), save_path)

        metrics.to_csv(f"./models/{run_name}/metrics.csv")

        if (epoch % 50 == 0):
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
        loop_time = (end_time - start_time) / 60  # in minutes
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
