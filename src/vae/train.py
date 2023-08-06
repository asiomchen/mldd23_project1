from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import src.vae.vae as vae
import pandas as pd
import time
import wandb


def train_vae(config, model, train_loader, val_loader):
    """
    Training loop for VAE model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epochs = int(config['VAE']['epochs'])
    run_name = config['VAE']['run_name']
    learning_rate = float(config['VAE']['learning_rate'])
    use_wandb = config.getboolean('VAE', 'use_wandb')

    # start a new wandb run to track this script
    if use_wandb:
        log_dict = {s: dict(config.items(s)) for s in config.sections()}
        wandb.init(
            project='gmum-servers',
            config=log_dict,
            name=run_name
        )

    criterion = vae.VAELoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=50, verbose=True)

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

        if use_wandb:
            log_dict = {s: dict(config.items(s)) for s in config.sections()}
            wandb.init(
                project="vae",
                config=log_dict
            )

        if use_wandb:
            wandb.log(metrics_dict)

        scheduler.step(avg_loss)

        # Update metrics df
        metrics.loc[len(metrics)] = metrics_dict

        if epoch % 25 == 0:
            save_path = f"models/{run_name}/encoder_epoch_{epoch}.pt"
            torch.save(model.encoder.state_dict(), save_path)
            save_path = f"models/{run_name}/vae_epoch_{epoch}.pt"
            torch.save(model.state_dict(), save_path)

        metrics.to_csv(f"./models/{run_name}/metrics.csv")

        end_time = time.time()
        loop_time = (end_time - start_time) / 60  # in minutes
        print(f'Executed in {loop_time} minutes')

    return model


def train_cvae(config, model, train_loader, val_loader):
    """
    Training loop for CVAE model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epochs = int(config['VAE']['epochs'])
    run_name = config['VAE']['run_name']
    learning_rate = float(config['VAE']['learning_rate'])
    use_wandb = config.getboolean('VAE', 'use_wandb')

    # start a new wandb run to track this script
    if use_wandb:
        log_dict = {s: dict(config.items(s)) for s in config.sections()}
        wandb.init(
            project='cvae',
            config=log_dict,
            name=run_name
        )

    criterion = vae.VAELoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=50, verbose=True)

    # Define dataframe for logging progress
    metrics = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss'])

    for epoch in range(1, epochs + 1):
        print(f'Epoch: {epoch}')
        epoch_loss = 0
        start_time = time.time()

        for fp, y in train_loader:
            fp = fp.to(device)
            y = y.to(device)
            encoded, mu, logvar = model(fp, y)
            loss = criterion(encoded, fp, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # calculate loss and log to wandb
        avg_loss = epoch_loss / len(train_loader)
        val_loss = evaluate_cvae(model, val_loader)
        metrics_dict = {'epoch': epoch,
                        'train_loss': avg_loss,
                        'val_loss': val_loss}

        scheduler.step(avg_loss)

        # Update metrics df
        metrics.loc[len(metrics)] = metrics_dict

        if use_wandb:
            wandb.log(metrics_dict)

        if epoch % 10 == 0:
            save_path = f"models/{run_name}/encoder_epoch_{epoch}.pt"
            torch.save(model.encoder.state_dict(), save_path)
            save_path = f"models/{run_name}/vae_epoch_{epoch}.pt"
            torch.save(model.state_dict(), save_path)

        metrics.to_csv(f"./models/{run_name}/metrics.csv", index=False)

        end_time = time.time()
        loop_time = (end_time - start_time) / 60  # in minutes
        print(f'Executed in {loop_time} minutes')

    wandb.finish()
    return None


def evaluate(model, val_loader):
    model.eval()
    with torch.no_grad():
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


def evaluate_cvae(model, val_loader):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = vae.VAELoss()
    epoch_loss = 0
    for fp, y in val_loader:
        fp = fp.to(device)
        y = y.to(device)
        encoded, mu, logvar = model(fp, y)
        loss = criterion(encoded, fp, mu, logvar)
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(val_loader)
    return avg_loss
