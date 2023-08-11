import torch
import src.vae.vae as vae
import pandas as pd
import time
import wandb
from src.utils.annealing import Annealer


def train_vae(config, model, train_loader, val_loader):
    """
    Training loop for VAE model
    Args:
        config: configparser object
        model: VAE model
        train_loader: torch DataLoader object for training data
        val_loader: torch DataLoader object for validation data
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = int(config['RUN']['epochs'])
    start_epoch = int(config['RUN']['start_epoch'])
    run_name = config['RUN']['run_name']
    learning_rate = float(config['RUN']['learning_rate'])
    use_wandb = config.getboolean('RUN', 'use_wandb')
    kld_weight = float(config['RUN']['kld_weight'])
    recon_weight = float(config['RUN']['recon_weight'])
    kld_annealing = config.getboolean('RUN', 'kld_annealing')
    annealing_epochs = int(config['RUN']['annealing_epochs'])
    annealing_shape = str(config['RUN']['annealing_shape'])
    annealing_agent = Annealer(total_steps=annealing_epochs, shape=annealing_shape, disable=not kld_annealing)

    # start a new wandb run to track this script
    if use_wandb:
        log_dict = {s: dict(config.items(s)) for s in config.sections()}
        wandb.init(
            project='vae',
            config=log_dict,
            name=run_name
        )

    criterion = vae.VAELoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    # Define dataframe for logging progress
    metrics = pd.DataFrame(columns=['epoch', 'train_bce', 'train_kld', 'val_bce', 'val_kld', 'kld_annealing'])

    for epoch in range(start_epoch, epochs + start_epoch):
        print(f'Epoch: {epoch}')
        epoch_bce = 0
        epoch_kld = 0
        start_time = time.time()

        for fp in train_loader:
            fp = fp.to(device)
            encoded, mu, logvar = model(fp)
            bce, kld = criterion(encoded, fp, mu, logvar)
            epoch_bce += bce.item()
            epoch_kld += kld.item()
            bce = bce * recon_weight
            kld = annealing_agent(kld * kld_weight)
            annealing_agent.step()
            loss = bce + kld
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # calculate loss and log to wandb
        avg_bce = epoch_bce / len(train_loader)
        avg_kld = epoch_kld / len(train_loader)
        val_bce, val_kld = evaluate(model, val_loader)
        metrics_dict = {'epoch': epoch,
                        'train_bce': avg_bce,
                        'train_kld': avg_kld,
                        'val_bce': val_bce,
                        'val_kld': val_kld,
                        'kld_annealing': annealing_agent.slope()
                        }

        if use_wandb:
            log_dict = {s: dict(config.items(s)) for s in config.sections()}
            wandb.init(
                project="vae",
                config=log_dict
            )

        if use_wandb:
            wandb.log(metrics_dict)

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


def evaluate(model, val_loader):
    """
    Evaluate model on validation set
    Args:
        model: VAE model
        val_loader: validation set dataloader
    Returns:
        avg_bce (float): average binary cross entropy loss
        avg_kld (float): average KL divergence loss
    """
    model.eval()
    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = vae.VAELoss()
        epoch_bce = 0
        epoch_kld = 0
        for fp in val_loader:
            fp = fp.to(device)
            encoded, mu, logvar = model(fp)
            bce, kld = criterion(encoded, fp, mu, logvar)
            epoch_bce += bce.item()
            epoch_kld += kld.item()
        avg_bce = epoch_bce / len(val_loader)
        avg_kld = epoch_kld / len(val_loader)
        return avg_bce, avg_kld
