from src.utils.vectorizer import SELFIESVectorizer
import torch
import pandas as pd
import time
from src.gru.cce import CCE
import wandb
import selfies as sf
import rdkit.Chem as Chem


def train(config, model, train_loader, val_loader):
    """
    Training loop for GRU model
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epochs = int(config['RUN']['epochs'])
    run_name = str(config['RUN']['run_name'])
    learn_rate = float(config['RUN']['learn_rate'])
    use_wandb = config.getboolean('RUN', 'use_wandb')
    kld_backward = config.getboolean('RUN', 'kld_backward')
    start_epoch = int(config['RUN']['start_epoch'])
    kld_weight = float(config['RUN']['kld_weight'])

    # start a new wandb run to track this script
    if use_wandb:
        log_dict = {s: dict(config.items(s)) for s in config.sections()}
        wandb.init(
            project='gru',
            config=log_dict,
            name=run_name
        )

    # Define dataframe for logging progress
    epochs_range = range(start_epoch, epochs + start_epoch)
    metrics = pd.DataFrame(columns=['epoch', 'kld_loss', 'train_loss', 'val_loss'])

    # Define loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    criterion = CCE()

    print("Starting Training of GRU")
    print(f"Device: {device}")

    # Start training loop
    for epoch in epochs_range:
        model.train()
        start_time = time.time()
        print(f'Epoch: {epoch}')
        epoch_loss = 0
        kld_loss = 0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output, kld_loss = model(X, y, teacher_forcing=True, reinforcement=False)
            kld_loss = kld_loss * kld_weight
            loss = criterion(y, output)
            if kld_backward:
                (loss + kld_loss).backward()
            else:
                loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # calculate loss and log to wandb
        avg_loss = epoch_loss / len(train_loader)
        val_loss = evaluate(model, val_loader)
        metrics_dict = {'epoch': epoch,
                        'kld_loss': kld_loss.item(),
                        'train_loss': avg_loss,
                        'val_loss': val_loss,
                        }
        if use_wandb:
            wandb.log(metrics_dict)

        # Update metrics df
        metrics.loc[len(metrics)] = metrics_dict
        if epoch % 10 == 0:
            save_path = f"./models/{run_name}/epoch_{epoch}.pt"
            torch.save(model.state_dict(), save_path)

        metrics.to_csv(f"./models/{run_name}/metrics.csv", index=False)
        end_time = time.time()
        loop_time = (end_time - start_time) / 60  # in minutes
        print(f'Executed in {loop_time} minutes')

    wandb.finish()
    return None


def train_rl(config, model, train_loader, val_loader):
    """
        Training loop for GRU model with RL
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    supervised = config.getboolean('RUN', 'supervised')
    rl_weight = float(config['RUN']['rl_weight'])
    run_name = str(config['RUN']['run_name'])
    learn_rate = float(config['RUN']['learn_rate'])
    epochs = int(config['RUN']['epochs'])
    start_epoch = int(config['RUN']['start_epoch'])
    teacher_ratio = float(config['MODEL']['teacher_ratio'])
    use_teacher = True if teacher_ratio > 0 else False
    use_wandb = config.getboolean('RUN', 'use_wandb')
    kld_backward = config.getboolean('RUN', 'kld_backward')
    kld_weight = float(config['RUN']['kld_weight'])

    # start a new wandb run to track this script
    if use_wandb:
        log_dict = {s: dict(config.items(s)) for s in config.sections()}
        wandb.init(
            project='gru-rl',
            config=log_dict,
            name=run_name
        )

    # Define dataframe for logging progress
    epochs_range = range(start_epoch, epochs + start_epoch)
    metrics = pd.DataFrame(columns=['epoch', 'kld_loss', 'val_loss', 'rl_loss', 'total_reward'])

    # Define loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    criterion = CCE()

    print("Starting RL Training of GRU")
    print(f"Device: {device}")

    # Start training loop
    for epoch in epochs_range:
        start_time = time.time()
        print(f'Epoch: {epoch}')
        epoch_loss = 0
        epoch_rl_loss = 0
        epoch_total_reward = 0
        kld_loss = 0
        model.train()
        for batch_idx, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            output, kld_loss, rl_loss, total_reward = model(X, y, teacher_forcing=use_teacher, reinforcement=True)
            rl_loss = rl_loss * rl_weight
            kld_loss = kld_loss * kld_weight
            epoch_rl_loss += rl_loss.item()
            epoch_total_reward += total_reward
            loss = torch.tensor(0)
            if supervised:
                loss = criterion(y, output)
                epoch_loss += loss.item()
            if kld_backward:
                (loss + rl_loss + kld_loss).backward()
            else:
                (loss + rl_loss).backward()
            optimizer.step()
            if use_wandb:
                wandb.log(
                    {'batch_idx': batch_idx,
                     'batch_rl_loss': rl_loss.item(),
                     'batch_reward': total_reward})

        epoch_rl_loss = epoch_rl_loss / len(train_loader)
        epoch_total_reward = epoch_total_reward / len(train_loader)
        val_loss = evaluate(model, val_loader)
        metrics_dict = {'epoch': epoch,
                        'kld_loss': kld_loss.item(),
                        'val_loss': val_loss,
                        'rl_loss': epoch_rl_loss,
                        'total_reward': epoch_total_reward,
                        }
        if use_wandb:
            wandb.log(metrics_dict)

        # Update metrics df
        metrics.loc[len(metrics)] = metrics_dict
        if epoch % 5 == 0:
            save_path = f"./models/{run_name}/epoch_{epoch}.pt"
            torch.save(model.state_dict(), save_path)

        metrics.to_csv(f"./models/{run_name}/metrics.csv", index=False)

        end_time = time.time()
        loop_time = (end_time - start_time) / 60  # in minutes
        print(f'Executed in {loop_time} minutes')

    wandb.finish()
    return None


def evaluate(model, val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        criterion = CCE()
        epoch_loss = 0
        for batch_idx, (X, y) in enumerate(val_loader):
            X = X.to(device)
            y = y.to(device)
            output, _ = model(X, y, teacher_forcing=False, reinforcement=False)
            loss = criterion(y, output)
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(val_loader)
        return avg_loss


def eval_properties(model, val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vectorizer = SELFIESVectorizer(pad_to_len=128)
    model.eval()
    with torch.no_grad():
        epoch_qed = 0
        epoch_fp_score = 0
        for batch_idx, (X, y) in enumerate(val_loader):
            X = X.to(device)
            y = y.to(device)
            output, _ = model(X, y, teacher_forcing=False, reinforcement=False)

            sample_idxs = [0]
            for idx in sample_idxs:
                vectorized = output[idx].cpu().numpy()
                selfies = vectorizer.devectorize(vectorized, remove_special=True)
                try:
                    smiles = sf.decoder(selfies)
                    mol = Chem.MolFromSmiles(smiles)
                    epoch_qed += model.get_qed_reward(mol)
                    epoch_fp_score += model.get_fp_reward(mol, X[idx])
                except sf.DecoderError:
                    print('SELFIES decoding error during evaluation')
            epoch_qed = epoch_qed / len(sample_idxs)
            epoch_fp_score = epoch_fp_score / len(sample_idxs)

        mean_qed = epoch_qed / len(val_loader)
        fp_recon_score = epoch_fp_score / len(val_loader)
        return mean_qed, fp_recon_score
