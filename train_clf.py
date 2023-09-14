import src.clf.dataset
import src.clf.classifier as discriminator
import src.clf.train as train
import torch.utils.data as data
from torch.utils.data import DataLoader
import configparser
import argparse
import torch
import os


def main():
    """
    Training script for discriminator model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    NUM_WORKERS = 3
    train_size = 0.9

    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='config_files/clf_config.ini',
                        help='Path to config file')
    config_path = parser.parse_args().config
    config = configparser.ConfigParser()
    config.read(config_path)
    run_name = str(config['RUN']['run_name'])
    batch_size = int(config['RUN']['batch_size'])
    mu_path = str(config['RUN']['mu_path'])
    latent_size = int(config['MODEL']['latent_size'])
    checkpoint_path = str(config['MODEL']['checkpoint_path'])

    # create a directory for this model if not there
    if not os.path.isdir(f'models/{run_name}'):
        os.mkdir(f'models/{run_name}')

    with open(f'models/{run_name}/hyperparameters.ini', 'w') as configfile:
        config.write(configfile)

    dataset = src.clf.dataset.ClfDataset(mu_path)
    train_dataset, val_dataset = data.random_split(dataset, [train_size, 1 - train_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=NUM_WORKERS)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=NUM_WORKERS)

    print("Dataset size:", len(dataset))
    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))

    model = discriminator.NLPClassifier(latent_size).to(device)

    if checkpoint_path != 'None':
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    _ = train.train_clf(config, model, train_dataloader, val_dataloader)

    return None


if __name__ == '__main__':
    main()
