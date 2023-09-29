import configparser

import torch

from src.gen.generator import EncoderDecoderV3


def initialize_model(config_path, dropout, device):
    """
    Initialize model from a given path
    Args:
        config_path (str): path to the config file
        dropout (bool): whether to use dropout
        device (torch.device): device to be used for training
    Returns:
        model (torch.nn.Module): initialized model
    """
    config = configparser.ConfigParser()
    config.read(config_path)
    model = EncoderDecoderV3(fp_size=int(config['MODEL']['fp_len']),
                             encoding_size=int(config['MODEL']['encoding_size']),
                             hidden_size=int(config['MODEL']['hidden_size']),
                             num_layers=int(config['MODEL']['num_layers']),
                             dropout=float(config['MODEL']['dropout']) if dropout else 0,
                             output_size=31,
                             teacher_ratio=0.0,
                             random_seed=42,
                             fc1_size=int(config['MODEL']['fc1_size']),
                             fc2_size=int(config['MODEL']['fc2_size']),
                             fc3_size=int(config['MODEL']['fc3_size']),
                             encoder_activation=config['MODEL']['encoder_activation']
                             ).to(device)
    return model
