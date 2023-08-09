import torch
import math


class Annealing:
    """
    Annealing class for KL divergence.
    """

    def __init__(self, epochs: int, shape='linear'):
        self.epochs = epochs
        self.current_epoch = 1
        self.shape = shape

    def __call__(self, kld: torch.tensor):
        out = kld * self.slope()
        return out

    def slope(self):
        if self.shape == 'linear':
            slope = self.current_epoch / self.epochs
        elif self.slope == 'cosine':
            slope = 0.5 + 0.5 * math.cos(math.pi * (self.current_epoch / self.epochs - 1))
        elif self.shape == 'none':
            slope = 1.0
        return slope

    def step(self):
        if self.current_epoch < self.epochs:
            self.current_epoch += 1
