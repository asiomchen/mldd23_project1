import torch


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
            return self.current_epoch / self.epochs
        elif self.shape == 'none':
            return 1.0

    def step(self):
        if self.current_epoch < self.epochs:
            self.current_epoch += 1
