import math


class Annealing:
    """
    This class is used to anneal the KL divergence loss over the course of training VAE.
    After each call, the step() function should be called to update the current epoch.
    Parameters:
        epochs (int): Number of epochs to reach full KL divergence weight.
        shape (str): Shape of the annealing function. Can be 'linear', 'cosine', or 'logistic'.
    """

    def __init__(self, epochs: int, shape: str, disable=False):
        self.epochs = epochs
        self.current_epoch = 1
        if not disable:
            self.shape = shape
        else:
            self.shape = 'none'
        self.pi = 3.14159265359

    def __call__(self, kld):
        """
        Args:
            kld (torch.tensor): KL divergence loss
        Returns:
            out (torch.tensor): KL divergence loss multiplied by the slope of the annealing function.
        """
        out = kld * self.slope()
        return out

    def slope(self):
        if self.shape == 'linear':
            slope = (self.current_epoch / self.epochs)
        elif self.shape == 'cosine':
            slope = 0.5 + 0.5 * math.cos(self.pi * (self.current_epoch / self.epochs - 1))
        elif self.shape == 'logistic':
            smoothness = self.epochs / 10
            exponent = ((self.epochs / 2) - self.current_epoch) / smoothness
            slope = 1 / (1 + math.exp(exponent))
        elif self.shape == 'none':
            slope = 1.0
        else:
            raise ValueError('Invalid shape for annealing function. Must be linear, cosine, or logistic.')
        return slope

    def step(self):
        if self.current_epoch < self.epochs:
            self.current_epoch += 1
