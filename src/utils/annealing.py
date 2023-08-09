import torch


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
        self.pi = torch.tensor(3.14159265359)

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
        if self.slope == 'linear':
            slope = (self.current_epoch / self.epochs)
        elif self.slope == 'cosine':
            slope = 0.5 + 0.5 * torch.cos(self.pi * self.current_epoch / self.epochs - self.pi)
        elif self.slope == 'logistic':
            smoothness = 5
            exponent = ((self.epochs / 2) - self.current_epoch) / smoothness
            slope = 1 / (1 + torch.exp(exponent))
        elif self.slope == 'none':
            slope = 1
        else:
            raise ValueError('Invalid shape for annealing function. Must be linear, cosine, or logistic.')
        return slope

    def step(self):
        if self.current_epoch < self.epochs:
            self.current_epoch += 1
