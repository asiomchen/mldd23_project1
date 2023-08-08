import torch


class Annealing:
    """
    This class is used to anneal the KL divergence loss over the course of training VAE.
    After each call, the step() function should be called to update the current epoch.
    Parameters:
        epochs (int): Number of epochs to train the VAE.
        slope (str): Slope of the annealing function. Can be 'linear', 'cosine', or 'logistic'.
    Args:
        kld (torch.Tensor): KL divergence loss.
    Returns:
        torch.Tensor: Annealed KL divergence loss.
    """

    def __init__(self, epochs: int, slope='linear'):
        self.epochs = epochs
        self.current_epoch = 1
        self.slope = slope
        self.pi = torch.tensor(3.14159265359)

    def __call__(self, kld):
        if self.slope == 'linear':
            output = kld * (self.current_epoch / self.epochs)
            return output
        elif self.slope == 'cosine':
            cosine_slope = 0.5 + 0.5 * torch.cos(self.pi * self.current_epoch / self.epochs - self.pi)
            output = kld * cosine_slope
            return output
        elif self.slope == 'logistic':
            smoothness = 5
            exponent = ((self.epochs / 2) - self.current_epoch) / smoothness
            sigmoid_slope = 1 / (1 + torch.exp(exponent))
            output = kld * sigmoid_slope
            return output

    def step(self):
        if self.current_epoch < self.epochs:
            self.current_epoch += 1
