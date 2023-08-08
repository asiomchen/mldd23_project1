class Annealing:
    """
    Annealing class for KL divergence.
    """

    def __init__(self, epochs: int, slope='linear'):
        self.epochs = epochs
        self.current_epoch = 1
        self.slope = slope

    def call(self, kld):
        if self.slope == 'linear':
            output = kld * (self.current_epoch / self.epochs)
        return output

    def step(self):
        if self.current_epoch < self.epochs:
            self.current_epoch += 1
