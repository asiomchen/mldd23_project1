import numpy as np
import pickle
import torch
from src.clf.classifier import MLPClassifier


class MLPScorer:
    """
    Scorer class for Bayesian optimization, based on MLP
    """

    def __init__(self, path, latent_size, penalize=False, device='cpu'):
        """
        Args:
            path: path to the saved model
            latent_size: size of the latent space
            penalize: if True, penalize for values outside of bounds
        """
        self.model = MLPClassifier(latent_size=latent_size, use_sigmoid=True).to(device)
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.penalize = penalize

    def __call__(self, **args) -> float:
        input_vector = np.array(list({**args}.values()))
        input_tensor = torch.from_numpy(input)
        input_tensor = input_tensor.to(torch.float32)
        pred = self.model(input_tensor)
        output = pred.cpu().detach().numpy()[0]
        if self.penalize:
            output = output * (gaussian_reward(input_vector, 0, 10))
        return output


class SKLearnScorer:
    """
    Scorer class for Bayesian optimization, based on scikit-learn models
    """

    def __init__(self, path, penalize=False):
        """
        Args:
            path: path to the saved model
            penalize: if True, penalize for values outside of bounds
        """
        with open(path, 'rb') as file:
            self.model = pickle.load(file)
        self.penalize = penalize

    def __call__(self, **args) -> float:
        input_vector = list({**args}.values())
        input_vector = np.array(input_vector).reshape(1, -1)
        output = self.model.predict_proba(input_vector)[0][0]
        if self.penalize:
            output = output * gaussian_reward(input_vector, mu=6.47, sigma=2.44)
        return output


def gaussian_reward(vec: np.array, mu: float, sigma: float):
    x = np.linalg.norm(vec)
    c = np.sqrt(2 * np.pi)
    score = np.exp(-0.5 * ((x - mu) / sigma)**2) / sigma / c
    return score
