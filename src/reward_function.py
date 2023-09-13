import numpy as np

def reward(vec, penalty:int = 4):
    score = 1 - np.tanh(np.linalg.norm(vec) / penalty)
    return reward