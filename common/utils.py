import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs
