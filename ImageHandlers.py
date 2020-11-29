import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Module


def print_image_from_ndarray(image: np.ndarray):
    plt.imshow(image)
    plt.show()


def extract_features(extractor: Module, image: np.ndarray) -> np.ndarray:
    return np.array([])
