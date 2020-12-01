import matplotlib.pyplot as plt
import numpy as np
import torch


def print_image(image: torch.Tensor):
    image = torch.mean(image[0], 0)
    print(image.shape)
    plt.imshow(image.numpy(), cmap='Greys')
    plt.show()
