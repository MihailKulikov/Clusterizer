from LoadDataset import get_dataset
from GetModel import get_model
from efficientnet_pytorch import EfficientNet
import torch
from ImageHandlers import print_image_from_ndarray

data = get_dataset('./Orbits')
print(data[1][2].shape)
print_image_from_ndarray(data[1][2])
