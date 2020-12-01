from torch import nn
from efficientnet_pytorch import EfficientNet
from typing import List
import numpy as np
import torch


def get_model(model_name: str) -> nn.Module:
    model = EfficientNet.from_pretrained(model_name)
    model._fc = nn.Identity()
    model._swish = nn.Identity()
    return model


def get_patients_features(backbone: nn.Module, data: List[List[np.ndarray]]) -> np.ndarray:
    with torch.no_grad():
        return torch.stack(
            list(map(lambda patient_data: torch.stack(
                list(map(lambda image: backbone(image), patient_data))), data))).reshape(len(data), -1).numpy()
