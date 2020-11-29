from torch import nn
from efficientnet_pytorch import EfficientNet


def get_model(model_name: str) -> nn.Module:
    model = EfficientNet.from_pretrained(model_name)
    model._fc = nn.Identity()
    model._swish = nn.Identity()
    return model
