import torch.nn as nn
from torchvision import models


def create_model(architecture: str = "mobilenet_v3_small", pretrained: bool = True,
                 dropout: float = 0.2) -> nn.Module:
    """Create a two-class transfer-learning model."""
    if architecture == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 2))
    elif architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, 2))
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return model
