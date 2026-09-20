"""Canonical convolutional baselines and the SimCLR projection wrapper."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torchvision import models


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class SmallCNN(nn.Module):
    """A reproducible four-stage CNN baseline for 96 x 96 PCam patches."""

    def __init__(self, width: int = 32, dropout: float = 0.25) -> None:
        super().__init__()
        channels = [width, 2 * width, 4 * width, 8 * width]
        stages: list[nn.Module] = []
        in_channels = 3
        for out_channels in channels:
            stages.extend(
                [
                    ConvNormAct(in_channels, out_channels),
                    ConvNormAct(out_channels, out_channels),
                    nn.MaxPool2d(2),
                ]
            )
            in_channels = out_channels
        self.features = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = channels[-1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], 1),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x)).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier[1:](self.forward_features(x))


class TorchvisionClassifier(nn.Module):
    """Normalizes torchvision classifiers to a scalar-logit PCam interface."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        feature_forward: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self._feature_forward = feature_forward
        self.classifier = nn.Linear(feature_dim, 1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self._feature_forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


def _resnet(name: str, pretrained: bool) -> TorchvisionClassifier:
    constructor = getattr(models, name)
    weights_enum = getattr(models, f"{name.replace('resnet', 'ResNet')}_Weights")
    backbone = constructor(weights=weights_enum.DEFAULT if pretrained else None)
    feature_dim = int(backbone.fc.in_features)
    backbone.fc = nn.Identity()
    return TorchvisionClassifier(backbone, feature_dim, backbone.forward)


def _densenet121(pretrained: bool) -> TorchvisionClassifier:
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    backbone = models.densenet121(weights=weights)
    feature_dim = int(backbone.classifier.in_features)
    backbone.classifier = nn.Identity()
    return TorchvisionClassifier(backbone, feature_dim, backbone.forward)


class InceptionPCam(nn.Module):
    """Inception-v3 with disabled auxiliary head, valid for 96 x 96 input."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.Inception_V3_Weights.DEFAULT if pretrained else None
        # torchvision requires the auxiliary branch while loading official
        # weights; remove it immediately afterwards to retain one-output API.
        inception_kwargs = {
            "weights": weights,
            "aux_logits": pretrained,
        }
        if not pretrained:
            inception_kwargs["init_weights"] = False
        backbone = models.inception_v3(**inception_kwargs)
        backbone.aux_logits = False
        backbone.AuxLogits = None
        self.feature_dim = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(self.feature_dim, 1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


def make_torchvision_classifier(name: str, pretrained: bool = False) -> nn.Module:
    if name in {"resnet18", "resnet50"}:
        return _resnet(name, pretrained)
    if name == "densenet121":
        return _densenet121(pretrained)
    if name == "inception_v3":
        return InceptionPCam(pretrained)
    raise KeyError(f"Unsupported torchvision architecture: {name}")


class SimCLRModel(nn.Module):
    """Encoder with the two-layer nonlinear projection head from SimCLR."""

    def __init__(self, encoder: nn.Module, projection_dim: int = 128) -> None:
        super().__init__()
        if not hasattr(encoder, "forward_features"):
            raise TypeError("A SimCLR encoder must expose forward_features().")
        feature_dim = getattr(encoder, "feature_dim", None)
        if feature_dim is None and isinstance(encoder, SmallCNN):
            feature_dim = encoder.classifier[-1].in_features
        if feature_dim is None:
            raise ValueError("Could not determine encoder feature dimension.")
        self.encoder = encoder
        self.feature_dim = int(feature_dim)
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, projection_dim),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.projector(self.forward_features(x)), dim=1)
