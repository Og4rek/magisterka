"""Architecture registry for all reproduction tracks."""

from __future__ import annotations

from typing import Any

from torch import nn

from ..config import ModelConfig
from .cnn import SimCLRModel, SmallCNN, make_torchvision_classifier
from .d4 import GDenseNetD4
from .d4loc import PCamD4Loc
from .lotenet import LoTeNet
from .multiscale import MultiScaleAttentionEnsemble
from .partial import PartialSE2CNN
from .transformers import LocalGlobalViT, ViTSmall


MODEL_NAMES = (
    "small_cnn",
    "resnet18",
    "resnet50",
    "densenet121",
    "inception_v3",
    "gdensenet_d4",
    "pcam_d4loc",
    "partial_se2",
    "vit_small",
    "lgvit",
    "lotenet",
    "multiscale_attention",
)


def _make_base(name: str, pretrained: bool, parameters: dict[str, Any]) -> nn.Module:
    if name == "small_cnn":
        return SmallCNN(**parameters)
    if name in {"resnet18", "resnet50", "densenet121", "inception_v3"}:
        return make_torchvision_classifier(name, pretrained)
    if name == "gdensenet_d4":
        return GDenseNetD4(**parameters)
    if name == "pcam_d4loc":
        return PCamD4Loc(**parameters)
    if name == "partial_se2":
        return PartialSE2CNN(**parameters)
    if name == "vit_small":
        return ViTSmall(**parameters)
    if name == "lgvit":
        return LocalGlobalViT(**parameters)
    if name == "lotenet":
        return LoTeNet(**parameters)
    raise KeyError(f"Unknown model '{name}'. Available models: {', '.join(MODEL_NAMES)}")


def build_model(config: ModelConfig, training_mode: str = "supervised") -> nn.Module:
    parameters = dict(config.parameters)
    projection_dim = int(parameters.pop("projection_dim", 128))
    if config.name == "multiscale_attention":
        encoder_name = str(parameters.pop("encoder", "resnet18"))
        encoder_pretrained = bool(parameters.pop("encoder_pretrained", config.pretrained))
        encoder_parameters = dict(parameters.pop("encoder_parameters", {}))
        model = MultiScaleAttentionEnsemble(
            _make_base(encoder_name, encoder_pretrained, encoder_parameters),
            **parameters,
        )
    else:
        model = _make_base(config.name, config.pretrained, parameters)
    if training_mode == "simclr":
        return SimCLRModel(model, projection_dim)
    return model


__all__ = ["MODEL_NAMES", "build_model"]
