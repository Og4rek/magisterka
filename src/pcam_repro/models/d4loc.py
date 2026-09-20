"""Efficient center-aware D4-equivariant localization network for PCam.

The model keeps D4 group features until the final multi-scale fusion.  Only
point-wise channel projections are applied after orientation pooling, so the
evidence map remains spatially equivariant while the centrally pooled scalar
prediction is invariant to rotations and reflections of the square.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .d4 import D4DenseLayer, D4GroupConv2d, D4LiftingConv2d, GroupBatchNorm


class DenseLayer2d(nn.Module):
    """Non-equivariant control with the same dense connectivity pattern."""

    def __init__(self, in_channels: int, growth_rate: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(
            in_channels,
            growth_rate,
            kernel_size=3,
            padding=1,
            bias=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.conv(F.silu(self.norm(inputs), inplace=True))
        return torch.cat((inputs, features), dim=1)


class Transition2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.conv(F.silu(self.norm(inputs), inplace=True))
        return F.avg_pool2d(inputs, kernel_size=2)


class D4TransitionFast(nn.Module):
    """Point-wise D4 projection followed by spatial downsampling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm = GroupBatchNorm(in_channels)
        self.conv = D4GroupConv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.conv(F.silu(self.norm(inputs), inplace=True))
        batch, channels, group, height, width = inputs.shape
        flat = inputs.reshape(batch, channels * group, height, width)
        flat = F.avg_pool2d(flat, kernel_size=2)
        return flat.reshape(batch, channels, group, flat.shape[-2], flat.shape[-1])


def _dense_stage(
    channels: int,
    growth_rate: int,
    layers: int,
    equivariant: bool,
) -> tuple[nn.Sequential, int]:
    modules: list[nn.Module] = []
    layer_type = D4DenseLayer if equivariant else DenseLayer2d
    for _ in range(layers):
        modules.append(layer_type(channels, growth_rate))
        channels += growth_rate
    return nn.Sequential(*modules), channels


class PCamD4Loc(nn.Module):
    """Center-aware classifier exposing an intrinsic tumor-evidence map.

    Parameters implement the planned ablations without maintaining separate
    architectures: ``use_d4=False`` replaces group convolutions with ordinary
    convolutions, ``center_pooling=False`` uses global evidence pooling, and
    ``multiscale=False`` removes the 24x24 lateral connection.
    """

    def __init__(
        self,
        initial_channels: int = 8,
        growth_rate: int = 4,
        stage_layers: tuple[int, int, int] | list[int] = (2, 2, 2),
        transition_channels: tuple[int, int] | list[int] = (16, 24),
        fusion_channels: int = 48,
        dropout: float = 0.15,
        center_fraction: float = 1.0 / 3.0,
        lse_beta: float = 4.0,
        use_d4: bool = True,
        center_pooling: bool = True,
        multiscale: bool = True,
    ) -> None:
        super().__init__()
        if len(stage_layers) != 3 or len(transition_channels) != 2:
            raise ValueError("PCamD4Loc requires three stages and two transitions.")
        if not 0 < center_fraction <= 1:
            raise ValueError("center_fraction must be in (0, 1].")
        if lse_beta <= 0:
            raise ValueError("lse_beta must be positive.")

        self.use_d4 = bool(use_d4)
        self.center_pooling = bool(center_pooling)
        self.multiscale = bool(multiscale)
        self.center_fraction = float(center_fraction)
        self.lse_beta = float(lse_beta)

        if self.use_d4:
            self.lift = D4LiftingConv2d(3, initial_channels)
            transition_type: type[nn.Module] = D4TransitionFast
        else:
            self.lift = nn.Conv2d(
                3,
                initial_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            )
            transition_type = Transition2d

        channels = initial_channels
        self.stage1, channels = _dense_stage(
            channels,
            growth_rate,
            int(stage_layers[0]),
            self.use_d4,
        )
        self.transition1 = transition_type(channels, int(transition_channels[0]))
        channels = int(transition_channels[0])
        self.stage2, channels = _dense_stage(
            channels,
            growth_rate,
            int(stage_layers[1]),
            self.use_d4,
        )
        local_channels = channels
        self.transition2 = transition_type(channels, int(transition_channels[1]))
        channels = int(transition_channels[1])
        self.stage3, channels = _dense_stage(
            channels,
            growth_rate,
            int(stage_layers[2]),
            self.use_d4,
        )
        deep_channels = channels

        norm_type = GroupBatchNorm if self.use_d4 else nn.BatchNorm2d
        self.local_norm = norm_type(local_channels)
        self.deep_norm = norm_type(deep_channels)
        self.local_projection = nn.Conv2d(
            local_channels,
            fusion_channels,
            kernel_size=1,
            bias=False,
        )
        self.deep_projection = nn.Conv2d(
            deep_channels,
            fusion_channels,
            kernel_size=1,
            bias=False,
        )
        hidden = max(8, fusion_channels // 4)
        self.context_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(deep_channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, fusion_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fusion_activation = nn.SiLU(inplace=False)
        self.dropout = nn.Dropout2d(dropout)
        self.evidence_head = nn.Conv2d(fusion_channels, 1, kernel_size=1)
        self.logit_scale = nn.Parameter(torch.ones(()))
        self.logit_bias = nn.Parameter(torch.zeros(()))
        self.feature_dim = fusion_channels

    @staticmethod
    def _orientation_pool(features: torch.Tensor) -> torch.Tensor:
        return features.mean(dim=2) if features.ndim == 5 else features

    @staticmethod
    def _initial_spatial_pool(features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 5:
            batch, channels, group, height, width = features.shape
            flat = features.reshape(batch, channels * group, height, width)
            flat = F.avg_pool2d(flat, kernel_size=2)
            return flat.reshape(
                batch,
                channels,
                group,
                flat.shape[-2],
                flat.shape[-1],
            )
        return F.avg_pool2d(features, kernel_size=2)

    def forward_evidence_features(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self._initial_spatial_pool(self.lift(inputs))
        features = self.stage1(features)
        features = self.transition1(features)
        local = self.stage2(features)
        deep = self.stage3(self.transition2(local))

        local_2d = self._orientation_pool(
            F.silu(self.local_norm(local), inplace=False)
        )
        deep_2d = self._orientation_pool(
            F.silu(self.deep_norm(deep), inplace=False)
        )
        deep_projected = self.deep_projection(deep_2d)
        deep_upsampled = F.interpolate(
            deep_projected,
            size=local_2d.shape[-2:],
            mode="nearest",
        )
        if self.multiscale:
            fused = self.local_projection(local_2d) + deep_upsampled
        else:
            fused = deep_upsampled
        gate = self.context_gate(deep_2d)
        return self.fusion_activation(fused * (1.0 + gate))

    def evidence_map(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return unnormalized tumor evidence at one quarter input resolution."""
        features = self.forward_evidence_features(inputs)
        return self.evidence_head(self.dropout(features)).squeeze(1)

    def _pool_evidence(self, evidence: torch.Tensor) -> torch.Tensor:
        if self.center_pooling:
            height, width = evidence.shape[-2:]
            center_height = max(1, round(height * self.center_fraction))
            center_width = max(1, round(width * self.center_fraction))
            # A center crop must have the same parity as its parent dimension;
            # otherwise a 90-degree rotation moves it by one cell.
            if (height - center_height) % 2:
                center_height = min(height, center_height + 1)
            if (width - center_width) % 2:
                center_width = min(width, center_width + 1)
            top = (height - center_height) // 2
            left = (width - center_width) // 2
            selected = evidence[
                :,
                top : top + center_height,
                left : left + center_width,
            ]
        else:
            selected = evidence
        flattened = selected.flatten(1)
        normalizer = math.log(flattened.shape[1])
        return (
            torch.logsumexp(self.lse_beta * flattened, dim=1) - normalizer
        ) / self.lse_beta

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_evidence_features(inputs).mean(dim=(-2, -1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pooled = self._pool_evidence(self.evidence_map(inputs))
        logits = self.logit_scale * pooled + self.logit_bias
        return logits.unsqueeze(1)
