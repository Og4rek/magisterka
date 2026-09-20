"""Multi-scale attention ensemble for histopathology patches."""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F


class GatedAttentionFusion(nn.Module):
    def __init__(self, feature_dim: int, branches: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.branches = branches
        self.attention_v = nn.Linear(feature_dim, hidden_dim)
        self.attention_u = nn.Linear(feature_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # features: [B, scales, feature_dim]
        scores = self.attention_w(
            torch.tanh(self.attention_v(features)) * torch.sigmoid(self.attention_u(features))
        ).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(features * weights.unsqueeze(-1), dim=1), weights


class MultiScaleAttentionEnsemble(nn.Module):
    """Shared encoder evaluated at several effective fields of view."""

    def __init__(
        self,
        encoder: nn.Module,
        scales: tuple[float, ...] | list[float] = (0.75, 1.0, 1.25),
        share_encoder: bool = True,
        attention_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not hasattr(encoder, "forward_features"):
            raise TypeError("Multi-scale encoder must expose forward_features().")
        feature_dim = getattr(encoder, "feature_dim", None)
        if feature_dim is None:
            raise ValueError("Encoder must expose feature_dim.")
        self.scales = tuple(float(scale) for scale in scales)
        self.encoders = nn.ModuleList(
            [encoder if branch == 0 or share_encoder else copy.deepcopy(encoder) for branch in range(len(scales))]
        )
        self.share_encoder = share_encoder
        self.fusion = GatedAttentionFusion(int(feature_dim), len(scales), attention_dim)
        self.feature_dim = int(feature_dim)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feature_dim, 1))
        self.last_attention: torch.Tensor | None = None

    @staticmethod
    def _rescaled_view(x: torch.Tensor, scale: float) -> torch.Tensor:
        if scale == 1.0:
            return x
        height, width = x.shape[-2:]
        scaled_height = max(16, round(height * scale))
        scaled_width = max(16, round(width * scale))
        view = F.interpolate(x, size=(scaled_height, scaled_width), mode="bilinear", align_corners=False)
        if scale > 1:
            top = (scaled_height - height) // 2
            left = (scaled_width - width) // 2
            return view[..., top : top + height, left : left + width]
        pad_height, pad_width = height - scaled_height, width - scaled_width
        return F.pad(
            view,
            (pad_width // 2, pad_width - pad_width // 2, pad_height // 2, pad_height - pad_height // 2),
            mode="reflect",
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        views = [self._rescaled_view(x, scale) for scale in self.scales]
        if self.share_encoder and not self.training:
            # One larger encoder call is considerably more efficient than
            # launching the same kernels once per scale. Restrict this path to
            # evaluation so BatchNorm training statistics remain unchanged.
            encoded = self.encoders[0].forward_features(torch.cat(views, dim=0))
            branch_features = list(encoded.split(x.shape[0], dim=0))
        else:
            branch_features = [
                encoder.forward_features(view)
                for encoder, view in zip(self.encoders, views, strict=True)
            ]
        fused, attention = self.fusion(torch.stack(branch_features, dim=1))
        self.last_attention = attention.detach()
        return fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))
