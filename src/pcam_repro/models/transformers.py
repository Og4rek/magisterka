"""Vision Transformer and local-global hybrid used in the PCam matrix."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, embedding_dim: int) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size.")
        self.grid_size = image_size // patch_size
        self.projection = nn.Conv2d(3, embedding_dim, patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x).flatten(2).transpose(1, 2)


class ViTSmall(nn.Module):
    """ViT-S/16-like classifier adapted to 96 x 96 PCam patches."""

    def __init__(
        self,
        image_size: int = 96,
        patch_size: int = 16,
        embedding_dim: int = 384,
        depth: int = 8,
        heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, embedding_dim)
        token_count = self.patch_embedding.grid_size**2
        self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position = nn.Parameter(torch.zeros(1, token_count + 1, embedding_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=int(embedding_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, norm=nn.LayerNorm(embedding_dim))
        self.dropout = nn.Dropout(dropout)
        self.feature_dim = embedding_dim
        self.classifier = nn.Linear(embedding_dim, 1)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embedding(x)
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat((class_token, tokens), dim=1)
        if tokens.shape[1] != self.position.shape[1]:
            raise ValueError("Input resolution differs from the configured ViT image_size.")
        return self.encoder(self.dropout(tokens + self.position))[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


class LocalConvStem(nn.Module):
    """Overlapping local feature extractor before global self-attention."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        hidden = embedding_dim // 2
        self.layers = nn.Sequential(
            nn.Conv2d(3, hidden, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, embedding_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.GELU(),
            nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1, groups=embedding_dim, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LocalGlobalViT(nn.Module):
    """CNN-local / Transformer-global hybrid reconstructed for PCam."""

    def __init__(
        self,
        image_size: int = 96,
        embedding_dim: int = 256,
        depth: int = 6,
        heads: int = 8,
        mlp_ratio: float = 3.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if image_size % 4:
            raise ValueError("LGViT image size must be divisible by four.")
        self.stem = LocalConvStem(embedding_dim)
        grid = image_size // 4
        token_count = grid * grid
        self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position = nn.Parameter(torch.zeros(1, token_count + 1, embedding_dim))
        layer = nn.TransformerEncoderLayer(
            embedding_dim,
            heads,
            int(embedding_dim * mlp_ratio),
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, depth, norm=nn.LayerNorm(embedding_dim))
        self.feature_dim = embedding_dim
        self.classifier = nn.Linear(embedding_dim, 1)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.stem(x)
        tokens = features.flatten(2).transpose(1, 2)
        tokens = torch.cat((self.class_token.expand(x.shape[0], -1, -1), tokens), dim=1)
        if tokens.shape[1] != self.position.shape[1]:
            # Allow evaluation at another resolution while preserving the class position.
            side = int((self.position.shape[1] - 1) ** 0.5)
            spatial = self.position[:, 1:].transpose(1, 2).reshape(1, self.feature_dim, side, side)
            spatial = F.interpolate(spatial, size=features.shape[-2:], mode="bicubic", align_corners=False)
            position = torch.cat((self.position[:, :1], spatial.flatten(2).transpose(1, 2)), dim=1)
        else:
            position = self.position
        return self.encoder(tokens + position)[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))
