"""Hierarchical locally orderless tensor-network classifier."""

from __future__ import annotations

import itertools

import torch
from torch import nn
from torch.nn import functional as F


class MatrixProductState(nn.Module):
    """Open-boundary MPS mapping a short sequence to a latent vector."""

    def __init__(self, physical_dim: int, bond_dim: int, sequence_length: int = 4) -> None:
        super().__init__()
        if sequence_length < 2:
            raise ValueError("MPS sequence must contain at least two sites.")
        self.sequence_length = sequence_length
        self.left = nn.Parameter(torch.empty(physical_dim, bond_dim))
        self.cores = nn.ParameterList(
            [
                nn.Parameter(torch.empty(physical_dim, bond_dim, bond_dim))
                for _ in range(sequence_length - 1)
            ]
        )
        nn.init.xavier_uniform_(self.left)
        for core in self.cores:
            nn.init.xavier_uniform_(core)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.shape[-2] != self.sequence_length:
            raise ValueError("Unexpected sequence length for the MPS block.")
        state = torch.einsum("...d,db->...b", sequence[..., 0, :], self.left)
        for site, core in enumerate(self.cores, start=1):
            state = torch.einsum("...b,...d,dbe->...e", state, sequence[..., site, :], core)
        return state


class LocallyOrderlessMPSBlock(nn.Module):
    """Contracts every 2 x 2 neighbourhood and averages scan permutations."""

    def __init__(
        self,
        in_channels: int,
        physical_dim: int,
        bond_dim: int,
        permutation_count: int = 4,
    ) -> None:
        super().__init__()
        self.project = nn.Conv2d(in_channels, physical_dim, 1)
        self.mps = MatrixProductState(physical_dim, bond_dim, sequence_length=4)
        d4_orders = [
            (0, 1, 2, 3),
            (1, 3, 0, 2),
            (3, 2, 1, 0),
            (2, 0, 3, 1),
            (1, 0, 3, 2),
            (2, 3, 0, 1),
            (0, 2, 1, 3),
            (3, 1, 2, 0),
        ]
        remaining = [
            order for order in itertools.permutations(range(4)) if order not in d4_orders
        ]
        canonical = d4_orders + remaining
        if permutation_count <= 0 or permutation_count > len(canonical):
            raise ValueError("permutation_count must be in [1,24].")
        # D4 transformations occur first, followed by the remaining permutations.
        selected = canonical[:permutation_count]
        self.register_buffer("permutations", torch.tensor(selected, dtype=torch.long))
        self.norm = nn.BatchNorm2d(bond_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.project(x))
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2))
        batch, physical_dim, height, width = x.shape
        patches = F.unfold(x, kernel_size=2, stride=2)
        patches = patches.reshape(batch, physical_dim, 4, -1).permute(0, 3, 2, 1)
        # Evaluate all selected scan orders as one extra tensor dimension.
        # This retains the orderless average while avoiding a Python-level MPS
        # call for every permutation.
        permuted = patches[:, :, self.permutations, :]
        out = self.mps(permuted).mean(dim=2).transpose(1, 2)
        out = out.reshape(batch, -1, height // 2, width // 2)
        return F.gelu(self.norm(out))


class LoTeNet(nn.Module):
    """Three-level LoTeNet-style hierarchical MPS for 96 x 96 RGB input."""

    def __init__(
        self,
        physical_dim: int = 6,
        bond_dims: tuple[int, ...] | list[int] = (12, 20, 32),
        permutation_count: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = 3
        for bond_dim in bond_dims:
            blocks.append(
                LocallyOrderlessMPSBlock(
                    in_channels,
                    physical_dim,
                    int(bond_dim),
                    permutation_count,
                )
            )
            in_channels = int(bond_dim)
        self.blocks = nn.Sequential(*blocks)
        self.feature_dim = in_channels
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_channels, 1))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x).mean(dim=(2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))
