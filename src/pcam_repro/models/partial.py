"""Learnable partial C_N/SE(2)-style group convolutions."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def rotate_kernel(kernel: torch.Tensor, angle_radians: float) -> torch.Tensor:
    """Differentiably rotate the last two kernel dimensions."""
    flat = kernel.reshape(-1, 1, kernel.shape[-2], kernel.shape[-1])
    cosine, sine = math.cos(angle_radians), math.sin(angle_radians)
    theta = flat.new_tensor([[cosine, -sine, 0.0], [sine, cosine, 0.0]])
    theta = theta.unsqueeze(0).expand(flat.shape[0], -1, -1)
    grid = F.affine_grid(theta, flat.shape, align_corners=True)
    rotated = F.grid_sample(flat, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return rotated.reshape_as(kernel)


def _rotation_grids(
    orientations: int,
    kernel_size: int,
) -> torch.Tensor:
    """Precompute sampling grids for all C_N filter orientations."""
    angles = [
        2 * math.pi * group_index / orientations
        for group_index in range(orientations)
    ]
    theta = torch.tensor(
        [
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
            ]
            for angle in angles
        ],
        dtype=torch.float32,
    )
    return F.affine_grid(
        theta,
        torch.Size((orientations, 1, kernel_size, kernel_size)),
        align_corners=True,
    )


def _rotate_all_orientations(
    kernel: torch.Tensor,
    rotation_grids: torch.Tensor,
) -> torch.Tensor:
    """Rotate every filter to every orientation in one grid_sample call."""
    kernel_size = kernel.shape[-1]
    flat = kernel.reshape(-1, 1, kernel_size, kernel_size)
    orientations = rotation_grids.shape[0]
    filter_count = flat.shape[0]
    expanded_filters = flat.unsqueeze(0).expand(
        orientations,
        filter_count,
        1,
        kernel_size,
        kernel_size,
    ).reshape(
        orientations * filter_count,
        1,
        kernel_size,
        kernel_size,
    )
    expanded_grids = rotation_grids[:, None].expand(
        orientations,
        filter_count,
        kernel_size,
        kernel_size,
        2,
    ).reshape(
        orientations * filter_count,
        kernel_size,
        kernel_size,
        2,
    )
    rotated = F.grid_sample(
        expanded_filters,
        expanded_grids,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return rotated.reshape(orientations, *kernel.shape)


class CyclicLiftingConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, orientations: int = 16) -> None:
        super().__init__()
        self.orientations = orientations
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 3, 3))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer(
            "_rotation_grids",
            _rotation_grids(orientations, 3),
            persistent=False,
        )
        self.register_buffer(
            "_cached_kernel",
            None,
            persistent=False,
        )
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

    def _expanded_kernel(self) -> torch.Tensor:
        out_channels, in_channels, kernel_size, _ = self.weight.shape
        rotated = _rotate_all_orientations(
            self.weight,
            self._rotation_grids,
        )
        kernel = rotated.reshape(
            self.orientations * out_channels,
            in_channels,
            kernel_size,
            kernel_size,
        )
        return kernel

    def train(self, mode: bool = True):
        # Rebuild inference kernels whenever the module changes mode.  In
        # particular, evaluate() calls eval() after loading the best checkpoint;
        # retaining a kernel cached for the final epoch would otherwise mix two
        # different parameter states during test evaluation.
        self._cached_kernel = None
        return super().train(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            kernel = self._expanded_kernel()
        else:
            if self._cached_kernel is None:
                self._cached_kernel = self._expanded_kernel().detach()
            kernel = self._cached_kernel
        output = F.conv2d(
            x,
            kernel,
            self.bias.repeat(self.orientations),
            padding=1,
        )
        batch, _channels, height, width = output.shape
        return output.reshape(
            batch, self.orientations, self.weight.shape[0], height, width
        ).permute(0, 2, 1, 3, 4)


class PartialCyclicGroupConv2d(nn.Module):
    """C_N group convolution with learnable gates over relative orientations."""

    def __init__(self, in_channels: int, out_channels: int, orientations: int = 16) -> None:
        super().__init__()
        self.orientations = orientations
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, orientations, 3, 3))
        self.gate_logits = nn.Parameter(torch.full((orientations,), 2.0))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer(
            "_rotation_grids",
            _rotation_grids(orientations, 3),
            persistent=False,
        )
        self.register_buffer(
            "_relative_indices",
            torch.tensor(
                [
                    [
                        (input_group - output_group) % orientations
                        for input_group in range(orientations)
                    ]
                    for output_group in range(orientations)
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_cached_kernel",
            None,
            persistent=False,
        )
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

    @property
    def gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logits)

    def _expanded_kernel(self) -> torch.Tensor:
        out_channels, in_channels, _group, kernel_size, _ = self.weight.shape
        rotated = _rotate_all_orientations(
            self.weight,
            self._rotation_grids,
        )
        gather_indices = self._relative_indices[
            :, None, None, :, None, None
        ].expand(
            self.orientations,
            out_channels,
            in_channels,
            self.orientations,
            kernel_size,
            kernel_size,
        )
        selected = rotated.gather(3, gather_indices)
        gate_weights = self.gates[self._relative_indices][
            :, None, None, :, None, None
        ]
        selected = selected * gate_weights
        kernel = selected.permute(0, 1, 3, 2, 4, 5).reshape(
            self.orientations * out_channels,
            self.orientations * in_channels,
            kernel_size,
            kernel_size,
        )
        return kernel

    def train(self, mode: bool = True):
        self._cached_kernel = None
        return super().train(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[2] != self.orientations:
            raise ValueError(
                "Cyclic feature map must have shape [B,C,orientations,H,W]."
            )
        if self.training:
            kernel = self._expanded_kernel()
        else:
            if self._cached_kernel is None:
                self._cached_kernel = self._expanded_kernel().detach()
            kernel = self._cached_kernel

        batch, in_channels, _group, height, width = x.shape
        flat_input = x.permute(0, 2, 1, 3, 4).reshape(
            batch, self.orientations * in_channels, height, width
        )
        output = F.conv2d(
            flat_input,
            kernel,
            self.bias.repeat(self.orientations),
            padding=1,
        )
        _batch, _channels, out_height, out_width = output.shape
        return output.reshape(
            batch,
            self.orientations,
            self.weight.shape[0],
            out_height,
            out_width,
        ).permute(0, 2, 1, 3, 4)

    def regularization_loss(self) -> torch.Tensor:
        # L1 promotes a smaller learned subgroup; entropy term sharpens decisions.
        gates = self.gates
        entropy = -(gates * torch.log(gates + 1e-8) + (1 - gates) * torch.log(1 - gates + 1e-8))
        return 0.5 * gates.mean() + 0.5 * entropy.mean()


class PartialGroupBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, orientations: int) -> None:
        super().__init__()
        self.conv = PartialCyclicGroupConv2d(in_channels, out_channels, orientations)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, group, height, width = x.shape
        x = self.conv(x)
        x = x.permute(0, 2, 1, 3, 4).reshape(batch * group, x.shape[1], height, width)
        x = F.relu(self.norm(x), inplace=True)
        x = F.avg_pool2d(x, 2)
        return x.reshape(batch, group, x.shape[1], x.shape[-2], x.shape[-1]).permute(0, 2, 1, 3, 4)


class PartialSE2CNN(nn.Module):
    """Finite C_N approximation of the partial continuous SE(2) model."""

    def __init__(
        self,
        orientations: int = 16,
        channels: tuple[int, ...] | list[int] = (12, 20, 32),
        gate_regularization: float = 1e-4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if orientations < 4:
            raise ValueError("At least four orientations are required.")
        channels = tuple(int(value) for value in channels)
        self.orientations = orientations
        self.gate_regularization_weight = gate_regularization
        self.lift = CyclicLiftingConv2d(3, channels[0], orientations)
        blocks: list[nn.Module] = []
        for in_channels, out_channels in zip(channels[:-1], channels[1:], strict=True):
            blocks.append(PartialGroupBlock(in_channels, out_channels, orientations))
        self.blocks = nn.Sequential(*blocks)
        self.feature_dim = channels[-1]
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels[-1], 1))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.lift(x), inplace=True)
        x = self.blocks(x)
        return x.mean(dim=(2, 3, 4))

    def regularization_loss(self) -> torch.Tensor:
        penalties = [
            module.regularization_loss()
            for module in self.modules()
            if isinstance(module, PartialCyclicGroupConv2d)
        ]
        if not penalties:
            return next(self.parameters()).new_zeros(())
        return self.gate_regularization_weight * torch.stack(penalties).mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))
