"""Native D4-equivariant convolutions and a DenseNet-style classifier.

The implementation uses the eight rotations/reflections of the square without
an external group-equivariance package. Feature tensors have shape B,C,8,H,W.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


D4_ORDER = 8


def _element(index: int) -> tuple[int, int]:
    return index % 4, index // 4


def _index(rotation: int, reflection: int) -> int:
    return (rotation % 4) + 4 * (reflection % 2)


def d4_inverse(index: int) -> int:
    rotation, reflection = _element(index)
    inverse_rotation = (-rotation if reflection == 0 else rotation) % 4
    return _index(inverse_rotation, reflection)


def d4_multiply(left: int, right: int) -> int:
    r_left, f_left = _element(left)
    r_right, f_right = _element(right)
    rotation = (r_left + (-1 if f_left else 1) * r_right) % 4
    return _index(rotation, f_left ^ f_right)


def transform_kernel(kernel: torch.Tensor, group_index: int) -> torch.Tensor:
    """Apply g=R^r F^f to the final two dimensions of a kernel."""
    rotation, reflection = _element(group_index)
    if reflection:
        kernel = torch.flip(kernel, dims=(-1,))
    return torch.rot90(kernel, rotation, dims=(-2, -1))


def _spatial_permutations(kernel_size: int) -> torch.Tensor:
    """Return source-pixel indices for every D4 spatial transformation."""
    source = torch.arange(kernel_size * kernel_size).reshape(
        kernel_size, kernel_size
    )
    return torch.stack(
        [
            transform_kernel(source, group_index).reshape(-1)
            for group_index in range(D4_ORDER)
        ]
    )


class D4LiftingConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer(
            "_spatial_indices",
            _spatial_permutations(kernel_size),
            persistent=False,
        )
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_channels, in_channels, kernel_size, _ = self.weight.shape
        flat_weight = self.weight.flatten(-2)
        expanded = flat_weight[:, :, None, :].expand(
            out_channels,
            in_channels,
            D4_ORDER,
            kernel_size * kernel_size,
        )
        spatial_indices = self._spatial_indices[None, None, :, :].expand_as(
            expanded
        )
        transformed = expanded.gather(-1, spatial_indices)
        kernel = transformed.permute(2, 0, 1, 3).reshape(
            D4_ORDER * out_channels,
            in_channels,
            kernel_size,
            kernel_size,
        )
        bias = self.bias.repeat(D4_ORDER)
        output = F.conv2d(x, kernel, bias, padding=self.padding)
        batch, _channels, height, width = output.shape
        return output.reshape(
            batch, D4_ORDER, self.weight.shape[0], height, width
        ).permute(0, 2, 1, 3, 4)


class D4GroupConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, D4_ORDER, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.register_buffer(
            "_relative_indices",
            torch.tensor(
                [
                    [
                        d4_multiply(d4_inverse(g), h)
                        for h in range(D4_ORDER)
                    ]
                    for g in range(D4_ORDER)
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_spatial_indices",
            _spatial_permutations(kernel_size),
            persistent=False,
        )
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[2] != D4_ORDER:
            raise ValueError("D4 feature map must have shape [B,C,8,H,W].")

        # Expand the complete structured D4 filter without Python loops.
        # [O,I,R,K,K] -> [O,I,G,H,K*K] -> [G*O,H*I,K,K].
        out_channels, in_channels, _group, kernel_size, _ = self.weight.shape
        flat_weight = self.weight.flatten(-2)
        selected = flat_weight[:, :, self._relative_indices, :]
        spatial_indices = self._spatial_indices[
            None, None, :, None, :
        ].expand_as(selected)
        transformed = selected.gather(-1, spatial_indices)
        kernel = transformed.permute(2, 0, 3, 1, 4).reshape(
            D4_ORDER * out_channels,
            D4_ORDER * in_channels,
            kernel_size,
            kernel_size,
        )

        batch, in_channels, _group, height, width = x.shape
        flat_input = x.permute(0, 2, 1, 3, 4).reshape(
            batch, D4_ORDER * in_channels, height, width
        )
        bias = self.bias.repeat(D4_ORDER) if self.bias is not None else None
        output = F.conv2d(flat_input, kernel, bias, padding=self.padding)
        _batch, _channels, out_height, out_width = output.shape
        return output.reshape(
            batch, D4_ORDER, self.weight.shape[0], out_height, out_width
        ).permute(0, 2, 1, 3, 4)


class GroupBatchNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, group, height, width = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(batch * group, channels, height, width)
        y = self.norm(y)
        return y.reshape(batch, group, channels, height, width).permute(0, 2, 1, 3, 4)


class D4DenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int) -> None:
        super().__init__()
        self.norm = GroupBatchNorm(in_channels)
        self.conv = D4GroupConv2d(in_channels, growth_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.conv(F.relu(self.norm(x), inplace=True))
        return torch.cat((x, new_features), dim=1)


class D4Transition(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm = GroupBatchNorm(in_channels)
        self.conv = D4GroupConv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(F.relu(self.norm(x), inplace=True))
        batch, channels, group, height, width = x.shape
        y = x.reshape(batch, channels * group, height, width)
        y = F.avg_pool2d(y, 2)
        return y.reshape(batch, channels, group, y.shape[-2], y.shape[-1])


class GDenseNetD4(nn.Module):
    """D4 G-DenseNet suitable for the rotation-equivariant PCam experiment."""

    def __init__(
        self,
        initial_channels: int = 16,
        growth_rate: int = 8,
        block_layers: tuple[int, ...] | list[int] = (3, 4, 5),
        compression: float = 0.5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lift = D4LiftingConv2d(3, initial_channels)
        channels = initial_channels
        blocks: list[nn.Module] = []
        for block_index, number_layers in enumerate(block_layers):
            for _ in range(int(number_layers)):
                blocks.append(D4DenseLayer(channels, growth_rate))
                channels += growth_rate
            if block_index != len(block_layers) - 1:
                out_channels = max(initial_channels, int(channels * compression))
                blocks.append(D4Transition(channels, out_channels))
                channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.norm = GroupBatchNorm(channels)
        self.dropout = nn.Dropout(dropout)
        self.feature_dim = channels
        self.classifier = nn.Linear(channels, 1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(self.lift(x))
        x = F.relu(self.norm(x), inplace=True)
        return x.mean(dim=(2, 3, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.forward_features(x)))
