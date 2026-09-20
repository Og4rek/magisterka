"""Dependency-light attribution methods implemented directly in PyTorch."""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from ..models.d4 import d4_inverse, transform_kernel


def normalize_saliency(saliency: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize each non-negative map independently to the [0, 1] interval."""
    if saliency.ndim == 2:
        saliency = saliency.unsqueeze(0)
    flattened = saliency.flatten(1)
    minimum = flattened.min(dim=1).values[:, None, None]
    shifted = saliency - minimum
    maximum = shifted.flatten(1).max(dim=1).values[:, None, None]
    return shifted / maximum.clamp_min(eps)


def _target_signs(
    logits: torch.Tensor,
    target_classes: torch.Tensor | None,
) -> torch.Tensor:
    if target_classes is None:
        target_classes = (logits.detach().reshape(-1) >= 0).long()
    target_classes = target_classes.to(logits.device).reshape(-1)
    if target_classes.shape[0] != logits.reshape(-1).shape[0]:
        raise ValueError("One binary target class is required per input image.")
    return target_classes.float().mul(2).sub(1)


def _class_scores(
    logits: torch.Tensor,
    target_classes: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    flattened = logits.reshape(-1)
    signs = _target_signs(flattened, target_classes)
    return flattened * signs, signs


def resolve_target_layer(model: nn.Module, dotted_name: str | None = None) -> nn.Module:
    """Resolve a Grad-CAM layer for every architecture used in the thesis."""
    if dotted_name:
        module: nn.Module = model
        for component in dotted_name.split("."):
            module = getattr(module, component)
        return module
    if hasattr(model, "fusion_activation"):
        return model.fusion_activation
    if hasattr(model, "backbone"):
        backbone = model.backbone
        if hasattr(backbone, "layer4"):
            return backbone.layer4
        if hasattr(backbone, "features"):
            return backbone.features
        if hasattr(backbone, "Mixed_7c"):
            return backbone.Mixed_7c
    if hasattr(model, "norm") and model.__class__.__name__ == "GDenseNetD4":
        return model.norm
    if hasattr(model, "stem"):
        return model.stem
    raise ValueError(
        f"No automatic Grad-CAM layer for {model.__class__.__name__}; "
        "pass an explicit dotted module name."
    )


def _activation_to_cam(
    activation: torch.Tensor,
    gradient: torch.Tensor,
) -> torch.Tensor:
    if activation.ndim == 5:  # [B,C,G,H,W] D4 feature map
        weights = gradient.mean(dim=(2, 3, 4), keepdim=True)
        return (weights * activation).sum(dim=1).mean(dim=1)
    if activation.ndim == 4:
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        return (weights * activation).sum(dim=1)
    if activation.ndim == 3:  # Transformer tokens without a spatial hook.
        token_count = activation.shape[1]
        if int(math.isqrt(token_count)) ** 2 != token_count:
            activation = activation[:, 1:]
            gradient = gradient[:, 1:]
            token_count -= 1
        side = int(math.isqrt(token_count))
        if side * side != token_count:
            raise ValueError("Transformer token count is not a square grid.")
        weights = gradient.mean(dim=1, keepdim=True)
        token_cam = (weights * activation).sum(dim=2)
        return token_cam.reshape(activation.shape[0], side, side)
    raise ValueError(f"Unsupported Grad-CAM activation rank: {activation.ndim}")


def gradcam(
    model: nn.Module,
    inputs: torch.Tensor,
    target_classes: torch.Tensor | None = None,
    target_layer: nn.Module | None = None,
) -> torch.Tensor:
    """Compute class-conditional Grad-CAM maps for a batch."""
    model.eval()
    activation: torch.Tensor | None = None

    def capture(_module: nn.Module, _args: tuple[torch.Tensor, ...], output: torch.Tensor):
        nonlocal activation
        activation = output

    layer = target_layer or resolve_target_layer(model)
    handle = layer.register_forward_hook(capture)
    try:
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            logits = model(inputs)
            scores, _signs = _class_scores(logits, target_classes)
            if activation is None:
                raise RuntimeError("The Grad-CAM target layer did not execute.")
            gradient = torch.autograd.grad(scores.sum(), activation)[0]
            maps = F.relu(_activation_to_cam(activation, gradient))
            maps = F.interpolate(
                maps[:, None],
                size=inputs.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0]
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
    return normalize_saliency(maps.detach())


def integrated_gradients(
    model: nn.Module,
    inputs: torch.Tensor,
    target_classes: torch.Tensor | None = None,
    baselines: torch.Tensor | None = None,
    steps: int = 32,
    alpha_batch_size: int = 8,
) -> torch.Tensor:
    """Compute midpoint-rule Integrated Gradients in bounded GPU batches."""
    if steps <= 0 or alpha_batch_size <= 0:
        raise ValueError("steps and alpha_batch_size must be positive.")
    model.eval()
    baselines = torch.zeros_like(inputs) if baselines is None else baselines
    if baselines.shape != inputs.shape:
        baselines = torch.broadcast_to(baselines, inputs.shape)
    delta = inputs - baselines
    with torch.no_grad():
        logits = model(inputs)
        signs = _target_signs(logits, target_classes)
    accumulated = torch.zeros_like(inputs, dtype=torch.float32)
    alphas = (torch.arange(steps, device=inputs.device, dtype=torch.float32) + 0.5) / steps
    for alpha_chunk in alphas.split(alpha_batch_size):
        scaled = (
            baselines[None]
            + alpha_chunk[:, None, None, None, None] * delta[None]
        ).flatten(0, 1)
        scaled.requires_grad_(True)
        logits = model(scaled).reshape(-1)
        repeated_signs = signs.repeat(alpha_chunk.shape[0])
        gradients = torch.autograd.grad((logits * repeated_signs).sum(), scaled)[0]
        accumulated += gradients.reshape(
            alpha_chunk.shape[0], *inputs.shape
        ).sum(dim=0).float()
    attribution = delta.float() * accumulated / steps
    maps = F.relu(attribution.sum(dim=1))
    model.zero_grad(set_to_none=True)
    return normalize_saliency(maps.detach())


def generate_rise_masks(
    count: int,
    height: int,
    width: int,
    grid_size: int = 7,
    probability: float = 0.5,
    seed: int = 17,
) -> torch.Tensor:
    if count <= 0 or grid_size <= 1 or not 0 < probability < 1:
        raise ValueError("Invalid RISE mask parameters.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    coarse = (
        torch.rand(count, 1, grid_size, grid_size, generator=generator)
        < probability
    ).float()
    return F.interpolate(
        coarse,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )


@torch.inference_mode()
def rise(
    model: nn.Module,
    inputs: torch.Tensor,
    target_classes: torch.Tensor | None = None,
    baselines: torch.Tensor | None = None,
    mask_count: int = 1024,
    mask_batch_size: int = 128,
    grid_size: int = 7,
    probability: float = 0.5,
    seed: int = 17,
) -> torch.Tensor:
    """Compute black-box RISE maps without materializing all masked images."""
    model.eval()
    baselines = torch.zeros_like(inputs) if baselines is None else baselines
    if baselines.shape != inputs.shape:
        baselines = torch.broadcast_to(baselines, inputs.shape)
    logits = model(inputs)
    signs = _target_signs(logits, target_classes)
    cpu_masks = generate_rise_masks(
        mask_count,
        inputs.shape[-2],
        inputs.shape[-1],
        grid_size,
        probability,
        seed,
    )
    maps: list[torch.Tensor] = []
    for image, baseline, sign in zip(inputs, baselines, signs, strict=True):
        weighted = torch.zeros(inputs.shape[-2:], device=inputs.device)
        for mask_chunk in cpu_masks.split(mask_batch_size):
            masks = mask_chunk.to(inputs.device, non_blocking=True)
            masked = baseline[None] + masks * (image[None] - baseline[None])
            scores = torch.sigmoid(model(masked).reshape(-1) * sign)
            weighted += torch.einsum("n,nhw->hw", scores, masks[:, 0])
        maps.append(weighted / (mask_count * probability))
    return normalize_saliency(torch.stack(maps).detach())


def native_evidence(
    model: nn.Module,
    inputs: torch.Tensor,
    target_classes: torch.Tensor | None = None,
) -> torch.Tensor:
    evidence_function = getattr(model, "evidence_map", None)
    if not callable(evidence_function):
        raise TypeError(f"{model.__class__.__name__} has no native evidence map.")
    with torch.no_grad():
        logits = model(inputs)
        signs = _target_signs(logits, target_classes)
        evidence = evidence_function(inputs) * signs[:, None, None]
        evidence = F.relu(evidence)
        evidence = F.interpolate(
            evidence[:, None],
            size=inputs.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[:, 0]
    return normalize_saliency(evidence)


def apply_d4(tensor: torch.Tensor, group_index: int) -> torch.Tensor:
    """Apply the same D4 convention as the equivariant convolution kernels."""
    return transform_kernel(tensor, group_index)


def undo_d4(tensor: torch.Tensor, group_index: int) -> torch.Tensor:
    return apply_d4(tensor, d4_inverse(group_index))


def reset_parameters(module: nn.Module) -> None:
    """Reinitialize a module, including custom layers without a reset hook."""
    reset = getattr(module, "reset_parameters", None)
    if callable(reset):
        reset()
        return
    for parameter in module.parameters(recurse=False):
        if parameter.ndim >= 2:
            nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
        elif parameter.ndim == 1:
            nn.init.zeros_(parameter)
        else:
            # A zero scalar scale would collapse all logits to a constant.
            nn.init.normal_(parameter, mean=1.0, std=0.1)


def randomized_model(
    model: nn.Module,
    scope: str,
    seed: int = 17,
) -> nn.Module:
    """Deep-copy and randomize either the classifier or every learned module."""
    if scope not in {"classifier", "full"}:
        raise ValueError("scope must be classifier or full.")
    torch.manual_seed(seed)
    randomized = copy.deepcopy(model).cpu()
    if scope == "classifier":
        candidates: Iterable[nn.Module] = (
            module
            for name in ("classifier", "evidence_head")
            if (module := getattr(randomized, name, None)) is not None
        )
        for name in ("logit_scale", "logit_bias"):
            parameter = getattr(randomized, name, None)
            if isinstance(parameter, nn.Parameter):
                nn.init.normal_(parameter, std=0.1)
    else:
        candidates = randomized.modules()
    for module in candidates:
        reset_parameters(module)
    return randomized
