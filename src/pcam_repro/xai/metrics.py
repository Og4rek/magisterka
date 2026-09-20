"""Quantitative faithfulness, localization, and stability metrics for XAI."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from ..metrics import average_precision


def _numpy_map(values: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().float().cpu().numpy()
    return np.asarray(values, dtype=np.float64).squeeze()


def center_mass_ratio(
    saliency: torch.Tensor | np.ndarray,
    center_fraction: float = 1.0 / 3.0,
) -> float:
    values = np.maximum(_numpy_map(saliency), 0)
    height, width = values.shape
    center_height = max(1, round(height * center_fraction))
    center_width = max(1, round(width * center_fraction))
    if (height - center_height) % 2:
        center_height = min(height, center_height + 1)
    if (width - center_width) % 2:
        center_width = min(width, center_width + 1)
    top = (height - center_height) // 2
    left = (width - center_width) // 2
    denominator = values.sum()
    if denominator <= 0:
        return float("nan")
    return float(
        values[top : top + center_height, left : left + center_width].sum()
        / denominator
    )


def _binary_overlap(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    prediction_size = prediction.sum()
    target_size = target.sum()
    dice = 2 * intersection / max(1, prediction_size + target_size)
    iou = intersection / max(1, union)
    return float(dice), float(iou)


def localization_metrics(
    saliency: torch.Tensor | np.ndarray,
    tumor_mask: torch.Tensor | np.ndarray,
    thresholds: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5),
) -> dict[str, float]:
    values = np.maximum(_numpy_map(saliency), 0)
    mask = _numpy_map(tumor_mask) > 0.5
    if values.shape != mask.shape:
        raise ValueError("Saliency and tumor mask must have identical shapes.")
    if not mask.any():
        return {
            "mass_inside": float("nan"),
            "pointing_game": float("nan"),
            "pixel_auprc": float("nan"),
            **{
                f"{metric}_at_{threshold:.1f}": float("nan")
                for threshold in thresholds
                for metric in ("dice", "iou")
            },
        }
    total = values.sum()
    peak = np.unravel_index(int(values.argmax()), values.shape)
    result = {
        "mass_inside": float(values[mask].sum() / total) if total > 0 else float("nan"),
        "pointing_game": float(mask[peak]),
        "pixel_auprc": average_precision(mask.astype(np.int64), values),
    }
    maximum = values.max()
    for threshold in thresholds:
        selected = values >= maximum * threshold if maximum > 0 else np.zeros_like(mask)
        dice, iou = _binary_overlap(selected, mask)
        result[f"dice_at_{threshold:.1f}"] = dice
        result[f"iou_at_{threshold:.1f}"] = iou
    return result


def map_similarity(
    reference: torch.Tensor | np.ndarray,
    candidate: torch.Tensor | np.ndarray,
    top_fraction: float = 0.2,
) -> dict[str, float]:
    first = _numpy_map(reference)
    second = _numpy_map(candidate)
    if first.shape != second.shape:
        raise ValueError("Compared saliency maps must have identical shapes.")
    first_flat = first.reshape(-1)
    second_flat = second.reshape(-1)
    first_centered = first_flat - first_flat.mean()
    second_centered = second_flat - second_flat.mean()
    denominator = np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    correlation = (
        float(np.dot(first_centered, second_centered) / denominator)
        if denominator > 0
        else float("nan")
    )
    cosine_denominator = np.linalg.norm(first_flat) * np.linalg.norm(second_flat)
    cosine = (
        float(np.dot(first_flat, second_flat) / cosine_denominator)
        if cosine_denominator > 0
        else float("nan")
    )
    count = max(1, round(first_flat.size * top_fraction))
    first_top = np.argpartition(first_flat, -count)[-count:]
    second_top = np.argpartition(second_flat, -count)[-count:]
    intersection = len(np.intersect1d(first_top, second_top, assume_unique=False))
    union = len(np.union1d(first_top, second_top))
    return {
        "pearson": correlation,
        "cosine": cosine,
        "top_iou": float(intersection / max(1, union)),
        "mae": float(np.abs(first - second).mean()),
    }


@torch.inference_mode()
def perturbation_faithfulness(
    model: nn.Module,
    image: torch.Tensor,
    saliency: torch.Tensor,
    target_class: int,
    baseline: torch.Tensor | None = None,
    steps: int = 20,
    batch_size: int = 64,
) -> dict[str, float]:
    """Compute deletion/insertion AUC with simultaneous batched perturbations."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.shape[0] != 1:
        raise ValueError("Faithfulness evaluation accepts one image at a time.")
    baseline = torch.zeros_like(image) if baseline is None else baseline
    values = saliency.reshape(-1)
    pixel_count = values.numel()
    order = torch.argsort(values, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(pixel_count, device=order.device)
    fractions = torch.linspace(0, 1, steps + 1, device=image.device)
    counts = torch.round(fractions * pixel_count).long()
    selected = ranks[None] < counts[:, None]
    selected = selected.reshape(steps + 1, 1, image.shape[-2], image.shape[-1])
    delta = image - baseline
    deletion = image - selected * delta
    insertion = baseline + selected * delta
    variants = torch.cat((deletion, insertion), dim=0)
    probabilities: list[torch.Tensor] = []
    sign = 1.0 if int(target_class) == 1 else -1.0
    for chunk in variants.split(batch_size):
        probabilities.append(torch.sigmoid(model(chunk).reshape(-1) * sign))
    scores = torch.cat(probabilities)
    deletion_scores, insertion_scores = scores.split(steps + 1)
    x = fractions.float()
    return {
        "deletion_auc": float(torch.trapezoid(deletion_scores.float(), x).cpu()),
        "insertion_auc": float(torch.trapezoid(insertion_scores.float(), x).cpu()),
        "deletion_drop": float((deletion_scores[0] - deletion_scores[-1]).cpu()),
        "insertion_gain": float((insertion_scores[-1] - insertion_scores[0]).cpu()),
    }
