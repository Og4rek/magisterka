"""Dependency-light binary classification metrics for PCam."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_arrays(targets: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(targets, dtype=np.int64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if y.shape != p.shape:
        raise ValueError("Targets and probabilities must have the same shape.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Binary targets must contain only 0 and 1.")
    if not np.isfinite(p).all():
        raise ValueError("Scores must be finite.")
    return y, p


def roc_auc(targets: np.ndarray, probabilities: np.ndarray) -> float:
    y, p = _as_arrays(targets, probabilities)
    positives = y.sum()
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    ranks = np.empty(len(p), dtype=np.float64)
    start = 0
    while start < len(p):
        end = start + 1
        while end < len(p) and sorted_p[end] == sorted_p[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = ranks[y == 1].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def average_precision(targets: np.ndarray, probabilities: np.ndarray) -> float:
    y, p = _as_arrays(targets, probabilities)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    sorted_y = y[order]
    # All equal scores cross the decision threshold together. Evaluating each
    # item separately makes AP depend on the input order, notably for saliency
    # maps with large constant regions.
    endpoints = np.r_[np.flatnonzero(np.diff(p[order])), len(y) - 1]
    tp = np.cumsum(sorted_y)[endpoints]
    precision = tp / (endpoints + 1)
    recall_increments = np.diff(np.r_[0, tp]) / positives
    return float(np.sum(recall_increments * precision))


def expected_calibration_error(
    targets: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 15,
) -> float:
    y, p = _as_arrays(targets, probabilities)
    predictions = (p >= 0.5).astype(np.int64)
    confidence = np.maximum(p, 1 - p)
    edges = np.linspace(0.5, 1.0, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (confidence >= left) & (confidence < right if right < 1 else confidence <= right)
        if mask.any():
            accuracy = (predictions[mask] == y[mask]).mean()
            ece += mask.mean() * abs(accuracy - confidence[mask].mean())
    return float(ece)


def binary_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y, p = _as_arrays(targets, probabilities)
    pred = (p >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    f1 = 2 * precision * sensitivity / max(1e-12, precision + sensitivity)
    log_p = np.clip(p, 1e-7, 1 - 1e-7)
    nll = -np.mean(y * np.log(log_p) + (1 - y) * np.log(1 - log_p))
    return {
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "auc_roc": roc_auc(y, p),
        "average_precision": average_precision(y, p),
        "f1": float(f1),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "nll": float(nll),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": expected_calibration_error(y, p),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }
