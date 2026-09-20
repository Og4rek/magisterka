"""Classification, label-noise, and contrastive objectives."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = logits.reshape(-1), targets.float().reshape(-1)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probability = torch.sigmoid(logits)
        p_t = probability * targets + (1 - probability) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t).pow(self.gamma) * bce).mean()


class SymmetricCrossEntropy(nn.Module):
    """Binary SCE = alpha CE + beta reverse CE."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, epsilon: float = 1e-4) -> None:
        super().__init__()
        self.alpha, self.beta, self.epsilon = alpha, beta, epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = logits.reshape(-1), targets.float().reshape(-1)
        ce = F.binary_cross_entropy_with_logits(logits, targets)
        probability = torch.sigmoid(logits).clamp(self.epsilon, 1 - self.epsilon)
        clipped_target = targets.clamp(self.epsilon, 1 - self.epsilon)
        reverse_ce = -(
            probability * torch.log(clipped_target)
            + (1 - probability) * torch.log(1 - clipped_target)
        ).mean()
        return self.alpha * ce + self.beta * reverse_ce


class GeneralizedCrossEntropy(nn.Module):
    """Noise-robust L_q objective; q -> 0 approaches ordinary cross-entropy."""

    def __init__(self, q: float = 0.7) -> None:
        super().__init__()
        if not 0 < q <= 1:
            raise ValueError("GCE q must be in (0,1].")
        self.q = q

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = logits.reshape(-1), targets.float().reshape(-1)
        probability = torch.sigmoid(logits)
        p_correct = probability * targets + (1 - probability) * (1 - targets)
        return ((1 - p_correct.clamp_min(1e-7).pow(self.q)) / self.q).mean()


class NTXentLoss(nn.Module):
    """Normalized temperature-scaled cross entropy for paired views."""

    def __init__(self, temperature: float = 0.2) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.temperature = temperature

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if first.shape != second.shape:
            raise ValueError("Contrastive views must have equal embedding shapes.")
        batch = first.shape[0]
        embeddings = F.normalize(torch.cat((first, second), dim=0), dim=1)
        similarity = embeddings @ embeddings.T / self.temperature
        # Avoid allocating and applying a full [2B,2B] boolean identity mask
        # at every SimCLR step. The in-place diagonal update has the same
        # mathematical effect and preserves zero gradient for self-similarity.
        similarity.fill_diagonal_(torch.finfo(similarity.dtype).min)
        targets = torch.arange(2 * batch, device=similarity.device)
        targets = (targets + batch) % (2 * batch)
        return F.cross_entropy(similarity, targets)


def build_classification_loss(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "bce":
        return nn.BCEWithLogitsLoss()
    if normalized == "focal":
        return BinaryFocalLoss()
    if normalized in {"sce", "symmetric_cross_entropy"}:
        return SymmetricCrossEntropy()
    if normalized in {"gce", "generalized_cross_entropy"}:
        return GeneralizedCrossEntropy()
    raise KeyError(f"Unknown robust loss '{name}'.")
