"""Explainability and tumor-localization utilities for PCam classifiers."""

from .attribution import (
    gradcam,
    integrated_gradients,
    native_evidence,
    rise,
)
from .masks import CamelyonMaskProvider, PCamMetadata
from .metrics import (
    center_mass_ratio,
    localization_metrics,
    map_similarity,
    perturbation_faithfulness,
)

__all__ = [
    "CamelyonMaskProvider",
    "PCamMetadata",
    "center_mass_ratio",
    "gradcam",
    "integrated_gradients",
    "localization_metrics",
    "map_similarity",
    "native_evidence",
    "perturbation_faithfulness",
    "rise",
]
