from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pcam_repro.config import load_config
from pcam_repro.metrics import average_precision, binary_metrics, roc_auc


def test_every_shipped_configuration_loads() -> None:
    config_directory = Path(__file__).parents[1] / "configs"
    paths = sorted(config_directory.glob("*.toml"))
    assert len(paths) >= 17
    for path in paths:
        config = load_config(path)
        assert config.model.name
        assert config.training.epochs > 0
        if path.name == "e07_gdensenet_d4.toml":
            assert config.data.batch_size == 32
            assert config.data.eval_batch_size == 16


def test_metrics_on_perfect_predictions() -> None:
    targets = np.array([0, 0, 1, 1])
    probabilities = np.array([0.01, 0.2, 0.8, 0.99])
    metrics = binary_metrics(targets, probabilities)
    assert metrics["accuracy"] == 1.0
    assert metrics["auc_roc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert roc_auc(targets, np.full(4, 0.5)) == 0.5


def test_average_precision_groups_ties_and_is_order_independent() -> None:
    targets = np.array([1, 0, 1, 0])
    scores = np.array([0.8, 0.8, 0.2, 0.2])
    for order in ([0, 1, 2, 3], [1, 0, 3, 2], [3, 2, 1, 0]):
        assert average_precision(targets[order], scores[order]) == 0.5
    assert average_precision(targets, np.ones(4)) == 0.5
    # A first, pure-positive threshold and a later mixed group:
    assert average_precision(np.array([1, 0, 1]), np.array([0.9, 0.5, 0.5])) == pytest.approx(5 / 6)


def test_ranking_metrics_do_not_clip_distinct_extreme_scores() -> None:
    targets = np.array([0, 1])
    scores = np.array([1e-10, 2e-10])
    assert roc_auc(targets, scores) == 1.0
    assert average_precision(targets, scores) == 1.0
    assert np.isfinite(binary_metrics(targets, np.array([0.0, 1.0]))["nll"])
