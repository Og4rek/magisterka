from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from pcam_repro.config import ModelConfig
from pcam_repro.models import build_model
from pcam_repro.xai.attribution import (
    apply_d4,
    gradcam,
    integrated_gradients,
    native_evidence,
    randomized_model,
    rise,
    undo_d4,
)
from pcam_repro.xai.masks import (
    CamelyonMaskProvider,
    PCamMetadataRecord,
    parse_camelyon_xml,
)
from pcam_repro.xai.metrics import (
    center_mass_ratio,
    localization_metrics,
    map_similarity,
    perturbation_faithfulness,
)


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs).mean(dim=(-2, -1)))


def test_attribution_methods_return_normalized_input_sized_maps() -> None:
    model = TinyClassifier().eval()
    inputs = torch.randn(2, 3, 16, 16)
    targets = torch.tensor([1, 0])
    cam = gradcam(model, inputs, targets, model.features)
    integrated = integrated_gradients(
        model,
        inputs,
        targets,
        steps=4,
        alpha_batch_size=2,
    )
    sampled = rise(
        model,
        inputs,
        targets,
        mask_count=16,
        mask_batch_size=8,
        grid_size=4,
    )
    for maps in (cam, integrated, sampled):
        assert maps.shape == (2, 16, 16)
        assert torch.isfinite(maps).all()
        assert maps.min() >= 0
        assert maps.max() <= 1


def test_native_evidence_and_d4_roundtrip() -> None:
    model = build_model(
        ModelConfig(
            name="pcam_d4loc",
            parameters={
                "initial_channels": 3,
                "growth_rate": 2,
                "stage_layers": [1, 1, 1],
                "transition_channels": [4, 6],
                "fusion_channels": 8,
                "dropout": 0.0,
            },
        )
    ).eval()
    image = torch.randn(1, 3, 32, 32)
    maps = native_evidence(model, image, torch.tensor([1]))
    assert maps.shape == (1, 32, 32)
    for group_index in range(8):
        assert torch.equal(undo_d4(apply_d4(image, group_index), group_index), image)


def test_full_randomization_changes_custom_d4_weights() -> None:
    model = build_model(ModelConfig(name="pcam_d4loc"))
    randomized = randomized_model(model, "full", seed=123)
    assert not torch.equal(model.lift.weight, randomized.lift.weight)
    assert not torch.equal(
        model.evidence_head.weight,
        randomized.evidence_head.weight,
    )


def test_localization_similarity_and_faithfulness_metrics() -> None:
    saliency = np.zeros((8, 8), dtype=np.float32)
    saliency[3:5, 3:5] = 1
    mask = saliency.copy()
    metrics = localization_metrics(saliency, mask)
    assert metrics["pointing_game"] == 1
    assert metrics["mass_inside"] == 1
    assert metrics["pixel_auprc"] == 1
    assert center_mass_ratio(saliency) == 1
    assert map_similarity(saliency, saliency)["top_iou"] == 1

    model = TinyClassifier().eval()
    image = torch.randn(3, 8, 8)
    faithfulness = perturbation_faithfulness(
        model,
        image,
        torch.from_numpy(saliency),
        target_class=1,
        steps=4,
    )
    assert set(faithfulness) == {
        "deletion_auc",
        "insertion_auc",
        "deletion_drop",
        "insertion_gain",
    }


def test_camelyon_xml_rasterization(tmp_path: Path) -> None:
    xml = """<?xml version="1.0"?>
    <ASAP_Annotations><Annotations>
      <Annotation Name="tumor" PartOfGroup="Tumor">
        <Coordinates>
          <Coordinate Order="0" X="40" Y="40" />
          <Coordinate Order="1" X="80" Y="40" />
          <Coordinate Order="2" X="80" Y="80" />
          <Coordinate Order="3" X="40" Y="80" />
        </Coordinates>
      </Annotation>
    </Annotations></ASAP_Annotations>"""
    path = tmp_path / "tumor_001.xml"
    path.write_text(xml, encoding="utf-8")
    polygons = parse_camelyon_xml(str(path.resolve()))
    assert len(polygons.positive) == 1
    provider = CamelyonMaskProvider(tmp_path, downsample=1, coordinate_mode="top_left")
    record = PCamMetadataRecord(0, 0, 0, True, True, "camelyon16_train_tumor_001")
    mask, available = provider.mask(record)
    assert available
    assert mask[50, 50] == 1
    assert provider.center_contains_tumor(mask)
