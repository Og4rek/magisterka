from __future__ import annotations

import copy
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.data import DataLoader, TensorDataset

from pcam_repro.config import ExperimentConfig, ModelConfig, TrainingConfig
from pcam_repro.engine import _flush_pending_losses, fit
from pcam_repro.models import build_model
from pcam_repro.utils import update_ema


def _classification_loader() -> DataLoader:
    images = torch.randn(8, 3, 32, 32)
    targets = torch.tensor([0.0, 1.0] * 4)
    identifiers = torch.arange(8)
    return DataLoader(TensorDataset(images, targets, identifiers), batch_size=4)


def test_one_epoch_supervised_fit_writes_auditable_outputs(tmp_path: Path) -> None:
    config = ExperimentConfig(
        model=ModelConfig(name="small_cnn", parameters={"width": 2}),
        training=TrainingConfig(epochs=1, amp=False, patience=1),
    )
    loader = _classification_loader()
    result = fit(
        build_model(config.model),
        {"train": loader, "validation": loader, "test": loader},
        config,
        torch.device("cpu"),
        tmp_path,
    )
    assert "test" in result and "auc_roc" in result["test"]
    for filename in ("best.pt", "last.pt", "history.json", "result.json"):
        assert (tmp_path / filename).exists()
    assert any((tmp_path / "tensorboard").glob("events.out.tfevents.*"))
    events = EventAccumulator(str(tmp_path / "tensorboard"))
    events.Reload()
    scalar_tags = set(events.Tags()["scalars"])
    assert {
        "train/loss",
        "validation/loss",
        "validation/auc_roc",
        "test/auc_roc",
    } <= scalar_tags


def test_supervised_fit_can_resume_from_last_checkpoint(tmp_path: Path) -> None:
    first_config = ExperimentConfig(
        model=ModelConfig(name="small_cnn", parameters={"width": 2}),
        training=TrainingConfig(epochs=1, amp=False, patience=5),
    )
    loader = _classification_loader()
    loaders = {"train": loader, "validation": loader, "test": loader}
    fit(
        build_model(first_config.model),
        loaders,
        first_config,
        torch.device("cpu"),
        tmp_path,
    )

    resumed_config = ExperimentConfig(
        model=ModelConfig(name="small_cnn", parameters={"width": 2}),
        training=TrainingConfig(epochs=2, amp=False, patience=5),
    )
    result = fit(
        build_model(resumed_config.model),
        loaders,
        resumed_config,
        torch.device("cpu"),
        tmp_path,
        tmp_path / "last.pt",
    )
    assert len(result["history"]) == 2
    checkpoint = torch.load(
        tmp_path / "last.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["epoch"] == 2


def test_compile_happens_after_resume_and_preserves_checkpoint_keys(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first_config = ExperimentConfig(
        model=ModelConfig(name="small_cnn", parameters={"width": 2}),
        training=TrainingConfig(epochs=1, amp=False, patience=5),
    )
    loader = _classification_loader()
    loaders = {"train": loader, "validation": loader, "test": loader}
    fit(
        build_model(first_config.model),
        loaders,
        first_config,
        torch.device("cpu"),
        tmp_path,
    )
    first_checkpoint = torch.load(
        tmp_path / "last.pt",
        map_location="cpu",
        weights_only=False,
    )
    expected_state = first_checkpoint["model"]
    compile_calls: list[dict[str, object]] = []

    def fake_compile(module, *args, **kwargs) -> None:
        actual_state = module.state_dict()
        assert actual_state.keys() == expected_state.keys()
        assert all(
            torch.equal(actual_state[key], expected_state[key])
            for key in expected_state
        )
        compile_calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(torch.nn.Module, "compile", fake_compile)
    resumed_config = ExperimentConfig(
        model=ModelConfig(name="small_cnn", parameters={"width": 2}),
        training=TrainingConfig(epochs=2, amp=False, patience=5),
    )
    fit(
        build_model(resumed_config.model),
        loaders,
        resumed_config,
        torch.device("cpu"),
        tmp_path,
        tmp_path / "last.pt",
        compile_mode="default",
    )

    assert compile_calls == [
        {"args": (), "kwargs": {"mode": "default", "dynamic": False}}
    ]
    compiled_checkpoint = torch.load(
        tmp_path / "last.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert not any(
        key.startswith("_orig_mod.") for key in compiled_checkpoint["model"]
    )
    plain_model = build_model(resumed_config.model)
    plain_model.load_state_dict(compiled_checkpoint["model"], strict=True)


def test_pending_loss_buffer_preserves_weighted_order() -> None:
    pending = [
        (torch.tensor(0.25), 4),
        (torch.tensor(0.75), 2),
        (torch.tensor(0.50), 1),
    ]
    total = _flush_pending_losses(pending, total_loss=1.0)
    assert total == 4.0
    assert pending == []


def test_foreach_ema_matches_scalar_reference() -> None:
    student = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.BatchNorm1d(4),
    )
    teacher = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.BatchNorm1d(4),
    )
    reference = copy.deepcopy(teacher)
    decay = 0.91

    with torch.no_grad():
        for reference_value, student_value in zip(
            reference.state_dict().values(),
            student.state_dict().values(),
            strict=True,
        ):
            if reference_value.is_floating_point():
                reference_value.mul_(decay).add_(
                    student_value,
                    alpha=1.0 - decay,
                )
            else:
                reference_value.copy_(student_value)

    update_ema(teacher, student, decay)
    for actual, expected in zip(
        teacher.state_dict().values(),
        reference.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(actual, expected)
