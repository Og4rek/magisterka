"""Training and evaluation loops shared by every PCam experiment."""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .config import ExperimentConfig
from .losses import NTXentLoss, build_classification_loss
from .metrics import binary_metrics
from .utils import WarmupCosineScheduler, save_checkpoint, update_ema, write_json


_LOSS_REPORT_INTERVAL = 32


def build_optimizer(model: nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    fused = bool(parameters and parameters[0].is_cuda)
    name = config.optimizer.name.lower()
    if name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
            fused=fused,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
            fused=fused,
        )
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=config.optimizer.learning_rate,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=True,
        )
    raise KeyError(f"Unknown optimizer '{config.optimizer.name}'.")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    name = config.scheduler.name.lower()
    if name == "none":
        return None
    if name == "cosine":
        minimum_ratio = config.scheduler.minimum_learning_rate / config.optimizer.learning_rate
        return WarmupCosineScheduler(
            optimizer,
            config.training.epochs,
            config.scheduler.warmup_epochs,
            minimum_ratio,
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, config.training.epochs // 3),
            gamma=0.1,
        )
    raise KeyError(f"Unknown scheduler '{config.scheduler.name}'.")


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16)
    return nullcontext()


def _model_penalty(model: nn.Module) -> torch.Tensor:
    regularizer = getattr(model, "regularization_loss", None)
    if callable(regularizer):
        return regularizer()
    return next(model.parameters()).new_zeros(())


def _tensorboard_tag(metric_name: str) -> str:
    if metric_name == "train_loss":
        return "train/loss"
    if metric_name == "learning_rate":
        return "optimizer/learning_rate"
    if metric_name.startswith("validation_"):
        return f"validation/{metric_name.removeprefix('validation_')}"
    return metric_name


def _colour_enabled() -> bool:
    return os.environ.get("NO_COLOR") is None and (
        sys.stdout.isatty() or os.environ.get("FORCE_COLOR") == "1"
    )


def _progress_colour(colour: str) -> str | None:
    return colour if _colour_enabled() else None


def _styled(text: str, ansi_code: str) -> str:
    if not _colour_enabled():
        return text
    return f"\033[{ansi_code}m{text}\033[0m"


def _flush_pending_losses(
    pending: list[tuple[torch.Tensor, int]],
    total_loss: float,
) -> float:
    """Transfer scalar losses to the CPU in small, ordered batches.

    Calling ``float(cuda_tensor)`` for every training batch forces a CUDA
    synchronization.  Buffering detached scalar tensors preserves the order
    and weighting of the reported epoch mean while allowing data loading and
    GPU execution to overlap between progress updates.
    """
    if not pending:
        return total_loss
    values = torch.stack([value for value, _batch_size in pending]).cpu().tolist()
    for value, (_loss, batch_size) in zip(values, pending, strict=True):
        total_loss += float(value) * batch_size
    pending.clear()
    return total_loss


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp: bool = True,
    predictions_path: str | Path | None = None,
    description: str = "evaluation",
    writer: SummaryWriter | None = None,
    global_step: int | None = None,
    tensorboard_prefix: str | None = None,
) -> dict[str, float]:
    model.eval()
    targets: list[np.ndarray] = []
    probabilities: list[torch.Tensor] = []
    identifiers: list[np.ndarray] = []
    progress = tqdm(
        loader,
        desc=description,
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        mininterval=1.0,
        file=sys.stdout,
        colour=_progress_colour("green" if description != "test" else "magenta"),
    )
    for images, labels, sample_ids in progress:
        images = images.to(device, non_blocking=True)
        with _autocast(device, amp):
            logits = model(images).reshape(-1)
        targets.append(labels.numpy().reshape(-1))
        probabilities.append(torch.sigmoid(logits).float())
        identifiers.append(np.asarray(sample_ids).reshape(-1))
    y = np.concatenate(targets)
    p = torch.cat(probabilities).cpu().numpy()
    metrics = binary_metrics(y, p)
    metrics["loss"] = metrics["nll"]
    if writer is not None and tensorboard_prefix is not None:
        step = global_step or 0
        writer.add_pr_curve(
            f"{tensorboard_prefix}/precision_recall_curve",
            torch.from_numpy(y),
            torch.from_numpy(p),
            global_step=step,
        )
        writer.add_histogram(
            f"{tensorboard_prefix}/predicted_probability",
            p,
            global_step=step,
        )
    if predictions_path is not None:
        target_path = Path(predictions_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("sample_id", "target", "probability"))
            writer.writerows(zip(np.concatenate(identifiers), y, p, strict=True))
    return metrics


def _backward_step(
    loss: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    is_last_step: bool,
    config: ExperimentConfig,
) -> None:
    scaled_loss = loss / config.training.accumulation_steps
    scaler.scale(scaled_loss).backward()
    if (step + 1) % config.training.accumulation_steps == 0 or is_last_step:
        scaler.unscale_(optimizer)
        if config.training.gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


def train_supervised_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: ExperimentConfig,
    epoch: int = 1,
) -> float:
    model.train()
    if config.model.freeze_encoder:
        backbone = getattr(model, "backbone", None)
        if backbone is None:
            raise ValueError(
                "freeze_encoder requires a model exposing a backbone."
            )
        # A linear-probe protocol freezes both encoder parameters and its
        # BatchNorm/dropout behavior; only the classifier remains in train mode.
        backbone.eval()
    optimizer.zero_grad(set_to_none=True)
    total_loss, observations = 0.0, 0
    pending_losses: list[tuple[torch.Tensor, int]] = []
    progress = tqdm(
        loader,
        desc=f"epoch {epoch:03d} train",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        mininterval=1.0,
        file=sys.stdout,
        colour=_progress_colour("cyan"),
    )
    for step, (images, targets, _sample_ids) in enumerate(progress):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with _autocast(device, config.training.amp):
            loss = criterion(model(images).reshape(-1), targets.reshape(-1)) + _model_penalty(model)
        _backward_step(loss, model, optimizer, scaler, step, step + 1 == len(loader), config)
        batch_size = images.shape[0]
        pending_losses.append((loss.detach(), batch_size))
        observations += batch_size
        should_report = (
            (step + 1) % _LOSS_REPORT_INTERVAL == 0
            or step + 1 == len(loader)
        )
        if should_report:
            total_loss = _flush_pending_losses(pending_losses, total_loss)
            progress.set_postfix(
                loss=f"{total_loss / observations:.5f}",
                refresh=False,
            )
    return total_loss / max(1, observations)


def train_mean_teacher_epoch(
    student: nn.Module,
    teacher: nn.Module,
    labeled_loader: torch.utils.data.DataLoader,
    unlabeled_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: ExperimentConfig,
    epoch: int = 1,
) -> float:
    student.train()
    teacher.eval()
    optimizer.zero_grad(set_to_none=True)
    total_loss, observations = 0.0, 0
    pending_losses: list[tuple[torch.Tensor, int]] = []
    labeled_iterator = iter(labeled_loader)
    progress = tqdm(
        unlabeled_loader,
        desc=f"epoch {epoch:03d} mean-teacher",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        mininterval=1.0,
        file=sys.stdout,
        colour=_progress_colour("yellow"),
    )
    for step, unlabeled_batch in enumerate(progress):
        try:
            labeled_batch = next(labeled_iterator)
        except StopIteration:
            labeled_iterator = iter(labeled_loader)
            labeled_batch = next(labeled_iterator)
        _labeled_weak, labeled_strong, targets, _ = labeled_batch
        unlabeled_weak, unlabeled_strong, _unused_targets, _ = unlabeled_batch
        labeled_strong = labeled_strong.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        unlabeled_weak = unlabeled_weak.to(device, non_blocking=True)
        unlabeled_strong = unlabeled_strong.to(device, non_blocking=True)
        with _autocast(device, config.training.amp):
            supervised = criterion(
                student(labeled_strong).reshape(-1),
                targets.reshape(-1),
            )
            student_probability = torch.sigmoid(
                student(unlabeled_strong).reshape(-1)
            )
            with torch.no_grad():
                teacher_probability = torch.sigmoid(teacher(unlabeled_weak).reshape(-1))
            consistency = nn.functional.mse_loss(student_probability, teacher_probability)
            loss = supervised + config.training.consistency_weight * consistency + _model_penalty(student)
        is_last_step = step + 1 == len(unlabeled_loader)
        _backward_step(loss, student, optimizer, scaler, step, is_last_step, config)
        if (step + 1) % config.training.accumulation_steps == 0 or is_last_step:
            update_ema(teacher, student, config.training.ema_decay)
        batch_size = labeled_strong.shape[0]
        pending_losses.append((loss.detach(), batch_size))
        observations += batch_size
        should_report = (
            (step + 1) % _LOSS_REPORT_INTERVAL == 0
            or step + 1 == len(unlabeled_loader)
        )
        if should_report:
            total_loss = _flush_pending_losses(pending_losses, total_loss)
            progress.set_postfix(
                loss=f"{total_loss / observations:.5f}",
                refresh=False,
            )
    return total_loss / max(1, observations)


def train_simclr_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: ExperimentConfig,
    epoch: int = 1,
) -> float:
    model.train()
    criterion = NTXentLoss(config.training.temperature)
    optimizer.zero_grad(set_to_none=True)
    total_loss, observations = 0.0, 0
    pending_losses: list[tuple[torch.Tensor, int]] = []
    progress = tqdm(
        loader,
        desc=f"epoch {epoch:03d} simclr",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        mininterval=1.0,
        file=sys.stdout,
        colour=_progress_colour("blue"),
    )
    for step, (view_one, view_two, _targets, _sample_ids) in enumerate(progress):
        view_one = view_one.to(device, non_blocking=True)
        view_two = view_two.to(device, non_blocking=True)
        with _autocast(device, config.training.amp):
            embeddings = model(torch.cat((view_one, view_two), dim=0))
            first, second = embeddings.split(view_one.shape[0], dim=0)
            loss = criterion(first, second)
        _backward_step(loss, model, optimizer, scaler, step, step + 1 == len(loader), config)
        batch_size = view_one.shape[0]
        pending_losses.append((loss.detach(), batch_size))
        observations += batch_size
        should_report = (
            (step + 1) % _LOSS_REPORT_INTERVAL == 0
            or step + 1 == len(loader)
        )
        if should_report:
            total_loss = _flush_pending_losses(pending_losses, total_loss)
            progress.set_postfix(
                loss=f"{total_loss / observations:.5f}",
                refresh=False,
            )
    return total_loss / max(1, observations)


def fit(
    model: nn.Module,
    loaders: dict[str, torch.utils.data.DataLoader],
    config: ExperimentConfig,
    device: torch.device,
    output_directory: str | Path,
    resume_checkpoint: str | Path | None = None,
    compile_mode: str | None = None,
) -> dict[str, Any]:
    """Run one configured experiment and persist a complete audit trail."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    teacher = copy.deepcopy(model).to(device) if config.training.mode == "mean_teacher" else None
    if teacher is not None:
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler("cuda", enabled=config.training.amp and device.type == "cuda")
    criterion = build_classification_loss(config.training.robust_loss)
    writer = SummaryWriter(log_dir=output / "tensorboard")
    writer.add_custom_scalars(
        {
            "01 Loss": {
                "Train vs validation": [
                    "Multiline",
                    ["train/loss", "validation/loss"],
                ],
            },
            "02 Discrimination": {
                "ROC AUC": [
                    "Multiline",
                    ["validation/auc_roc", "test/auc_roc"],
                ],
                "Average precision": [
                    "Multiline",
                    ["validation/average_precision", "test/average_precision"],
                ],
            },
            "03 Classification quality": {
                "Accuracy": [
                    "Multiline",
                    ["validation/accuracy", "test/accuracy"],
                ],
                "Balanced accuracy": [
                    "Multiline",
                    ["validation/balanced_accuracy", "test/balanced_accuracy"],
                ],
                "F1": [
                    "Multiline",
                    ["validation/f1", "test/f1"],
                ],
            },
            "04 Clinical operating characteristics": {
                "Sensitivity and specificity": [
                    "Multiline",
                    [
                        "validation/sensitivity",
                        "validation/specificity",
                        "test/sensitivity",
                        "test/specificity",
                    ],
                ],
                "Precision": [
                    "Multiline",
                    ["validation/precision", "test/precision"],
                ],
            },
            "05 Calibration": {
                "Calibration errors": [
                    "Multiline",
                    [
                        "validation/ece",
                        "validation/brier",
                        "test/ece",
                        "test/brier",
                    ],
                ],
            },
            "06 Optimization": {
                "Learning rate": [
                    "Multiline",
                    ["optimizer/learning_rate"],
                ],
            },
        }
    )
    writer.add_text(
        "experiment/resolved_config",
        f"```json\n{json.dumps(asdict(config), indent=2, ensure_ascii=False)}\n```",
        0,
    )
    history: list[dict[str, float]] = []
    best_score = -math.inf
    epochs_without_improvement = 0
    start_epoch = 0

    if resume_checkpoint is not None:
        if config.training.mode == "mean_teacher":
            raise ValueError(
                "Mean Teacher cannot yet be resumed exactly because the legacy "
                "checkpoint does not contain separate student and teacher states."
            )
        checkpoint_path = Path(resume_checkpoint)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"])
        history_path = output / "history.json"
        if not history_path.is_file():
            raise FileNotFoundError(
                f"Cannot resume without training history: {history_path}"
            )
        loaded_history = json.loads(history_path.read_text(encoding="utf-8"))
        history = loaded_history[:start_epoch]
        last_best_index = -1
        for index, record in enumerate(history):
            if config.training.mode == "simclr":
                historical_score = -float(record["train_loss"])
            else:
                historical_score = float(
                    record.get(
                        f"validation_{config.training.monitor}",
                        float("nan"),
                    )
                )
            if math.isfinite(historical_score) and historical_score > best_score:
                best_score = historical_score
                last_best_index = index
        epochs_without_improvement = (
            len(history) - last_best_index - 1
            if last_best_index >= 0
            else len(history)
        )
        print(
            f"resumed_from={checkpoint_path} next_epoch={start_epoch + 1} "
            f"best_score={best_score:.6f} "
            f"epochs_without_improvement={epochs_without_improvement}",
            flush=True,
        )

    if compile_mode is not None:
        model_keys = tuple(model.state_dict())
        model.compile(mode=compile_mode, dynamic=False)
        if tuple(model.state_dict()) != model_keys:
            raise RuntimeError(
                "In-place torch.compile changed model state_dict keys; "
                "refusing to create an incompatible checkpoint."
            )
        if teacher is not None:
            teacher_keys = tuple(teacher.state_dict())
            teacher.compile(mode=compile_mode, dynamic=False)
            if tuple(teacher.state_dict()) != teacher_keys:
                raise RuntimeError(
                    "In-place torch.compile changed teacher state_dict keys."
                )
        print(
            f"torch_compile mode={compile_mode} dynamic=False "
            "first_batch_may_compile_for_several_minutes",
            flush=True,
        )

    for epoch in range(start_epoch, config.training.epochs):
        if config.training.mode == "supervised":
            train_loss = train_supervised_epoch(
                model,
                loaders["train"],
                optimizer,
                criterion,
                scaler,
                device,
                config,
                epoch + 1,
            )
            evaluation_model = model
        elif config.training.mode == "mean_teacher":
            assert teacher is not None
            train_loss = train_mean_teacher_epoch(
                model,
                teacher,
                loaders["labeled"],
                loaders["unlabeled"],
                optimizer,
                criterion,
                scaler,
                device,
                config,
                epoch + 1,
            )
            evaluation_model = teacher
        else:
            train_loss = train_simclr_epoch(
                model,
                loaders["train"],
                optimizer,
                scaler,
                device,
                config,
                epoch + 1,
            )
            evaluation_model = model
        if scheduler is not None:
            scheduler.step()

        if config.training.mode == "simclr":
            epoch_record = {
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            score = -train_loss
        else:
            validation = evaluate(
                evaluation_model,
                loaders["validation"],
                device,
                config.training.amp,
                description=f"epoch {epoch + 1:03d} validation",
                writer=writer,
                global_step=epoch + 1,
                tensorboard_prefix="validation",
            )
            epoch_record = {
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{f"validation_{key}": value for key, value in validation.items()},
            }
            score = validation.get(config.training.monitor, float("nan"))
        history.append(epoch_record)
        write_json(output / "history.json", history)
        for metric_name, metric_value in epoch_record.items():
            if metric_name != "epoch" and math.isfinite(float(metric_value)):
                writer.add_scalar(_tensorboard_tag(metric_name), metric_value, epoch + 1)
        writer.flush()

        if math.isfinite(score) and score > best_score:
            best_score = score
            epochs_without_improvement = 0
            save_checkpoint(
                output / "best.pt",
                evaluation_model,
                optimizer,
                epoch + 1,
                epoch_record,
                config,
                scheduler,
                scaler,
            )
        else:
            epochs_without_improvement += 1
        save_checkpoint(
            output / "last.pt",
            evaluation_model,
            optimizer,
            epoch + 1,
            epoch_record,
            config,
            scheduler,
            scaler,
        )
        score_name = (
            "neg_train_loss"
            if config.training.mode == "simclr"
            else config.training.monitor
        )
        summary = (
            f"epoch={epoch + 1:03d} loss={train_loss:.5f} "
            f"{score_name}={score:.5f} "
            f"lr={optimizer.param_groups[0]['lr']:.3e}"
        )
        print(_styled(summary, "1;32"), flush=True)
        if epochs_without_improvement >= config.training.patience:
            break

    result: dict[str, Any] = {"best_score": best_score, "history": history}
    if config.training.mode != "simclr":
        checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
        evaluation_model.load_state_dict(checkpoint["model"])
        predictions_path = output / "test_predictions.csv" if config.output.save_predictions else None
        result["test"] = evaluate(
            evaluation_model,
            loaders["test"],
            device,
            config.training.amp,
            predictions_path,
            description="test",
            writer=writer,
            global_step=len(history),
            tensorboard_prefix="test",
        )
        for metric_name, metric_value in result["test"].items():
            writer.add_scalar(f"test/{metric_name}", metric_value, len(history))
    write_json(output / "result.json", result)
    writer.flush()
    writer.close()
    return result
