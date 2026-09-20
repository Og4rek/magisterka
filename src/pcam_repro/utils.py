"""Reproducibility, checkpoint, and runtime helpers."""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def parameter_count(model: nn.Module, trainable_only: bool = False) -> int:
    parameters: Iterable[nn.Parameter] = model.parameters()
    if trainable_only:
        parameters = (p for p in parameters if p.requires_grad)
    return sum(p.numel() for p in parameters)


def update_ema(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    with torch.no_grad():
        teacher_state = tuple(teacher.state_dict().values())
        student_state = tuple(student.state_dict().values())
        if len(teacher_state) != len(student_state):
            raise ValueError("EMA models have different state sizes.")

        floating_groups: dict[
            tuple[torch.device, torch.dtype],
            tuple[list[torch.Tensor], list[torch.Tensor]],
        ] = {}
        for teacher_value, student_value in zip(
            teacher_state,
            student_state,
            strict=True,
        ):
            if teacher_value.shape != student_value.shape:
                raise ValueError("EMA models have incompatible state tensors.")
            if teacher_value.is_floating_point():
                key = (teacher_value.device, teacher_value.dtype)
                teachers, students = floating_groups.setdefault(
                    key,
                    ([], []),
                )
                teachers.append(teacher_value)
                students.append(student_value)
            else:
                teacher_value.copy_(student_value)

        for teachers, students in floating_groups.values():
            torch._foreach_mul_(teachers, decay)
            torch._foreach_add_(teachers, students, alpha=1.0 - decay)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    config: Any,
    scheduler: Any | None = None,
    scaler: Any | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "metrics": metrics,
        "config": asdict(config),
    }
    torch.save(payload, target)


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_epochs: int,
        warmup_epochs: int,
        minimum_ratio: float,
    ) -> None:
        def schedule(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return max(1e-8, (epoch + 1) / warmup_epochs)
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            return minimum_ratio + (1.0 - minimum_ratio) * cosine

        super().__init__(optimizer, schedule)
