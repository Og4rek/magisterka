"""Command-line interface for reproducible PCam experiments."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from .config import load_config
from .data import PCamH5Dataset, build_dataloaders
from .engine import evaluate, fit
from .models import MODEL_NAMES, build_model
from .utils import parameter_count, resolve_device, seed_everything, write_json


def _resume_start_epoch(checkpoint_path: Path) -> int:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    return int(checkpoint["epoch"]) + 1


def _write_runtime_audit(
    run_directory: Path,
    runtime: dict[str, object],
    resume_checkpoint: Path | None,
    session_start_epoch: int,
) -> str:
    """Write the latest runtime and retain all execution-mode transitions."""
    runtime_path = run_directory / "runtime.json"
    sessions_path = run_directory / "runtime_sessions.json"
    sessions: list[dict[str, object]] = []
    if sessions_path.is_file():
        loaded_sessions = json.loads(sessions_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_sessions, list):
            raise ValueError(f"Runtime session history must be a list: {sessions_path}")
        sessions = loaded_sessions
    elif runtime_path.is_file():
        legacy_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if not isinstance(legacy_runtime, dict):
            raise ValueError(f"Legacy runtime must be an object: {runtime_path}")
        sessions.append({"legacy_import": True, **legacy_runtime})

    session_started_at = datetime.now().astimezone().isoformat()
    current_runtime = {
        **runtime,
        "session_started_at": session_started_at,
        "session_finished_at": None,
        "session_status": "started",
        "session_start_epoch": session_start_epoch,
        "compile_start_epoch": (
            session_start_epoch if runtime["compile_enabled"] else None
        ),
        "resume_checkpoint": (
            str(resume_checkpoint.resolve())
            if resume_checkpoint is not None
            else None
        ),
    }
    sessions.append(current_runtime)
    write_json(runtime_path, current_runtime)
    write_json(sessions_path, sessions)
    return session_started_at


def _finish_runtime_audit(
    run_directory: Path,
    session_started_at: str,
    status: str,
) -> None:
    if status not in {"completed", "failed", "interrupted"}:
        raise ValueError(f"Unknown runtime session status: {status}")
    runtime_path = run_directory / "runtime.json"
    sessions_path = run_directory / "runtime_sessions.json"
    current_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    sessions = json.loads(sessions_path.read_text(encoding="utf-8"))
    if not isinstance(current_runtime, dict) or not isinstance(sessions, list):
        raise ValueError("Malformed runtime audit files.")
    finished_at = datetime.now().astimezone().isoformat()
    for session in reversed(sessions):
        if session.get("session_started_at") == session_started_at:
            session["session_status"] = status
            session["session_finished_at"] = finished_at
            break
    else:
        raise ValueError(f"Runtime session not found: {session_started_at}")
    if current_runtime.get("session_started_at") == session_started_at:
        current_runtime["session_status"] = status
        current_runtime["session_finished_at"] = finished_at
    write_json(runtime_path, current_runtime)
    write_json(sessions_path, sessions)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pcam-repro",
        description="Reproduction suite for neural PCam metastasis classifiers.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="Train one TOML experiment.")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--device", default="auto")
    train.add_argument("--run-dir", type=Path)
    train.add_argument(
        "--fast",
        action="store_true",
        help="Enable cuDNN autotuning and non-deterministic high-throughput kernels.",
    )
    train.add_argument(
        "--resume",
        action="store_true",
        help="Resume from <run-dir>/last.pt and the existing history.json.",
    )
    train.add_argument(
        "--compile",
        dest="compile_model",
        action="store_true",
        help=(
            "Compile the model in-place with torch.compile(mode='default', "
            "dynamic=False). This preserves checkpoint keys but may change "
            "floating-point reduction order."
        ),
    )
    evaluation = subparsers.add_parser("evaluate", help="Evaluate a saved checkpoint.")
    evaluation.add_argument("--config", required=True, type=Path)
    evaluation.add_argument("--checkpoint", required=True, type=Path)
    evaluation.add_argument("--split", choices=("validation", "test"), default="test")
    evaluation.add_argument("--device", default="auto")
    evaluation.add_argument("--predictions", type=Path)
    evaluation.add_argument(
        "--update-result",
        type=Path,
        help="Replace the selected split metrics in an existing result JSON.",
    )
    inspect = subparsers.add_parser("inspect-data", help="Validate official PCam HDF5 files.")
    inspect.add_argument("--root", required=True, type=Path)
    subparsers.add_parser("list-models", help="List implemented architecture identifiers.")
    return parser


def _train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    seed_everything(config.training.seed, deterministic=not args.fast)
    if args.fast:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    device = resolve_device(args.device)
    compile_mode = "default" if args.compile_model else None
    triton_version: str | None = None
    triton_distribution: str | None = None
    triton_distribution_version: str | None = None
    if compile_mode is not None and device.type == "cuda":
        torch_release = tuple(
            int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2]
        )
        if os.name == "nt" and torch_release not in {(2, 12), (2, 13)}:
            raise RuntimeError(
                "The Windows compile extra is validated only with PyTorch "
                "2.12-2.13 and Triton 3.7."
            )
        try:
            import triton
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "CUDA torch.compile requires Triton. On Windows install the "
                "project's compile extra: python -m pip install -e \".[compile]\""
            ) from error
        triton_version = triton.__version__
        for distribution_name in ("triton-windows", "triton"):
            try:
                triton_distribution_version = importlib.metadata.version(
                    distribution_name
                )
            except importlib.metadata.PackageNotFoundError:
                continue
            triton_distribution = distribution_name
            break
    model = build_model(config.model, config.training.mode)
    if config.model.checkpoint:
        initialization = torch.load(config.model.checkpoint, map_location="cpu", weights_only=False)
        state = initialization.get("model", initialization)
        if any(key.startswith("encoder.") for key in state):
            state = {
                key.removeprefix("encoder."): value
                for key, value in state.items()
                if key.startswith("encoder.")
            }
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"initialized_from={config.model.checkpoint} "
            f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}",
            flush=True,
        )
    if config.model.freeze_encoder:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        classifier = getattr(model, "classifier", None)
        if classifier is None:
            raise ValueError("freeze_encoder requires a model with a classifier attribute.")
        for parameter in classifier.parameters():
            parameter.requires_grad_(True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_directory = args.run_dir or (
        Path(config.output.directory) / f"{config.output.experiment_name}-{timestamp}"
    )
    if args.resume and args.run_dir is None:
        raise ValueError("--resume requires an explicit --run-dir.")
    resume_checkpoint = run_directory / "last.pt" if args.resume else None
    if resume_checkpoint is not None and not resume_checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")
    write_json(run_directory / "resolved_config.json", asdict(config))
    session_started_at = _write_runtime_audit(
        run_directory,
        {
            "device": str(device),
            "torch_version": torch.__version__,
            "parameter_count": parameter_count(model),
            "trainable_parameter_count": parameter_count(model, trainable_only=True),
            "fast_mode": args.fast,
            "compile_enabled": compile_mode is not None,
            "compile_mode": compile_mode,
            "compile_dynamic": False if compile_mode is not None else None,
            "triton_version": triton_version,
            "triton_distribution": triton_distribution,
            "triton_distribution_version": triton_distribution_version,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
        resume_checkpoint,
        _resume_start_epoch(resume_checkpoint) if resume_checkpoint is not None else 1,
    )
    print(
        f"model={config.model.name} parameters={parameter_count(model):,} "
        f"device={device} run={run_directory}",
        flush=True,
    )
    try:
        loaders = build_dataloaders(
            config.data,
            config.training.mode,
            config.training.seed,
            config.training.noise_rate,
            config.training.noise_seed,
        )
        result = fit(
            model,
            loaders,
            config,
            device,
            run_directory,
            resume_checkpoint,
            compile_mode,
        )
    except KeyboardInterrupt:
        _finish_runtime_audit(run_directory, session_started_at, "interrupted")
        raise
    except Exception:
        _finish_runtime_audit(run_directory, session_started_at, "failed")
        raise
    else:
        _finish_runtime_audit(run_directory, session_started_at, "completed")
    if "test" in result:
        print("test:", result["test"], flush=True)


def _evaluate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if config.training.mode == "simclr":
        raise ValueError(
            "A SimCLR pretraining checkpoint has no classification head. "
            "Evaluate it through e11a linear probe or e11b full fine-tuning."
        )
    device = resolve_device(args.device)
    model = build_model(config.model, config.training.mode).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    loaders = build_dataloaders(
        config.data,
        config.training.mode,
        config.training.seed,
        config.training.noise_rate,
        config.training.noise_seed,
    )
    metrics = evaluate(
        model,
        loaders[args.split],
        device,
        config.training.amp,
        args.predictions,
    )
    if args.update_result is not None:
        if not args.update_result.is_file():
            raise FileNotFoundError(f"Result document not found: {args.update_result}")
        result = json.loads(args.update_result.read_text(encoding="utf-8"))
        result[args.split] = metrics
        result[f"{args.split}_evaluation"] = {
            "checkpoint": str(args.checkpoint.resolve()),
            "evaluated_at": datetime.now().astimezone().isoformat(),
            "cache_safe": True,
        }
        write_json(args.update_result, result)
    print(metrics)


def _inspect_data(root: Path) -> None:
    for split in ("train", "valid", "test"):
        dataset = PCamH5Dataset(root, split)
        negatives = int((dataset.labels == 0).sum())
        positives = int((dataset.labels == 1).sum())
        image, target, index = dataset[0]
        print(
            f"{split:5s}: n={len(dataset):6d}, negative={negatives:6d}, "
            f"positive={positives:6d}, sample_shape={tuple(image.shape)}, "
            f"first_target={float(target):.0f}, first_id={index}"
        )


def main() -> None:
    args = _parser().parse_args()
    if args.command == "train":
        _train(args)
    elif args.command == "evaluate":
        _evaluate(args)
    elif args.command == "inspect-data":
        _inspect_data(args.root)
    elif args.command == "list-models":
        print("\n".join(MODEL_NAMES))


if __name__ == "__main__":
    main()
