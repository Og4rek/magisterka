"""Sequentially execute the complete PCam experiment matrix.

The runner deliberately starts each experiment as a separate Python process.
This releases CUDA memory between models and makes a failed experiment
independent from the remaining queue.
"""

from __future__ import annotations

import argparse
import codecs
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIRECTORY = PROJECT_ROOT / "runs"
REQUIRED_DATA_FILES = tuple(
    f"camelyonpatch_level_2_split_{split}_{kind}.h5"
    for split in ("train", "valid", "test")
    for kind in ("x", "y")
)


@dataclass(frozen=True, slots=True)
class Experiment:
    key: str
    config: str
    run_name: str
    group: str
    depends_on: tuple[str, ...] = ()


EXPERIMENTS = (
    Experiment("e00", "e00_smallcnn.toml", "e00-smallcnn-control", "core"),
    Experiment("e01", "e01_resnet18_scratch.toml", "e01-resnet18-scratch", "core"),
    Experiment("e02", "e02_resnet18_imagenet.toml", "e02-resnet18-imagenet", "core"),
    Experiment("e03", "e03_resnet50_scratch.toml", "e03-resnet50-scratch", "core"),
    Experiment("e04", "e04_resnet50_imagenet.toml", "e04-resnet50-imagenet", "core"),
    Experiment("e05", "e05_densenet121_scratch.toml", "e05-densenet121-scratch", "core"),
    Experiment("e06", "e06_densenet121_imagenet.toml", "e06-densenet121-imagenet", "core"),
    Experiment("e07", "e07_gdensenet_d4.toml", "e07-gdensenet-d4", "core"),
    Experiment("e08", "e08_partial_se2.toml", "e08-partial-se2-c16", "core"),
    Experiment("e09", "e09_inception_5pct.toml", "e09-inception-supervised-5pct", "label"),
    Experiment("e10", "e10_mean_teacher_5pct.toml", "e10-mean-teacher-5pct", "label"),
    Experiment("e11", "e11_simclr_resnet50.toml", "e11-simclr-resnet50", "ssl"),
    Experiment(
        "e11a",
        "e11a_simclr_linear_probe.toml",
        "e11a-simclr-resnet50-linear-probe",
        "ssl",
        ("e11",),
    ),
    Experiment(
        "e11b",
        "e11b_simclr_finetune.toml",
        "e11b-simclr-resnet50-finetune",
        "ssl",
        ("e11",),
    ),
    Experiment("e12", "e12_vit_small.toml", "e12-vit-small-16", "extended"),
    Experiment("e13", "e13_lgvit.toml", "e13-local-global-vit", "extended"),
    Experiment("e14", "e14_noise_robust.toml", "e14-resnet18-noise20-sce", "extended"),
    Experiment("e15", "e15_lotenet.toml", "e15-lotenet", "extended"),
    Experiment(
        "e16",
        "e16_multiscale_attention.toml",
        "e16-multiscale-attention",
        "extended",
    ),
    Experiment("e17", "e17_pcam_d4loc_s17.toml", "e17-pcam-d4loc-s17", "custom"),
    Experiment("e17s43", "e17_pcam_d4loc_s43.toml", "e17-pcam-d4loc-s43", "custom"),
    Experiment("e17s71", "e17_pcam_d4loc_s71.toml", "e17-pcam-d4loc-s71", "custom"),
    Experiment("e17a", "e17a_d4loc_no_d4.toml", "e17a-d4loc-no-d4", "custom"),
    Experiment("e17b", "e17b_d4loc_global_pool.toml", "e17b-d4loc-global-pool", "custom"),
    Experiment(
        "e17c",
        "e17c_d4loc_no_multiscale.toml",
        "e17c-d4loc-no-multiscale",
        "custom",
    ),
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def styled(text: str, ansi_code: str) -> str:
    color_enabled = (
        os.environ.get("NO_COLOR") is None
        and (sys.stdout.isatty() or os.environ.get("FORCE_COLOR") == "1")
    )
    return f"\033[{ansi_code}m{text}\033[0m" if color_enabled else text


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ordered PCam experiment matrix in separate processes."
    )
    parser.add_argument(
        "--profile",
        choices=("all", "core", "label", "ssl", "extended", "custom", "smoke"),
        default="all",
        help="Experiment group. 'all' runs every research configuration.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="KEY",
        help="Run only selected keys, for example --only e00 e01 e07.",
    )
    parser.add_argument(
        "--start-from",
        metavar="KEY",
        help="Discard earlier items from the selected queue.",
    )
    parser.add_argument("--device", default="cuda", help="Device passed to the training CLI.")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIRECTORY,
        help="Root directory for checkpoints, predictions and logs.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Run even if result.json already exists. Existing files may be replaced.",
    )
    parser.add_argument(
        "--restart-incomplete",
        action="store_true",
        help="Allow reuse of a non-empty run directory without result.json.",
    )
    parser.add_argument(
        "--resume-incomplete",
        action="store_true",
        help="Resume an incomplete run from last.pt instead of restarting it.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with independent experiments after a failure.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print commands without starting training.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Do not ask for confirmation before a long full-matrix run.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use cuDNN autotuning and faster non-deterministic CUDA kernels.",
    )
    parser.add_argument(
        "--compile-models",
        nargs="+",
        metavar="KEY",
        default=(),
        help=(
            "Enable torch.compile only for the listed experiment keys. "
            "E11 and E13 are benchmarked on the local RTX 4060."
        ),
    )
    return parser.parse_args()


def selected_experiments(args: argparse.Namespace) -> list[Experiment]:
    if args.profile == "smoke":
        queue = [
            Experiment(
                "smoke",
                "smoke_smallcnn_2epochs.toml",
                "smoke-smallcnn-2epochs",
                "smoke",
            )
        ]
    elif args.profile == "all":
        queue = list(EXPERIMENTS)
    else:
        queue = [experiment for experiment in EXPERIMENTS if experiment.group == args.profile]

    if args.only:
        requested = set(args.only)
        known = {experiment.key for experiment in queue}
        unknown = requested - known
        if unknown:
            raise SystemExit(
                f"Unknown keys for profile '{args.profile}': {', '.join(sorted(unknown))}"
            )
        queue = [experiment for experiment in queue if experiment.key in requested]

    if args.start_from:
        positions = {experiment.key: index for index, experiment in enumerate(queue)}
        if args.start_from not in positions:
            raise SystemExit(f"--start-from key is not in the selected queue: {args.start_from}")
        queue = queue[positions[args.start_from] :]

    if not queue:
        raise SystemExit("The selected experiment queue is empty.")
    return queue


def validate_environment(device: str) -> dict[str, object]:
    import h5py
    import torch

    data_root = PROJECT_ROOT / "data" / "pcam"
    missing = [name for name in REQUIRED_DATA_FILES if not (data_root / name).is_file()]
    if missing:
        raise SystemExit("Missing PCam files:\n  " + "\n  ".join(missing))

    with h5py.File(data_root / REQUIRED_DATA_FILES[0], "r") as handle:
        if not handle.keys():
            raise SystemExit("The PCam training HDF5 file contains no datasets.")

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is False.")

    runtime: dict[str, object] = {
        "python": sys.version,
        "torch": torch.__version__,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        runtime.update(
            {
                "gpu": properties.name,
                "gpu_memory_gib": round(properties.total_memory / 1024**3, 2),
                "cuda_runtime": torch.version.cuda,
            }
        )
    return runtime


def tee_process(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONUTF8"] = "1"
    environment.setdefault("FORCE_COLOR", "1")
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
        )
        assert process.stdout is not None
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            while chunk := process.stdout.read(4096):
                text = decoder.decode(chunk)
                # Preserve UTF-8 progress-bar glyphs and carriage returns even
                # when Windows configures the parent stream as cp1250.
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                log.write(text)
                log.flush()
            remainder = decoder.decode(b"", final=True)
            if remainder:
                log.write(remainder)
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
        return process.wait()


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def dependency_result(experiment_key: str, queue_by_key: dict[str, Experiment], runs: Path) -> Path:
    dependency = queue_by_key.get(experiment_key)
    if dependency is None:
        dependency = next(item for item in EXPERIMENTS if item.key == experiment_key)
    return runs / dependency.run_name / "result.json"


def position_label(
    experiment: Experiment,
    queue_position: int,
    queue_length: int,
) -> str:
    full_positions = {
        item.key: position for position, item in enumerate(EXPERIMENTS, start=1)
    }
    if experiment.key in full_positions:
        return (
            f"[{full_positions[experiment.key]}/{len(EXPERIMENTS)}"
            f" | queue {queue_position}/{queue_length}]"
        )
    return f"[{queue_position}/{queue_length}]"


def main() -> int:
    args = parse_arguments()
    if args.resume_incomplete and (args.restart_incomplete or args.rerun):
        raise SystemExit(
            "--resume-incomplete cannot be combined with "
            "--restart-incomplete or --rerun."
        )
    queue = selected_experiments(args)
    compile_models = set(args.compile_models)
    known_experiments = {experiment.key for experiment in EXPERIMENTS}
    unknown_compile_models = compile_models - known_experiments
    if unknown_compile_models:
        raise SystemExit(
            "Unknown --compile-models keys: "
            + ", ".join(sorted(unknown_compile_models))
        )
    selected_keys = {experiment.key for experiment in queue}
    unselected_compile_models = compile_models - selected_keys
    if unselected_compile_models:
        raise SystemExit(
            "--compile-models keys are outside the selected queue: "
            + ", ".join(sorted(unselected_compile_models))
        )
    runs_directory = args.runs_dir.resolve()
    runtime = validate_environment(args.device)
    queue_by_key = {experiment.key: experiment for experiment in queue}

    print("PCam experiment runner")
    print(f"project: {PROJECT_ROOT}")
    print(f"runs:   {runs_directory}")
    print(f"python: {sys.executable}")
    print(f"torch:  {runtime['torch']}")
    if runtime.get("gpu"):
        print(f"gpu:    {runtime['gpu']} ({runtime['gpu_memory_gib']} GiB)")
    print("queue:  " + ", ".join(experiment.key for experiment in queue))
    print(
        "WARNING: the full matrix can require many days on a laptop GPU. "
        "The runner is sequential and safe to interrupt between experiments."
    )

    for experiment in queue:
        config_path = PROJECT_ROOT / "configs" / experiment.config
        if not config_path.is_file():
            raise SystemExit(f"Missing configuration: {config_path}")

    if not args.dry_run and not args.yes and sys.stdin.isatty():
        confirmation = input("Type RUN to start the selected queue: ").strip()
        if confirmation != "RUN":
            print("Cancelled.")
            return 0

    session_name = datetime.now().strftime("orchestration-%Y%m%d-%H%M%S")
    session_directory = runs_directory / "_orchestration" / session_name
    manifest_path = session_directory / "manifest.json"
    manifest: dict[str, object] = {
        "started_at": utc_now(),
        "finished_at": None,
        "profile": args.profile,
        "compile_models": sorted(compile_models),
        "runtime": runtime,
        "experiments": [],
    }

    failed_keys: set[str] = set()
    successful_keys: set[str] = set()
    for position, experiment in enumerate(queue, start=1):
        label = position_label(experiment, position, len(queue))
        run_directory = runs_directory / experiment.run_name
        result_path = run_directory / "result.json"
        config_path = PROJECT_ROOT / "configs" / experiment.config
        record: dict[str, object] = {
            "key": experiment.key,
            "config": str(config_path),
            "run_directory": str(run_directory),
            "started_at": None,
            "finished_at": None,
            "status": "pending",
            "return_code": None,
            "compile_model": experiment.key in compile_models,
        }
        experiments_log = manifest["experiments"]
        assert isinstance(experiments_log, list)
        experiments_log.append(record)

        unavailable_dependencies = [
            dependency
            for dependency in experiment.depends_on
            if dependency in failed_keys
            or (
                dependency not in successful_keys
                and not dependency_result(
                    dependency, queue_by_key, runs_directory
                ).is_file()
            )
        ]
        if unavailable_dependencies:
            record["status"] = "blocked_dependency"
            record["finished_at"] = utc_now()
            record["dependencies"] = unavailable_dependencies
            failed_keys.add(experiment.key)
            write_manifest(manifest_path, manifest)
            print(
                f"{label} BLOCKED {experiment.key}: "
                f"missing {', '.join(unavailable_dependencies)}"
            )
            if not args.continue_on_error:
                break
            continue

        if result_path.is_file() and not args.rerun:
            record["status"] = "skipped_completed"
            record["finished_at"] = utc_now()
            successful_keys.add(experiment.key)
            write_manifest(manifest_path, manifest)
            print(
                styled(
                    f"{label} SKIP {experiment.key}: "
                    "result.json exists",
                    "1;33",
                )
            )
            continue

        resume_this_experiment = False
        if run_directory.exists() and any(run_directory.iterdir()):
            if args.resume_incomplete:
                resume_path = run_directory / "last.pt"
                if resume_path.is_file():
                    resume_this_experiment = True
                    record["resumed_from"] = str(resume_path)
                else:
                    record["status"] = "blocked_missing_resume_checkpoint"
                    record["finished_at"] = utc_now()
                    failed_keys.add(experiment.key)
                    write_manifest(manifest_path, manifest)
                    print(
                        f"{label} BLOCKED {experiment.key}: "
                        f"resume checkpoint does not exist: {resume_path}"
                    )
                    if not args.continue_on_error:
                        break
                    continue
            elif not (args.rerun or args.restart_incomplete):
                record["status"] = "blocked_incomplete_directory"
                record["finished_at"] = utc_now()
                failed_keys.add(experiment.key)
                write_manifest(manifest_path, manifest)
                print(
                    f"{label} BLOCKED {experiment.key}: "
                    f"{run_directory} is non-empty and has no completed result. "
                    "Use --restart-incomplete after reviewing it."
                )
                if not args.continue_on_error:
                    break
                continue

        command = [
            sys.executable,
            "-m",
            "pcam_repro.cli",
            "train",
            "--config",
            str(config_path),
            "--device",
            args.device,
            "--run-dir",
            str(run_directory),
        ]
        if args.fast:
            command.append("--fast")
        if experiment.key in compile_models:
            command.append("--compile")
        if resume_this_experiment:
            command.append("--resume")
        record["command"] = command
        print(
            styled(
                f"{label} START {experiment.key}: "
                f"{experiment.config}",
                "1;36",
            )
        )

        if args.dry_run:
            record["status"] = "dry_run"
            record["finished_at"] = utc_now()
            successful_keys.add(experiment.key)
            print("  " + subprocess.list2cmdline(command))
            continue

        record["started_at"] = utc_now()
        record["status"] = "running"
        write_manifest(manifest_path, manifest)
        return_code = tee_process(command, session_directory / f"{experiment.key}.log")
        record["return_code"] = return_code
        record["finished_at"] = utc_now()
        record["status"] = "completed" if return_code == 0 and result_path.is_file() else "failed"
        write_manifest(manifest_path, manifest)

        if record["status"] == "completed":
            successful_keys.add(experiment.key)
            print(styled(f"{label} DONE {experiment.key}", "1;32"))
        else:
            failed_keys.add(experiment.key)
            print(
                styled(
                    f"{label} FAILED {experiment.key} "
                    f"(exit={return_code})",
                    "1;31",
                )
            )
            if not args.continue_on_error:
                break

    manifest["finished_at"] = utc_now()
    manifest["failed_keys"] = sorted(failed_keys)
    if not args.dry_run:
        write_manifest(manifest_path, manifest)
        print(f"manifest: {manifest_path}")
    return 1 if failed_keys else 0


if __name__ == "__main__":
    raise SystemExit(main())
