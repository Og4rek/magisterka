"""Run a reproducible qualitative and quantitative XAI audit on PCam models."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean

import numpy as np
import torch

from pcam_repro.config import ExperimentConfig, load_config
from pcam_repro.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PCAM_MEAN,
    PCAM_STD,
    PCamH5Dataset,
    build_transform,
)
from pcam_repro.models import build_model
from pcam_repro.utils import parameter_count, resolve_device
from pcam_repro.xai.attribution import (
    apply_d4,
    gradcam,
    integrated_gradients,
    native_evidence,
    randomized_model,
    rise,
    undo_d4,
)
from pcam_repro.xai.masks import CamelyonMaskProvider, PCamMetadata
from pcam_repro.xai.metrics import (
    center_mass_ratio,
    localization_metrics,
    map_similarity,
    perturbation_faithfulness,
)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    config: str
    run: str


MODEL_SPECS = {
    spec.key: spec
    for spec in (
        ModelSpec("e06", "e06_densenet121_imagenet.toml", "e06-densenet121-imagenet"),
        ModelSpec("e07", "e07_gdensenet_d4.toml", "e07-gdensenet-d4"),
        ModelSpec(
            "e11b",
            "e11b_simclr_finetune.toml",
            "e11b-simclr-resnet50-finetune",
        ),
        ModelSpec("e13", "e13_lgvit.toml", "e13-local-global-vit"),
        ModelSpec("e17", "e17_pcam_d4loc_s17.toml", "e17-pcam-d4loc-s17"),
    )
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--models", nargs="+", default=("e06", "e07", "e13"))
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("gradcam", "ig", "rise", "native"),
        default=("gradcam", "ig", "rise", "native"),
    )
    parser.add_argument(
        "--target-mode",
        choices=("predicted", "tumor", "truth"),
        default="predicted",
    )
    parser.add_argument("--analysis-batch-size", type=int, default=8)
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--ig-alpha-batch-size", type=int, default=8)
    parser.add_argument("--rise-masks", type=int, default=1024)
    parser.add_argument("--rise-batch-size", type=int, default=128)
    parser.add_argument("--rise-grid-size", type=int, default=7)
    parser.add_argument("--faithfulness-steps", type=int, default=20)
    parser.add_argument(
        "--skip-faithfulness",
        action="store_true",
        help="Skip deletion/insertion metrics during a very short smoke run.",
    )
    parser.add_argument(
        "--stability-transforms",
        nargs="*",
        type=int,
        default=(1, 4),
        help="D4 group indices; defaults to 90-degree rotation and reflection.",
    )
    parser.add_argument(
        "--stability-methods",
        nargs="+",
        default=("gradcam", "ig", "native"),
    )
    parser.add_argument(
        "--sanity-samples",
        type=int,
        default=4,
        help="Number of images used for classifier/full parameter randomization checks.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        help="CAMELYON16 XML directory. Omit to report center-only metrics.",
    )
    parser.add_argument(
        "--mapping-audit",
        type=Path,
        help="JSON produced by audit_pcam_annotation_mapping.py.",
    )
    parser.add_argument("--coordinate-mode", choices=("top_left", "center"))
    parser.add_argument("--downsample", type=float)
    parser.add_argument("--max-figure-samples", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/xai/latest"),
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Reuse per-model CSV files already completed in --output.",
    )
    return parser.parse_args()


def _validate_arguments(args: argparse.Namespace) -> None:
    unknown = set(args.models) - set(MODEL_SPECS)
    if unknown:
        raise SystemExit(f"Unknown XAI model keys: {', '.join(sorted(unknown))}")
    positive_values = {
        "samples": args.samples,
        "analysis-batch-size": args.analysis_batch_size,
        "ig-steps": args.ig_steps,
        "rise-masks": args.rise_masks,
        "faithfulness-steps": args.faithfulness_steps,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise SystemExit(f"These arguments must be positive: {', '.join(invalid)}")
    if any(index not in range(8) for index in args.stability_transforms):
        raise SystemExit("D4 stability transform indices must lie in [0, 7].")


def _read_predictions(path: Path) -> dict[int, tuple[int, float]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            int(row["sample_id"]): (int(float(row["target"])), float(row["probability"]))
            for row in csv.DictReader(handle)
        }


def _category(target: int, probability: float) -> str:
    predicted = int(probability >= 0.5)
    return ("t" if predicted == target else "f") + ("p" if predicted else "n")


def _stratified_sample_ids(
    predictions: dict[int, tuple[int, float]],
    metadata: PCamMetadata,
    sample_count: int,
    seed: int,
) -> list[int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for sample_id, (target, probability) in predictions.items():
        groups[_category(target, probability)].append(sample_id)
    rng = random.Random(seed)
    chosen: list[int] = []
    per_group = max(1, math.ceil(sample_count / 4))
    for name in ("tp", "tn", "fp", "fn"):
        candidates = groups[name]
        rng.shuffle(candidates)
        # Prefer distinct WSIs before filling from repeated slides.
        used_wsi: set[str] = set()
        preferred: list[int] = []
        repeated: list[int] = []
        for sample_id in candidates:
            wsi = metadata[sample_id].wsi
            if wsi in used_wsi:
                repeated.append(sample_id)
            else:
                used_wsi.add(wsi)
                preferred.append(sample_id)
        chosen.extend((preferred + repeated)[:per_group])
    if len(chosen) < sample_count:
        remainder = list(set(predictions) - set(chosen))
        rng.shuffle(remainder)
        chosen.extend(remainder[: sample_count - len(chosen)])
    return chosen[:sample_count]


def _load_model(
    project_root: Path,
    spec: ModelSpec,
    device: torch.device,
) -> tuple[torch.nn.Module, ExperimentConfig, Path]:
    config_path = project_root / "configs" / spec.config
    checkpoint_path = project_root / "runs" / spec.run / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint for {spec.key}: {checkpoint_path}")
    config = load_config(config_path)
    # A fine-tuned checkpoint is self-contained; do not require its SSL source.
    config.model.checkpoint = ""
    model = build_model(config.model, config.training.mode).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, config, checkpoint_path


def _target_classes(
    args: argparse.Namespace,
    sample_ids: list[int],
    predictions: dict[int, tuple[int, float]],
    device: torch.device,
) -> torch.Tensor:
    if args.target_mode == "tumor":
        values = [1] * len(sample_ids)
    elif args.target_mode == "truth":
        values = [predictions[sample_id][0] for sample_id in sample_ids]
    else:
        values = [int(predictions[sample_id][1] >= 0.5) for sample_id in sample_ids]
    return torch.tensor(values, dtype=torch.long, device=device)


def _compute_method(
    method: str,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    args: argparse.Namespace,
    seed_offset: int = 0,
) -> torch.Tensor:
    if method == "gradcam":
        return gradcam(model, inputs, targets)
    if method == "ig":
        return integrated_gradients(
            model,
            inputs,
            targets,
            steps=args.ig_steps,
            alpha_batch_size=args.ig_alpha_batch_size,
        )
    if method == "rise":
        mask_batch_size = args.rise_batch_size
        if model.__class__.__name__ in {"GDenseNetD4", "PCamD4Loc"}:
            # Each black-box RISE mask is a complete image in a batch. D4
            # features multiply activation memory by eight orientations.
            mask_batch_size = min(mask_batch_size, 32)
        return rise(
            model,
            inputs,
            targets,
            mask_count=args.rise_masks,
            mask_batch_size=mask_batch_size,
            grid_size=args.rise_grid_size,
            seed=args.seed + seed_offset,
        )
    if method == "native":
        return native_evidence(model, inputs, targets)
    raise KeyError(method)


def _batched_attribution(
    method: str,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    maps: list[torch.Tensor] = []
    batch_size = args.analysis_batch_size
    if method in {"gradcam", "ig"} and model.__class__.__name__ in {
        "GDenseNetD4",
        "PCamD4Loc",
    }:
        # IG expands a batch by alpha_batch_size, while Grad-CAM retains both
        # activations and gradients. Group feature maps add an orientation
        # axis of length eight, so a conservative cap prevents allocator
        # fragmentation on 8-GiB laptop GPUs.
        batch_size = min(batch_size, 2)
    start = 0
    chunk_index = 0
    while start < inputs.shape[0]:
        stop = min(inputs.shape[0], start + batch_size)
        try:
            computed = _compute_method(
                method,
                model,
                inputs[start:stop],
                targets[start:stop],
                args,
                seed_offset=chunk_index * 1009,
            )
        except torch.OutOfMemoryError:
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            gc.collect()
            if inputs.device.type == "cuda":
                torch.cuda.empty_cache()
            print(
                f"XAI OOM: retry method={method} with batch={batch_size}",
                flush=True,
            )
            continue
        maps.append(computed.cpu())
        del computed
        start = stop
        chunk_index += 1
    if inputs.device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(maps)


def _normalization(config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    if config.data.normalize == "imagenet":
        return np.asarray(IMAGENET_MEAN), np.asarray(IMAGENET_STD)
    if config.data.normalize == "pcam":
        return np.asarray(PCAM_MEAN), np.asarray(PCAM_STD)
    if config.data.normalize == "half":
        return np.asarray((0.5,) * 3), np.asarray((0.5,) * 3)
    raise ValueError(f"Unknown normalization: {config.data.normalize}")


def _denormalize(inputs: torch.Tensor, config: ExperimentConfig) -> np.ndarray:
    mean, std = _normalization(config)
    images = inputs.detach().cpu().numpy().transpose(0, 2, 3, 1)
    return np.clip(images * std[None, None, None] + mean[None, None, None], 0, 1)


def _mask_provider(args: argparse.Namespace) -> CamelyonMaskProvider | None:
    if args.annotations is None:
        return None
    mode = args.coordinate_mode
    downsample = args.downsample
    if args.mapping_audit is not None:
        audit = json.loads(args.mapping_audit.read_text(encoding="utf-8"))
        mode = mode or str(audit["best"]["coordinate_mode"])
        downsample = downsample or float(audit["best"]["downsample"])
    if mode is None or downsample is None:
        raise SystemExit(
            "Annotations require --mapping-audit or explicit "
            "--coordinate-mode and --downsample."
        )
    return CamelyonMaskProvider(
        args.annotations,
        coordinate_mode=mode,
        downsample=downsample,
    )


def _safe_mean(rows: list[dict[str, object]], key: str) -> float:
    values = [
        float(row[key])
        for row in rows
        if key in row and row[key] not in {None, ""} and math.isfinite(float(row[key]))
    ]
    return fmean(values) if values else float("nan")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _numeric_values(rows: list[dict[str, object]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(key)
        if value in {None, ""}:
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            values.append(numeric)
    return np.asarray(values, dtype=np.float64)


def _bootstrap_mean(
    values: np.ndarray,
    seed: int,
    repeats: int = 2000,
) -> tuple[float, float, float]:
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    if values.size == 1:
        return mean, mean, mean
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, values.size, size=(repeats, values.size))
    estimates = values[indices].mean(axis=1)
    low, high = np.quantile(estimates, (0.025, 0.975))
    return mean, float(low), float(high)


def _aggregate_rows(
    metric_rows: list[dict[str, object]],
    stability_rows: list[dict[str, object]],
    sanity_rows: list[dict[str, object]],
    seed: int,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    pairs = sorted({(str(row["model"]), str(row["method"])) for row in metric_rows})
    metric_sources = {
        "center_mass_ratio": metric_rows,
        "mass_inside": metric_rows,
        "pixel_auprc": metric_rows,
        "pointing_game": metric_rows,
        "deletion_auc": metric_rows,
        "insertion_auc": metric_rows,
        "d4_pearson": stability_rows,
        "sanity_full_pearson": [
            row for row in sanity_rows if row.get("scope") == "full"
        ],
    }
    for pair_index, (model_key, method) in enumerate(pairs):
        for metric_index, (metric, source) in enumerate(metric_sources.items()):
            source_key = (
                "pearson"
                if metric in {"d4_pearson", "sanity_full_pearson"}
                else metric
            )
            selected = [
                row
                for row in source
                if str(row.get("model")) == model_key
                and str(row.get("method")) == method
            ]
            values = _numeric_values(selected, source_key)
            mean, low, high = _bootstrap_mean(
                values,
                seed + pair_index * 101 + metric_index,
            )
            output.append(
                {
                    "model": model_key,
                    "method": method,
                    "metric": metric,
                    "n": int(values.size),
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return output


def _plot_aggregate_panels(
    output: Path,
    aggregate_rows: list[dict[str, object]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_keys = sorted({str(row["model"]) for row in aggregate_rows})
    methods = ("gradcam", "ig", "rise", "native")
    colors = {"gradcam": "#2563EB", "ig": "#F59E0B", "rise": "#10B981", "native": "#9333EA"}

    def panel(
        filename: str,
        specifications: tuple[tuple[str, str, str], ...],
    ) -> None:
        figure, axes = plt.subplots(
            1,
            len(specifications),
            figsize=(6.0 * len(specifications), 4.2),
            squeeze=False,
        )
        x = np.arange(len(model_keys), dtype=np.float64)
        width = 0.19
        lookup = {
            (str(row["model"]), str(row["method"]), str(row["metric"])): row
            for row in aggregate_rows
        }
        for axis, (metric, title, direction) in zip(
            axes[0], specifications, strict=True
        ):
            for method_index, method in enumerate(methods):
                means, lower, upper = [], [], []
                for model_key in model_keys:
                    row = lookup.get((model_key, method, metric))
                    mean = float(row["mean"]) if row is not None else float("nan")
                    low = float(row["ci95_low"]) if row is not None else float("nan")
                    high = float(row["ci95_high"]) if row is not None else float("nan")
                    means.append(mean)
                    lower.append(max(0.0, mean - low) if math.isfinite(mean) else 0.0)
                    upper.append(max(0.0, high - mean) if math.isfinite(mean) else 0.0)
                positions = x + (method_index - 1.5) * width
                axis.bar(
                    positions,
                    means,
                    width,
                    label=method.upper(),
                    color=colors[method],
                    alpha=0.9,
                    yerr=np.asarray((lower, upper)),
                    capsize=3,
                )
            axis.set_title(f"{title}\n({direction})", fontsize=10)
            axis.set_xticks(x, model_keys)
            axis.set_ylim(-1.02 if metric == "sanity_full_pearson" else 0, 1.02)
            axis.grid(axis="y", alpha=0.25)
            axis.set_axisbelow(True)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
        figure.tight_layout(rect=(0, 0, 1, 0.91))
        figure.savefig(output / filename, dpi=200, bbox_inches="tight")
        plt.close(figure)

    panel(
        "aggregate_localization.png",
        (
            ("pixel_auprc", "Pixel AP", "higher is better"),
            ("mass_inside", "Response mass in tumor mask", "higher is better"),
            ("pointing_game", "Pointing game", "higher is better"),
        ),
    )
    panel(
        "aggregate_faithfulness.png",
        (
            ("deletion_auc", "Deletion AUC", "lower is better"),
            ("insertion_auc", "Insertion AUC", "higher is better"),
        ),
    )
    panel(
        "aggregate_robustness.png",
        (
            ("d4_pearson", "D4 correlation", "higher is better"),
            ("sanity_full_pearson", "Correlation after randomization", "closer to 0 = lower similarity"),
        ),
    )


def _plot_qualitative(
    output: Path,
    model_key: str,
    images: np.ndarray,
    masks: list[np.ndarray | None],
    maps: dict[str, torch.Tensor],
    sample_ids: list[int],
    predictions: dict[int, tuple[int, float]],
    maximum_samples: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    count = min(maximum_samples, len(sample_ids))
    methods = list(maps)
    columns = 2 + len(methods)
    figure, axes = plt.subplots(count, columns, figsize=(2.0 * columns, 2.55 * count + 0.9),
                                squeeze=False)
    figure.subplots_adjust(left=0.025, right=0.99, top=0.91, bottom=0.055,
                           wspace=0.10, hspace=0.40)
    names = {"gradcam": "Grad-CAM", "ig": "IG", "rise": "RISE", "native": "Native map"}
    headings = ["Image + boundary", "Tumor mask"] + [names.get(m, m) for m in methods]
    for axis, heading in zip(axes[0], headings, strict=True):
        position = axis.get_position()
        figure.text((position.x0 + position.x1) / 2, 0.976, heading,
                    ha="center", va="top", fontsize=15, weight="bold")
    categories = {"tp": "true positive", "tn": "true negative",
                  "fp": "false positive", "fn": "false negative"}
    for row_index in range(count):
        sample_id = sample_ids[row_index]
        target, probability = predictions[sample_id]
        first = axes[row_index, 0]
        first.imshow(images[row_index], interpolation="nearest")
        position = first.get_position()
        probability_text = f"{probability:.3f}"
        figure.text(position.x0, position.y1 + 0.012,
                    f"ID {sample_id}  |  label: {target}  |  p(tumor) = {probability_text}"
                    f"  |  {categories[_category(target, probability)]} (threshold 0.5)",
                    fontsize=15, color="#991B1B" if _category(target, probability) in {"fn", "fp"} else "#172554")
        mask = masks[row_index]
        if mask is not None:
            axes[row_index, 1].imshow(mask, cmap=ListedColormap(["#FFFFFF", "#00A8B5"]),
                                     vmin=0, vmax=1, interpolation="nearest")
            # Pad so a mask covering the whole patch still has a visible boundary.
            padded = np.pad(mask, 1)
            first.contour(np.arange(-1, mask.shape[1] + 1), np.arange(-1, mask.shape[0] + 1),
                          padded, levels=[0.5], colors=["#00FFFF"], linewidths=1.4)
            first.set_xlim(-0.5, mask.shape[1] - 0.5)
            first.set_ylim(mask.shape[0] - 0.5, -0.5)
        else:
            axes[row_index, 1].text(0.5, 0.5, "No annotation", ha="center", va="center",
                                     transform=axes[row_index, 1].transAxes, fontsize=12)
        for column_index, method in enumerate(methods, start=2):
            axes[row_index, column_index].imshow(images[row_index])
            axes[row_index, column_index].imshow(
                maps[method][row_index],
                cmap="inferno",
                alpha=0.55,
                vmin=0,
                vmax=1,
            )
        height, width = images[row_index].shape[:2]
        for column_index, axis in enumerate(axes[row_index]):
            if column_index != 1 or mask is not None:
                axis.add_patch(Rectangle((width / 3 - 0.5, height / 3 - 0.5), width / 3, height / 3,
                                         fill=False, edgecolor="black" if column_index == 1 else "white",
                                         linewidth=1.1, linestyle="--"))
            axis.axis("off")
    figure.text(0.5, 0.035, "Cyan: tumor annotation   |   Dashed square: central 32 × 32 pixels",
                ha="center", fontsize=15)
    figure.text(0.5, 0.012, "Maps: bright/yellow areas show stronger relative responses; they are not tumor masks.",
                ha="center", fontsize=15)
    figure.savefig(output / f"qualitative_{model_key}.png", dpi=240, facecolor="white")
    plt.close(figure)


def _sanity_checks(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    reference_maps: dict[str, torch.Tensor],
    methods: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, object]]:
    count = min(args.sanity_samples, inputs.shape[0])
    if count <= 0:
        return []
    rows: list[dict[str, object]] = []
    for scope in ("classifier", "full"):
        randomized = randomized_model(model, scope, args.seed).to(device).eval()
        for method in methods:
            if method == "native" and not hasattr(randomized, "evidence_map"):
                continue
            randomized_maps = _batched_attribution(
                method,
                randomized,
                inputs[:count],
                targets[:count],
                args,
            )
            for index in range(count):
                rows.append(
                    {
                        "scope": scope,
                        "method": method,
                        "sample_index": index,
                        **map_similarity(
                            reference_maps[method][index],
                            randomized_maps[index],
                        ),
                    }
                )
        del randomized
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def main() -> None:
    args = parse_arguments()
    _validate_arguments(args)
    project_root = args.project_root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    metadata_path = (
        project_root
        / "data"
        / "pcam"
        / f"camelyonpatch_level_2_split_{args.split}_meta.csv"
    )
    metadata = PCamMetadata(metadata_path)
    provider = _mask_provider(args)

    reference_spec = MODEL_SPECS[args.models[0]]
    reference_predictions_path = (
        project_root / "runs" / reference_spec.run / "test_predictions.csv"
    )
    if args.split != "test":
        raise SystemExit(
            "Stratified XAI selection currently requires saved test_predictions.csv."
        )
    reference_predictions = _read_predictions(reference_predictions_path)
    sample_ids = _stratified_sample_ids(
        reference_predictions,
        metadata,
        args.samples,
        args.seed,
    )
    (args.output / "selected_samples.json").write_text(
        json.dumps(sample_ids, indent=2),
        encoding="utf-8",
    )

    metric_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    sanity_rows: list[dict[str, object]] = []
    model_reports: dict[str, object] = {}
    started = time.perf_counter()

    for model_key in args.models:
        model_started = time.perf_counter()
        metric_start = len(metric_rows)
        stability_start = len(stability_rows)
        sanity_start = len(sanity_rows)
        spec = MODEL_SPECS[model_key]
        completed_paths = {
            "metric": args.output / f"xai_metrics_{model_key}.csv",
            "stability": args.output / f"stability_metrics_{model_key}.csv",
            "sanity": args.output / f"sanity_metrics_{model_key}.csv",
            "figure": args.output / f"qualitative_{model_key}.png",
        }
        if args.resume_existing and all(path.is_file() for path in completed_paths.values()):
            reused_metrics = _read_csv_rows(completed_paths["metric"])
            reused_stability = _read_csv_rows(completed_paths["stability"])
            reused_sanity = _read_csv_rows(completed_paths["sanity"])
            metric_rows.extend(reused_metrics)
            stability_rows.extend(reused_stability)
            sanity_rows.extend(reused_sanity)
            methods = sorted({str(row["method"]) for row in reused_metrics})
            model_reports[model_key] = {
                "config": str((project_root / "configs" / spec.config).resolve()),
                "checkpoint": str(
                    (project_root / "runs" / spec.run / "best.pt").resolve()
                ),
                "parameters": None,
                "methods": methods,
                "runtime_seconds": 0.0,
                "reused_existing": True,
            }
            print(f"model={model_key} reused existing XAI outputs", flush=True)
            continue
        predictions = _read_predictions(
            project_root / "runs" / spec.run / "test_predictions.csv"
        )
        model, config, checkpoint_path = _load_model(project_root, spec, device)
        data_root = project_root / config.data.root
        transform = build_transform(
            config.data.image_size,
            "none",
            config.data.normalize,
        )
        dataset = PCamH5Dataset(data_root, args.split, transform=transform)
        items = [dataset[sample_id] for sample_id in sample_ids]
        inputs = torch.stack([item[0] for item in items]).to(device)
        target_classes = _target_classes(args, sample_ids, predictions, device)

        methods = [method for method in args.methods if method != "native"]
        if "native" in args.methods and hasattr(model, "evidence_map"):
            methods.append("native")
        saliency_maps: dict[str, torch.Tensor] = {}
        for method in methods:
            print(f"model={model_key} method={method} samples={len(sample_ids)}", flush=True)
            saliency_maps[method] = _batched_attribution(
                method,
                model,
                inputs,
                target_classes,
                args,
            )

        tumor_masks: list[np.ndarray | None] = []
        mask_available: list[bool] = []
        for sample_id in sample_ids:
            if provider is None:
                tumor_masks.append(None)
                mask_available.append(False)
            else:
                mask, available = provider.mask(metadata[sample_id])
                tumor_masks.append(mask if available else None)
                mask_available.append(available)

        for sample_index, sample_id in enumerate(sample_ids):
            target, probability = predictions[sample_id]
            predicted = int(probability >= 0.5)
            for method, maps in saliency_maps.items():
                row: dict[str, object] = {
                    "model": model_key,
                    "method": method,
                    "sample_id": sample_id,
                    "wsi": metadata[sample_id].wsi,
                    "target": target,
                    "predicted": predicted,
                    "probability": probability,
                    "category": _category(target, probability),
                    "explained_class": int(target_classes[sample_index].item()),
                    "center_mass_ratio": center_mass_ratio(maps[sample_index]),
                    "mask_available": mask_available[sample_index],
                }
                if tumor_masks[sample_index] is not None:
                    row.update(
                        localization_metrics(
                            maps[sample_index],
                            tumor_masks[sample_index],
                        )
                    )
                if not args.skip_faithfulness:
                    row.update(
                        perturbation_faithfulness(
                            model,
                            inputs[sample_index],
                            maps[sample_index].to(device),
                            int(target_classes[sample_index].item()),
                            steps=args.faithfulness_steps,
                        )
                    )
                metric_rows.append(row)

        stability_methods = [
            method for method in methods if method in args.stability_methods
        ]
        for group_index in args.stability_transforms:
            transformed_inputs = apply_d4(inputs, group_index)
            for method in stability_methods:
                transformed_maps = _batched_attribution(
                    method,
                    model,
                    transformed_inputs,
                    target_classes,
                    args,
                )
                aligned_maps = undo_d4(transformed_maps, group_index)
                for sample_index, sample_id in enumerate(sample_ids):
                    stability_rows.append(
                        {
                            "model": model_key,
                            "method": method,
                            "sample_id": sample_id,
                            "d4_index": group_index,
                            **map_similarity(
                                saliency_maps[method][sample_index],
                                aligned_maps[sample_index],
                            ),
                        }
                    )

        sanity_methods = [
            method for method in methods if method in {"gradcam", "ig", "native"}
        ]
        for row in _sanity_checks(
            model,
            inputs,
            target_classes,
            saliency_maps,
            sanity_methods,
            args,
            device,
        ):
            sanity_rows.append({"model": model_key, **row})

        _plot_qualitative(
            args.output,
            model_key,
            _denormalize(inputs, config),
            tumor_masks,
            saliency_maps,
            sample_ids,
            predictions,
            args.max_figure_samples,
        )
        # Preserve completed models even if a later architecture fails or the
        # analysis is interrupted. These files also make long audits easier to
        # inspect while they are still running.
        _write_csv(
            args.output / f"xai_metrics_{model_key}.csv",
            metric_rows[metric_start:],
        )
        _write_csv(
            args.output / f"stability_metrics_{model_key}.csv",
            stability_rows[stability_start:],
        )
        _write_csv(
            args.output / f"sanity_metrics_{model_key}.csv",
            sanity_rows[sanity_start:],
        )
        model_reports[model_key] = {
            "config": str((project_root / "configs" / spec.config).resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
            "parameters": parameter_count(model),
            "methods": methods,
            "runtime_seconds": time.perf_counter() - model_started,
        }
        print(
            f"model={model_key} completed "
            f"runtime={model_reports[model_key]['runtime_seconds']:.1f}s",
            flush=True,
        )
        dataset.close()
        del model, inputs, saliency_maps
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_csv(args.output / "xai_metrics.csv", metric_rows)
    _write_csv(args.output / "stability_metrics.csv", stability_rows)
    _write_csv(args.output / "sanity_metrics.csv", sanity_rows)
    aggregate_rows = _aggregate_rows(
        metric_rows,
        stability_rows,
        sanity_rows,
        args.seed,
    )
    _write_csv(args.output / "aggregate_metrics.csv", aggregate_rows)
    _plot_aggregate_panels(args.output, aggregate_rows)
    summary: dict[str, object] = {
        "created_at": datetime.now(UTC).isoformat(),
        "project_root": str(project_root),
        "device": str(device),
        "models": model_reports,
        "sample_count": len(sample_ids),
        "selected_sample_ids": sample_ids,
        "target_mode": args.target_mode,
        "settings": {
            "ig_steps": args.ig_steps,
            "rise_masks": args.rise_masks,
            "faithfulness_steps": args.faithfulness_steps,
            "stability_transforms": args.stability_transforms,
            "sanity_samples": args.sanity_samples,
        },
        "runtime_seconds": time.perf_counter() - started,
        "aggregates": {},
    }
    for model_key in args.models:
        for method in sorted({str(row["method"]) for row in metric_rows if row["model"] == model_key}):
            rows = [
                row
                for row in metric_rows
                if row["model"] == model_key and row["method"] == method
            ]
            stability = [
                row
                for row in stability_rows
                if row["model"] == model_key and row["method"] == method
            ]
            sanity = [
                row
                for row in sanity_rows
                if row["model"] == model_key and row["method"] == method
            ]
            summary["aggregates"][f"{model_key}:{method}"] = {
                "sample_count": len(rows),
                "localization_sample_count": sum(
                    1
                    for row in rows
                    if "pixel_auprc" in row
                    and row["pixel_auprc"] not in {None, ""}
                    and math.isfinite(float(row["pixel_auprc"]))
                ),
                "stability_comparison_count": len(stability),
                "sanity_comparison_count": len(sanity),
                "center_mass_ratio": _safe_mean(rows, "center_mass_ratio"),
                "mass_inside": _safe_mean(rows, "mass_inside"),
                "pixel_auprc": _safe_mean(rows, "pixel_auprc"),
                "pointing_game": _safe_mean(rows, "pointing_game"),
                "deletion_auc": _safe_mean(rows, "deletion_auc"),
                "insertion_auc": _safe_mean(rows, "insertion_auc"),
                "d4_pearson": _safe_mean(stability, "pearson"),
                "d4_top_iou": _safe_mean(stability, "top_iou"),
                "sanity_full_pearson": _safe_mean(
                    [row for row in sanity if row["scope"] == "full"],
                    "pearson",
                ),
            }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )

    lines = [
        "# PCam explainability audit",
        "",
        f"Generated: {summary['created_at']}",
        "",
        f"Sample: {len(sample_ids)} patches; target mode: `{args.target_mode}`.",
        "",
        "| Model and method | n | n mask | Center mass | Tumor mass | Pixel AP | Pointing | Deletion AUC | Insertion AUC | D4 corr. | Sanity corr. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, values in summary["aggregates"].items():
        def fmt(name: str) -> str:
            value = float(values[name])
            return "—" if not math.isfinite(value) else f"{value:.4f}"

        lines.append(
            f"| {key} | {values['sample_count']} | "
            f"{values['localization_sample_count']} | "
            f"{fmt('center_mass_ratio')} | {fmt('mass_inside')} | "
            f"{fmt('pixel_auprc')} | {fmt('pointing_game')} | "
            f"{fmt('deletion_auc')} | {fmt('insertion_auc')} | "
            f"{fmt('d4_pearson')} | {fmt('sanity_full_pearson')} |"
        )
    lines.extend(
        [
            "",
            "Attribution maps describe model behavior and are not independent diagnostic evidence. Quantitative tumor localization requires correctly mapped CAMELYON16 masks. The pixel_auprc CSV field contains AP with equal scores grouped at one threshold, not trapezoidal PR AUC. A dash in a correlation column denotes a missing or undefined measurement; consult the detailed files to distinguish these cases.",
        ]
    )
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report={args.output.resolve()} runtime={summary['runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()
