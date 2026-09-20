"""Find and validate the PCam-to-CAMELYON16 annotation coordinate mapping."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from pcam_repro.xai.masks import CamelyonMaskProvider, PCamMetadata


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/pcam/camelyonpatch_level_2_split_valid_meta.csv"),
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/camelyon16/annotations"),
    )
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=("top_left:4", "center:4", "top_left:1", "center:1"),
        metavar="MODE:DOWNSAMPLE",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/xai/annotation_mapping_audit.json"),
    )
    parser.add_argument("--minimum-agreement", type=float, default=0.95)
    parser.add_argument("--allow-low-agreement", action="store_true")
    return parser.parse_args()


def _candidate(specification: str) -> tuple[str, float]:
    try:
        mode, downsample_text = specification.split(":", 1)
        downsample = float(downsample_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Invalid candidate '{specification}', expected MODE:DOWNSAMPLE."
        ) from error
    if mode not in {"top_left", "center"} or downsample <= 0:
        raise argparse.ArgumentTypeError(f"Invalid coordinate candidate: {specification}")
    return mode, downsample


def _select_records(
    metadata: PCamMetadata,
    provider: CamelyonMaskProvider,
    sample_size: int,
    seed: int,
):
    available = [
        record
        for record in metadata.records
        if provider.annotation_path(record.wsi) is not None
    ]
    if not available:
        raise RuntimeError("No metadata WSI has a matching annotation XML file.")
    positive = [record for record in available if record.center_tumor_patch]
    negative = [record for record in available if not record.center_tumor_patch]
    rng = np.random.default_rng(seed)
    half = max(1, sample_size // 2)
    chosen = []
    for group in (positive, negative):
        count = min(len(group), half)
        indices = rng.choice(len(group), size=count, replace=False)
        chosen.extend(group[int(index)] for index in indices)
    rng.shuffle(chosen)
    return chosen


def _evaluate(provider: CamelyonMaskProvider, records) -> dict[str, object]:
    tp = tn = fp = fn = 0
    full_patch_matches = 0
    mismatches: list[int] = []
    for record in records:
        mask, available = provider.mask(record)
        if not available:
            continue
        center_prediction = provider.center_contains_tumor(mask)
        expected = record.center_tumor_patch
        if center_prediction and expected:
            tp += 1
        elif not center_prediction and not expected:
            tn += 1
        elif center_prediction:
            fp += 1
            mismatches.append(record.sample_id)
        else:
            fn += 1
            mismatches.append(record.sample_id)
        full_patch_matches += int(bool(mask.any()) == record.tumor_patch)
    count = tp + tn + fp + fn
    return {
        "count": count,
        "center_agreement": (tp + tn) / max(1, count),
        "center_sensitivity": tp / max(1, tp + fn),
        "center_specificity": tn / max(1, tn + fp),
        "full_patch_agreement": full_patch_matches / max(1, count),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "mismatch_sample_ids": mismatches[:50],
    }


def main() -> None:
    args = parse_arguments()
    metadata = PCamMetadata(args.metadata)
    seed_provider = CamelyonMaskProvider(args.annotations)
    records = _select_records(metadata, seed_provider, args.sample_size, args.seed)
    evaluations = []
    for specification in args.candidates:
        mode, downsample = _candidate(specification)
        provider = CamelyonMaskProvider(
            args.annotations,
            downsample=downsample,
            coordinate_mode=mode,
        )
        evaluations.append(
            {
                "coordinate_mode": mode,
                "downsample": downsample,
                **_evaluate(provider, records),
            }
        )
    best = max(evaluations, key=lambda item: float(item["center_agreement"]))
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "metadata": str(args.metadata.resolve()),
        "annotations": str(args.annotations.resolve()),
        "sample_size_requested": args.sample_size,
        "sample_size_used": len(records),
        "best": best,
        "candidates": evaluations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    for result in evaluations:
        print(
            f"mode={result['coordinate_mode']:8s} "
            f"downsample={result['downsample']:g} "
            f"center_agreement={result['center_agreement']:.4f} "
            f"full_patch_agreement={result['full_patch_agreement']:.4f}"
        )
    print(
        f"best={best['coordinate_mode']}:{best['downsample']:g} "
        f"agreement={best['center_agreement']:.4f} output={args.output.resolve()}"
    )
    if (
        float(best["center_agreement"]) < args.minimum_agreement
        and not args.allow_low_agreement
    ):
        raise SystemExit(
            "Annotation mapping agreement is below the required threshold; "
            "do not use these masks for quantitative XAI yet."
        )


if __name__ == "__main__":
    main()
