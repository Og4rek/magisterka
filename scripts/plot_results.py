"""Plot published classification summaries without loading data or checkpoints."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=ROOT / "results/classification.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "assets/results_overview.png")
    args = parser.parse_args()
    with args.table.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: float(row["test_auc_roc"]))
    labels = [f"{r['experiment'].upper()}  {r['model']}" for r in rows]
    colors = ["#e49037" if r["experiment"] == "e17" else "#237d91" if r["experiment"] == "e07" else "#4265a9" for r in rows]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.6), sharey=True, gridspec_kw={"width_ratios": [1.25, 1]}, layout="constrained")
    values = [float(r["test_auc_roc"]) for r in rows]
    sizes = [int(r["parameters"]) for r in rows]
    axes[0].barh(labels, values, color=colors, height=0.67)
    axes[0].set(xlim=(0.89, 0.98), xlabel="Test ROC AUC (axis starts at 0.89)", title="Discrimination")
    for i, value in enumerate(values):
        axes[0].text(value + .0006, i, f"{value:.4f}", va="center", fontsize=8)
    axes[1].scatter(sizes, range(len(rows)), color=colors, s=48, zorder=3)
    axes[1].set(xscale="log", xlim=(18000, 6e7), xlabel="Total parameters · logarithmic scale", title="Model size")
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e6:g}M" if x >= 1e6 else f"{x / 1e3:g}K"))
    for i, value in enumerate(sizes):
        axes[1].annotate(f"{value:,}", (value, i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8)
    for ax in axes:
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", color="#e2e6ed", linewidth=.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    fig.suptitle("PCam: recorded classification runs", fontsize=19, fontweight="bold", x=.46)
    fig.supxlabel("Official test split · 32,768 patches · single runs with different training regimes\nOrange: PCam-D4Loc | Teal: G-DenseNet D4 | No multi-seed or clinical-performance claim", fontsize=10, color="#4c596c")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
