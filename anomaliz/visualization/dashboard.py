"""Offline visualization dashboard for an Anomaliz artifact bundle.

Usage::

    python -m anomaliz.visualization.dashboard --bundle artifacts/dev/ --out reports/
    python -m anomaliz.visualization.dashboard --bundle artifacts/dev/ --show
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # non-interactive backend; overridden by --show

_DETECTORS = ("isolation_forest", "lstm_autoencoder", "lstm_forecaster", "fused")
_LABELS = {
    "isolation_forest": "Isolation Forest",
    "lstm_autoencoder": "LSTM Autoencoder",
    "lstm_forecaster": "LSTM Forecaster",
    "fused": "Fused (IF + AE)",
}
_COLORS = {
    "isolation_forest": "#4c72b0",
    "lstm_autoencoder": "#dd8452",
    "lstm_forecaster": "#55a868",
    "fused": "#c44e52",
}
_METRIC_LABELS = {"f1": "F1", "precision": "Precision", "recall": "Recall", "roc_auc": "ROC AUC"}


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def plot_roc_curves(metrics: dict[str, Any], ax: plt.Axes | None = None) -> plt.Figure:
    """Overlay ROC curves for all detectors on a single axes."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    ax.plot([0, 1], [0, 1], color="lightgrey", linestyle="--", linewidth=1)
    for det in _DETECTORS:
        m = metrics.get(det, {})
        rc = m.get("roc_curve")
        auc = m.get("roc_auc")
        if rc is None:
            continue
        label = f"{_LABELS[det]}"
        if auc is not None:
            label += f" (AUC={auc:.3f})"
        ax.plot(rc["fpr"], rc["tpr"], color=_COLORS[det], linewidth=2, label=label)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Detector Comparison")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower right", fontsize=8)
    if standalone:
        fig.tight_layout()
    return fig


def plot_metrics_comparison(metrics: dict[str, Any], ax: plt.Axes | None = None) -> plt.Figure:
    """Grouped bar chart: F1, Precision, Recall for each detector."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    metric_keys = ["f1", "precision", "recall"]
    n_metrics = len(metric_keys)
    n_detectors = len(_DETECTORS)
    x = np.arange(n_metrics)
    width = 0.18
    offsets = np.linspace(-(n_detectors - 1) / 2, (n_detectors - 1) / 2, n_detectors) * width

    for i, det in enumerate(_DETECTORS):
        m = metrics.get(det, {})
        vals = [m.get(k, 0.0) or 0.0 for k in metric_keys]
        bars = ax.bar(x + offsets[i], vals, width, label=_LABELS[det], color=_COLORS[det], alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0.05:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([_METRIC_LABELS[k] for k in metric_keys])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Detection Metrics — Detector Comparison")
    ax.legend(fontsize=8)
    if standalone:
        fig.tight_layout()
    return fig


def plot_seed_stability(metrics: dict[str, Any], ax: plt.Axes | None = None) -> plt.Figure:
    """Error-bar chart showing F1 mean ± std across seeds for each detector."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.get_figure()

    agg = metrics.get("seed_evaluation", {}).get("aggregate", {})
    if not agg:
        if standalone:
            ax.set_title("Seed stability data unavailable")
        return fig

    x = np.arange(len(_DETECTORS))
    means = []
    stds = []
    for det in _DETECTORS:
        f1_data = agg.get(det, {}).get("f1", {})
        means.append(f1_data.get("mean") or 0.0)
        stds.append(f1_data.get("std") or 0.0)

    colors = [_COLORS[d] for d in _DETECTORS]
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, capsize=5, width=0.5,
                  error_kw={"elinewidth": 1.5, "ecolor": "dimgrey"})
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 0.015,
            f"{mean:.3f}±{std:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_LABELS[d] for d in _DETECTORS], rotation=12, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1 Score")
    ax.set_title(f"Multi-seed F1 Stability (n={next(iter(agg.values()))['f1'].get('n', '?')} seeds)")
    if standalone:
        fig.tight_layout()
    return fig


def plot_comparison_summary(metrics: dict[str, Any], ax: plt.Axes | None = None) -> plt.Figure:
    """Table summarising the comparison_summary verdict."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 2))
    else:
        fig = ax.get_figure()

    summary = metrics.get("comparison_summary", {})
    verdict = summary.get("verdict", {})
    baseline = summary.get("baseline_if_f1")
    agg = metrics.get("seed_evaluation", {}).get("aggregate", {})

    rows = []
    for det in ("lstm_autoencoder", "lstm_forecaster", "fused"):
        f1_data = agg.get(det, {}).get("f1", {})
        mean = f1_data.get("mean")
        std = f1_data.get("std")
        v = verdict.get(det, "—")
        color = "#55a868" if v == "beats_baseline" else "#dd8452"
        mean_str = f"{mean:.3f}±{std:.3f}" if mean is not None else "—"
        rows.append((_LABELS[det], mean_str, v, color))

    baseline_str = f"{baseline:.3f}" if baseline is not None else "—"
    ax.axis("off")
    col_labels = ["Detector", f"F1 (baseline IF={baseline_str})", "Verdict"]
    cell_text = [[r[0], r[1], r[2]] for r in rows]
    cell_colors = [["white", "white", r[3]] for r in rows]
    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    ax.set_title("Comparison Summary vs Isolation Forest Baseline", pad=12, fontsize=10)
    if standalone:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def generate_report(bundle_dir: Path, out_dir: Path | None = None, show: bool = False) -> list[Path]:
    """Read metrics.json from bundle_dir and write all plots to out_dir.

    Returns the list of written file paths.
    """
    bundle_dir = Path(bundle_dir)
    metrics_path = bundle_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found in {bundle_dir}")
    metrics = json.loads(metrics_path.read_text())

    if show:
        matplotlib.use("TkAgg")

    written: list[Path] = []

    plots = [
        ("roc_curves.png", plot_roc_curves),
        ("metrics_comparison.png", plot_metrics_comparison),
        ("seed_stability.png", plot_seed_stability),
        ("comparison_summary.png", plot_comparison_summary),
    ]

    for fname, plot_fn in plots:
        fig = plot_fn(metrics)
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            dest = out_dir / fname
            fig.savefig(dest, dpi=120, bbox_inches="tight")
            written.append(dest)
        if show:
            plt.show()
        plt.close(fig)

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate Anomaliz visualisation report.")
    parser.add_argument("--bundle", type=Path, required=True, help="Path to artifact bundle directory.")
    parser.add_argument("--out", type=Path, default=None, help="Output directory for PNG files.")
    parser.add_argument("--show", action="store_true", help="Display plots interactively (requires display).")
    args = parser.parse_args(argv)

    if args.out is None and not args.show:
        args.out = args.bundle / "reports"

    paths = generate_report(args.bundle, out_dir=args.out, show=args.show)
    for p in paths:
        print(f"  Written: {p}")


if __name__ == "__main__":
    main()
