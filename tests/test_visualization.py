from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")

from anomaliz.visualization.dashboard import (
    generate_report,
    plot_comparison_summary,
    plot_metrics_comparison,
    plot_roc_curves,
    plot_seed_stability,
)


@pytest.fixture(scope="module")
def metrics(trained_bundle_dir) -> dict:
    return json.loads((trained_bundle_dir / "metrics.json").read_text())


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def test_plot_roc_curves_returns_figure(metrics):
    fig = plot_roc_curves(metrics)
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    # One line per detector that has roc_curve data, plus the reference diagonal
    assert len(ax.lines) >= 2
    plt.close(fig)


def test_plot_metrics_comparison_returns_figure(metrics):
    fig = plot_metrics_comparison(metrics)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_seed_stability_returns_figure(metrics):
    fig = plot_seed_stability(metrics)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_comparison_summary_returns_figure(metrics):
    fig = plot_comparison_summary(metrics)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plots_accept_external_axes(metrics):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for fn, ax in zip(
        [plot_roc_curves, plot_metrics_comparison, plot_seed_stability, plot_comparison_summary],
        axes,
    ):
        returned_fig = fn(metrics, ax=ax)
        assert returned_fig is fig
    plt.close(fig)


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------

def test_generate_report_writes_all_pngs(trained_bundle_dir, tmp_path):
    paths = generate_report(trained_bundle_dir, out_dir=tmp_path)
    expected = {"roc_curves.png", "metrics_comparison.png", "seed_stability.png", "comparison_summary.png"}
    assert {p.name for p in paths} == expected
    for p in paths:
        assert p.exists() and p.stat().st_size > 0


def test_generate_report_creates_out_dir(trained_bundle_dir, tmp_path):
    out = tmp_path / "nested" / "reports"
    generate_report(trained_bundle_dir, out_dir=out)
    assert out.is_dir()


def test_generate_report_raises_on_missing_metrics(tmp_path):
    with pytest.raises(FileNotFoundError):
        generate_report(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# Graceful handling of sparse metrics (no seed_evaluation)
# ---------------------------------------------------------------------------

def test_seed_stability_handles_missing_aggregate(tmp_path):
    sparse = {"isolation_forest": {"f1": 0.7}, "fused": {"f1": 0.71}}
    fig = plot_seed_stability(sparse)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_roc_curves_handles_missing_roc_data():
    no_roc = {det: {"f1": 0.7, "roc_auc": 0.85} for det in
               ("isolation_forest", "lstm_autoencoder", "lstm_forecaster", "fused")}
    fig = plot_roc_curves(no_roc)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
