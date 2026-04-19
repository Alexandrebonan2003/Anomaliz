from __future__ import annotations

import json


def test_bundle_is_complete(trained_bundle_dir):
    d = trained_bundle_dir
    assert (d / "isolation_forest" / "model.pkl").exists()
    assert (d / "isolation_forest" / "score_stats.json").exists()
    assert (d / "lstm_autoencoder" / "model.keras").exists()
    assert (d / "lstm_autoencoder" / "stats.json").exists()
    assert (d / "lstm_forecaster" / "model.keras").exists()
    assert (d / "lstm_forecaster" / "stats.json").exists()
    assert (d / "normalizer.json").exists()
    assert (d / "threshold.json").exists()
    assert (d / "metadata.json").exists()
    assert (d / "metrics.json").exists()


def test_metrics_contents(trained_bundle_dir):
    metrics = json.loads((trained_bundle_dir / "metrics.json").read_text())
    for key in ("f1", "precision", "recall", "n_test_windows", "n_test_anomalies"):
        assert key in metrics
    assert 0.0 <= metrics["f1"] <= 1.0
    for detector in ("fused", "isolation_forest", "lstm_autoencoder", "lstm_forecaster"):
        assert detector in metrics
        for key in ("f1", "precision", "recall", "threshold"):
            assert key in metrics[detector]
            assert 0.0 <= metrics[detector][key] <= 1.0
    assert "seed_evaluation" in metrics
    assert "aggregate" in metrics["seed_evaluation"]
    assert "comparison_summary" in metrics
    assert "verdict" in metrics["comparison_summary"]


def test_metadata_snapshots_config(trained_bundle_dir):
    metadata = json.loads((trained_bundle_dir / "metadata.json").read_text())
    assert "config" in metadata
    assert metadata["config"]["data"]["window_size"] == 10
    assert "timestamp" in metadata
