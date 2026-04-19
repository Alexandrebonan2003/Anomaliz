from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from ..config.settings import Settings
from ..data.dataset import split_series
from ..data.generator import generate_series
from ..detection.scorer import fuse
from ..models.isolation_forest import IFDetector
from ..models.lstm_autoencoder import LSTMAutoencoder
from ..preprocessing.normalizer import MinMaxNormalizer
from ..preprocessing.windowing import (
    flatten_windows,
    make_windows,
    select_normal_windows,
)

FEATURES = ["cpu", "memory", "latency"]
IF_NAME = "isolation_forest"
LSTM_NAME = "lstm_autoencoder"


@dataclass
class TrainingResult:
    metrics: dict[str, Any]
    bundle_dir: Path


def run_training(settings: Settings, out_dir: Path) -> TrainingResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(settings.seed)

    df = generate_series(settings.data, rng)
    train_df, val_df, test_df = split_series(
        df, settings.data.train_ratio, settings.data.val_ratio
    )

    normalizer = MinMaxNormalizer().fit(train_df[FEATURES].to_numpy())
    X_train = normalizer.transform(train_df[FEATURES].to_numpy())
    X_val = normalizer.transform(val_df[FEATURES].to_numpy())
    X_test = normalizer.transform(test_df[FEATURES].to_numpy())

    W_train, y_train = make_windows(
        X_train, train_df["label"].to_numpy(), settings.data.window_size
    )
    W_val, y_val = make_windows(
        X_val, val_df["label"].to_numpy(), settings.data.window_size
    )
    W_test, y_test = make_windows(
        X_test, test_df["label"].to_numpy(), settings.data.window_size
    )

    class_balance = {
        "train": _class_balance(y_train),
        "val": _class_balance(y_val),
        "test": _class_balance(y_test),
    }

    if_detector = IFDetector(**settings.model.isolation_forest.model_dump())
    if_detector.fit(flatten_windows(W_train))

    lstm_cfg = settings.model.lstm_autoencoder
    lstm_detector = LSTMAutoencoder(
        units_1=lstm_cfg.units_1,
        units_2=lstm_cfg.units_2,
        dropout=lstm_cfg.dropout,
        learning_rate=lstm_cfg.learning_rate,
        epochs=lstm_cfg.epochs,
        batch_size=lstm_cfg.batch_size,
        patience=lstm_cfg.patience,
        random_state=settings.seed,
    )
    W_train_normal = select_normal_windows(W_train, y_train)
    if W_train_normal.shape[0] == 0:
        raise RuntimeError("No normal training windows available for LSTM training")
    lstm_detector.fit(W_train_normal)

    val_errors = lstm_detector.reconstruction_error(W_val)
    lstm_detector.set_error_range(float(val_errors.min()), float(val_errors.max()))
    val_normal_errors = val_errors[y_val == 0]
    if val_normal_errors.size > 0:
        lstm_error_threshold = float(
            val_normal_errors.mean() + lstm_cfg.threshold_k * val_normal_errors.std()
        )
    else:
        lstm_error_threshold = float(val_errors.mean())

    if_val_scores = if_detector.score(flatten_windows(W_val))
    lstm_val_scores = lstm_detector.score(W_val)
    fused_val_scores = fuse(
        if_val_scores,
        lstm_val_scores,
        settings.detection.fusion.weight_if,
        settings.detection.fusion.weight_lstm,
    )

    if_threshold, if_sweep = _tune_threshold(
        if_val_scores, y_val, settings.detection.tuning.n_thresholds, settings.detection.threshold
    )
    lstm_threshold, lstm_sweep = _tune_threshold(
        lstm_val_scores, y_val, settings.detection.tuning.n_thresholds, settings.detection.threshold
    )
    fused_threshold, fused_sweep = _tune_threshold(
        fused_val_scores, y_val, settings.detection.tuning.n_thresholds, settings.detection.threshold
    )

    if_test_scores = if_detector.score(flatten_windows(W_test))
    lstm_test_scores = lstm_detector.score(W_test)
    fused_test_scores = fuse(
        if_test_scores,
        lstm_test_scores,
        settings.detection.fusion.weight_if,
        settings.detection.fusion.weight_lstm,
    )

    metrics = {
        "fused": _metrics_for(fused_test_scores, y_test, fused_threshold),
        "isolation_forest": _metrics_for(if_test_scores, y_test, if_threshold),
        "lstm_autoencoder": _metrics_for(lstm_test_scores, y_test, lstm_threshold),
        "n_test_windows": int(len(y_test)),
        "n_test_anomalies": int(y_test.sum()),
        "class_balance": class_balance,
    }
    # Keep Phase 1-compatible top-level keys for backwards compatibility.
    fused_metrics = metrics["fused"]
    metrics["f1"] = fused_metrics["f1"]
    metrics["precision"] = fused_metrics["precision"]
    metrics["recall"] = fused_metrics["recall"]
    metrics["roc_auc"] = fused_metrics["roc_auc"]

    if_detector.save(out_dir / IF_NAME)
    lstm_detector.save(out_dir / LSTM_NAME)
    normalizer.save(out_dir / "normalizer.json")
    _write_json(
        out_dir / "threshold.json",
        {
            "decision_threshold": float(fused_threshold),
            "tuned_on": "validation",
            "selection_metric": "f1",
            "sweep": fused_sweep,
            "isolation_forest": {"threshold": float(if_threshold), "sweep": if_sweep},
            "lstm_autoencoder": {
                "threshold": float(lstm_threshold),
                "sweep": lstm_sweep,
                "error_threshold": lstm_error_threshold,
                "error_min": float(val_errors.min()),
                "error_max": float(val_errors.max()),
            },
            "fusion": {
                "weight_if": settings.detection.fusion.weight_if,
                "weight_lstm": settings.detection.fusion.weight_lstm,
            },
        },
    )
    _write_json(
        out_dir / "metadata.json",
        {
            "config": settings.model_dump(),
            "git_sha": _git_sha(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "detectors": [IF_NAME, LSTM_NAME],
        },
    )
    _write_json(out_dir / "metrics.json", metrics)

    return TrainingResult(metrics=metrics, bundle_dir=out_dir)


def _metrics_for(scores: np.ndarray, y: np.ndarray, threshold: float) -> dict[str, Any]:
    preds = (scores > threshold).astype(int)
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores)) if y.max() > 0 else None,
    }


def _class_balance(y: np.ndarray) -> dict[str, float | int]:
    n = int(len(y))
    n_anom = int(np.sum(y))
    return {
        "n_windows": n,
        "n_anomalies": n_anom,
        "anomaly_rate": float(n_anom / n) if n > 0 else 0.0,
    }


def _tune_threshold(
    val_scores: np.ndarray,
    y_val: np.ndarray,
    n_thresholds: int,
    fallback: float,
) -> tuple[float, list[dict[str, float]]]:
    candidates = np.linspace(0.0, 1.0, n_thresholds)
    sweep = [
        {
            "threshold": float(t),
            "precision": float(precision_score(y_val, (val_scores > t).astype(int), zero_division=0)),
            "recall": float(recall_score(y_val, (val_scores > t).astype(int), zero_division=0)),
            "f1": float(f1_score(y_val, (val_scores > t).astype(int), zero_division=0)),
        }
        for t in candidates
    ]

    if y_val.sum() == 0:
        return float(fallback), sweep

    best = max(sweep, key=lambda r: r["f1"])
    return best["threshold"], sweep


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None
