from __future__ import annotations

import copy
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve

from ..config.settings import Settings
from ..data.dataset import split_series
from ..data.generator import generate_series
from ..detection.scorer import fuse
from ..models.isolation_forest import IFDetector
from ..models.lstm_autoencoder import LSTMAutoencoder
from ..models.lstm_forecaster import LSTMForecaster
from ..preprocessing.normalizer import MinMaxNormalizer
from ..preprocessing.windowing import (
    flatten_windows,
    make_windows,
    select_normal_windows,
)
from ..tracking.loggers import NoOpLogger

FEATURES = ["cpu", "memory", "latency"]
IF_NAME = "isolation_forest"
LSTM_NAME = "lstm_autoencoder"
FORECAST_NAME = "lstm_forecaster"


@dataclass
class TrainingResult:
    metrics: dict[str, Any]
    bundle_dir: Path


@dataclass
class _RunArtifacts:
    if_detector: IFDetector
    lstm_detector: LSTMAutoencoder
    forecaster: LSTMForecaster
    normalizer: MinMaxNormalizer
    metrics: dict[str, Any]
    thresholds: dict[str, Any]
    class_balance: dict[str, Any]
    val_errors: np.ndarray
    fcst_val_errors: np.ndarray
    lstm_error_threshold: float
    fcst_error_threshold: float


def run_training(
    settings: Settings,
    out_dir: Path,
    sweep: bool = False,
    logger=None,
) -> TrainingResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    primary_seed = settings.seed
    extra_seeds = [s for s in settings.evaluation.seeds if s != primary_seed]

    ref = _train_once(settings, primary_seed)

    # Persist reference bundle (primary seed only).
    ref.if_detector.save(out_dir / IF_NAME)
    ref.lstm_detector.save(out_dir / LSTM_NAME)
    ref.forecaster.save(out_dir / FORECAST_NAME)
    ref.normalizer.save(out_dir / "normalizer.json")

    fused_sweep = ref.thresholds["fused"]["sweep"]
    fused_threshold = ref.thresholds["fused"]["threshold"]

    _write_json(
        out_dir / "threshold.json",
        {
            "decision_threshold": float(fused_threshold),
            "tuned_on": "validation",
            "selection_metric": "f1",
            "sweep": fused_sweep,
            "isolation_forest": {
                "threshold": float(ref.thresholds["isolation_forest"]["threshold"]),
                "sweep": ref.thresholds["isolation_forest"]["sweep"],
            },
            "lstm_autoencoder": {
                "threshold": float(ref.thresholds["lstm_autoencoder"]["threshold"]),
                "sweep": ref.thresholds["lstm_autoencoder"]["sweep"],
                "error_threshold": ref.lstm_error_threshold,
                "error_min": float(ref.val_errors.min()),
                "error_max": float(ref.val_errors.max()),
            },
            "lstm_forecaster": {
                "threshold": float(ref.thresholds["lstm_forecaster"]["threshold"]),
                "sweep": ref.thresholds["lstm_forecaster"]["sweep"],
                "error_threshold": ref.fcst_error_threshold,
                "error_min": float(ref.fcst_val_errors.min()),
                "error_max": float(ref.fcst_val_errors.max()),
            },
            "fusion": {
                "weight_if": settings.detection.fusion.weight_if,
                "weight_lstm": settings.detection.fusion.weight_lstm,
            },
        },
    )

    # Multi-seed evaluation (extra seeds do not touch the bundle).
    per_seed = [{"seed": primary_seed, **_strip_for_report(ref.metrics)}]
    for s in extra_seeds:
        run = _train_once(settings, s)
        per_seed.append({"seed": s, **_strip_for_report(run.metrics)})

    seed_evaluation = _aggregate_seeds(per_seed)
    comparison_summary = _comparison_summary(seed_evaluation)

    metrics = dict(ref.metrics)
    metrics["seed_evaluation"] = seed_evaluation
    metrics["comparison_summary"] = comparison_summary

    if sweep:
        metrics["ablations"] = _run_ablations(settings)

    metadata: dict[str, Any] = {
        "config": settings.model_dump(),
        "git_sha": _git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detectors": [IF_NAME, LSTM_NAME, FORECAST_NAME],
        "seeds": [primary_seed, *extra_seeds],
        "reference_seed": primary_seed,
        "sweep_enabled": bool(sweep),
    }
    if sweep:
        metadata["sweep"] = settings.evaluation.ablation.model_dump()

    _write_json(out_dir / "metadata.json", metadata)
    _write_json(out_dir / "metrics.json", metrics)

    _log_run(logger or NoOpLogger(), settings, metrics, ref, out_dir)

    return TrainingResult(metrics=metrics, bundle_dir=out_dir)


def _train_once(settings: Settings, seed: int) -> _RunArtifacts:
    train_df, val_df, test_df, generation_seed = _generate_valid_splits(settings, seed)

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

    if_params = settings.model.isolation_forest.model_dump()
    if_params["random_state"] = seed
    if_detector = IFDetector(**if_params)
    if_detector.fit(flatten_windows(W_train))

    W_train_normal = select_normal_windows(W_train, y_train)
    if W_train_normal.shape[0] == 0:
        raise RuntimeError("No normal training windows available for DL training")
    if W_train_normal.shape[0] != int((y_train == 0).sum()):
        raise RuntimeError("Normal-only filter invariant violated")

    lstm_cfg = settings.model.lstm_autoencoder
    lstm_detector = LSTMAutoencoder(
        units_1=lstm_cfg.units_1,
        units_2=lstm_cfg.units_2,
        dropout=lstm_cfg.dropout,
        recurrent_dropout=lstm_cfg.recurrent_dropout,
        learning_rate=lstm_cfg.learning_rate,
        epochs=lstm_cfg.epochs,
        batch_size=lstm_cfg.batch_size,
        patience=lstm_cfg.patience,
        val_split=lstm_cfg.val_split,
        random_state=seed,
    )
    lstm_detector.fit(W_train_normal)

    fcst_cfg = settings.model.lstm_forecaster
    forecaster = LSTMForecaster(
        units=fcst_cfg.units,
        dropout=fcst_cfg.dropout,
        recurrent_dropout=fcst_cfg.recurrent_dropout,
        learning_rate=fcst_cfg.learning_rate,
        epochs=fcst_cfg.epochs,
        batch_size=fcst_cfg.batch_size,
        patience=fcst_cfg.patience,
        val_split=fcst_cfg.val_split,
        random_state=seed,
    )
    forecaster.fit(W_train_normal)

    val_errors = lstm_detector.reconstruction_error(W_val)
    lstm_detector.set_error_range(float(val_errors.min()), float(val_errors.max()))
    val_normal_errors = val_errors[y_val == 0]
    lstm_error_threshold = (
        float(val_normal_errors.mean() + lstm_cfg.threshold_k * val_normal_errors.std())
        if val_normal_errors.size > 0
        else float(val_errors.mean())
    )

    fcst_val_errors = forecaster.forecast_residuals(W_val)
    forecaster.set_error_range(float(fcst_val_errors.min()), float(fcst_val_errors.max()))
    fcst_val_normal_errors = fcst_val_errors[y_val == 0]
    fcst_error_threshold = (
        float(fcst_val_normal_errors.mean() + fcst_cfg.threshold_k * fcst_val_normal_errors.std())
        if fcst_val_normal_errors.size > 0
        else float(fcst_val_errors.mean())
    )

    if_val_scores = if_detector.score(flatten_windows(W_val))
    lstm_val_scores = lstm_detector.score(W_val)
    fcst_val_scores = forecaster.score(W_val)
    fused_val_scores = fuse(
        if_val_scores,
        lstm_val_scores,
        settings.detection.fusion.weight_if,
        settings.detection.fusion.weight_lstm,
    )

    n_thr = settings.detection.tuning.n_thresholds
    fallback = settings.detection.threshold
    if_t, if_sweep = _tune_threshold(if_val_scores, y_val, n_thr, fallback)
    lstm_t, lstm_sweep = _tune_threshold(lstm_val_scores, y_val, n_thr, fallback)
    fcst_t, fcst_sweep = _tune_threshold(fcst_val_scores, y_val, n_thr, fallback)
    fused_t, fused_sweep = _tune_threshold(fused_val_scores, y_val, n_thr, fallback)

    if_test_scores = if_detector.score(flatten_windows(W_test))
    lstm_test_scores = lstm_detector.score(W_test)
    fcst_test_scores = forecaster.score(W_test)
    fused_test_scores = fuse(
        if_test_scores,
        lstm_test_scores,
        settings.detection.fusion.weight_if,
        settings.detection.fusion.weight_lstm,
    )

    metrics: dict[str, Any] = {
        "fused": _metrics_for(fused_test_scores, y_test, fused_t),
        "isolation_forest": _metrics_for(if_test_scores, y_test, if_t),
        "lstm_autoencoder": _metrics_for(lstm_test_scores, y_test, lstm_t),
        "lstm_forecaster": _metrics_for(fcst_test_scores, y_test, fcst_t),
        "n_test_windows": int(len(y_test)),
        "n_test_anomalies": int(y_test.sum()),
        "class_balance": class_balance,
        "data_generation_seed": int(generation_seed),
    }
    # Phase 1-compatible top-level keys (fused).
    fused_metrics = metrics["fused"]
    metrics["f1"] = fused_metrics["f1"]
    metrics["precision"] = fused_metrics["precision"]
    metrics["recall"] = fused_metrics["recall"]
    metrics["roc_auc"] = fused_metrics["roc_auc"]

    return _RunArtifacts(
        if_detector=if_detector,
        lstm_detector=lstm_detector,
        forecaster=forecaster,
        normalizer=normalizer,
        metrics=metrics,
        thresholds={
            "isolation_forest": {"threshold": if_t, "sweep": if_sweep},
            "lstm_autoencoder": {"threshold": lstm_t, "sweep": lstm_sweep},
            "lstm_forecaster": {"threshold": fcst_t, "sweep": fcst_sweep},
            "fused": {"threshold": fused_t, "sweep": fused_sweep},
        },
        class_balance=class_balance,
        val_errors=val_errors,
        fcst_val_errors=fcst_val_errors,
        lstm_error_threshold=lstm_error_threshold,
        fcst_error_threshold=fcst_error_threshold,
    )


def _strip_for_report(metrics: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for detector in ("fused", "isolation_forest", "lstm_autoencoder", "lstm_forecaster"):
        out[detector] = {k: metrics[detector][k] for k in ("f1", "precision", "recall", "roc_auc")}
    return out


def _aggregate_seeds(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    detectors = ("fused", "isolation_forest", "lstm_autoencoder", "lstm_forecaster")
    metric_keys = ("f1", "precision", "recall", "roc_auc")
    agg: dict[str, Any] = {"per_seed": per_seed, "aggregate": {}}
    for d in detectors:
        agg["aggregate"][d] = {}
        for k in metric_keys:
            vals = [
                r[d][k] for r in per_seed
                if r.get(d) and r[d].get(k) is not None
            ]
            if vals:
                agg["aggregate"][d][k] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n": len(vals),
                }
            else:
                agg["aggregate"][d][k] = {"mean": None, "std": None, "n": 0}
    return agg


def _comparison_summary(seed_evaluation: dict[str, Any]) -> dict[str, Any]:
    agg = seed_evaluation["aggregate"]
    baseline = agg["isolation_forest"]["f1"]["mean"]
    summary = {"baseline_if_f1": baseline, "verdict": {}}
    for d in ("lstm_autoencoder", "lstm_forecaster", "fused"):
        m = agg[d]["f1"]
        if m["mean"] is None or baseline is None:
            summary["verdict"][d] = "insufficient_data"
            continue
        improves = m["mean"] > baseline + m["std"]
        summary["verdict"][d] = "beats_baseline" if improves else "no_improvement"
    return summary


def _run_ablations(settings: Settings) -> dict[str, Any]:
    results: dict[str, Any] = {"units_2": [], "window_size": []}
    base_seed = settings.seed
    for u in settings.evaluation.ablation.units_2:
        cfg = copy.deepcopy(settings)
        cfg.model.lstm_autoencoder.units_2 = u
        run = _train_once(cfg, base_seed)
        results["units_2"].append({"units_2": u, **_strip_for_report(run.metrics)})
    for w in settings.evaluation.ablation.window_size:
        cfg = copy.deepcopy(settings)
        cfg.data.window_size = w
        run = _train_once(cfg, base_seed)
        results["window_size"].append({"window_size": w, **_strip_for_report(run.metrics)})
    return results


_MAX_SPLIT_RETRIES = 20


def _generate_valid_splits(settings: Settings, seed: int):
    """Regenerate with deterministic seed offsets until:
    - val and test each contain at least one anomaly, AND
    - the window-level anomaly rate across the full series is within the
      configured [min_anomaly_rate, max_anomaly_rate] bounds.
    This guarantees both threshold-tuning validity and dataset realism."""
    w = settings.data.window_size
    lo = settings.data.min_anomaly_rate
    hi = settings.data.max_anomaly_rate
    for offset in range(_MAX_SPLIT_RETRIES):
        gen_seed = seed if offset == 0 else seed * 10_000 + offset
        rng = np.random.default_rng(gen_seed)
        df = generate_series(settings.data, rng)

        # Window-level anomaly rate: label of each window = label of its last timestep
        labels = df["label"].to_numpy()
        n_windows = len(labels) - w + 1
        if n_windows < 1:
            continue
        window_anomaly_rate = float(labels[w - 1 :].mean())
        if not (lo <= window_anomaly_rate <= hi):
            continue

        train_df, val_df, test_df = split_series(
            df, settings.data.train_ratio, settings.data.val_ratio
        )
        if int(val_df["label"].sum()) > 0 and int(test_df["label"].sum()) > 0:
            return train_df, val_df, test_df, gen_seed
    raise RuntimeError(
        f"Could not produce valid splits for seed={seed} after {_MAX_SPLIT_RETRIES} "
        f"retries (need anomaly_rate in [{lo}, {hi}] and anomalies in val+test). "
        f"Adjust anomaly_probability or n_points."
    )


_ROC_POINTS = 50


def _metrics_for(scores: np.ndarray, y: np.ndarray, threshold: float) -> dict[str, Any]:
    preds = (scores > threshold).astype(int)
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores)) if y.max() > 0 else None,
        "roc_curve": _roc_curve_data(scores, y),
    }


def _roc_curve_data(scores: np.ndarray, y: np.ndarray) -> dict | None:
    if y.max() == 0:
        return None
    fpr, tpr, _ = roc_curve(y, scores)
    idx = np.linspace(0, len(fpr) - 1, min(_ROC_POINTS, len(fpr))).astype(int)
    return {
        "fpr": [round(float(fpr[i]), 4) for i in idx],
        "tpr": [round(float(tpr[i]), 4) for i in idx],
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


def _log_run(logger, settings: Settings, metrics: dict, ref: _RunArtifacts, out_dir: Path) -> None:
    with logger:
        logger.log_params(_flatten_params(settings))
        logger.log_metrics(_flatten_metrics(metrics))
        for fname in ("metrics.json", "threshold.json", "metadata.json"):
            logger.log_artifact(out_dir / fname)
        if ref.if_detector._model is not None:
            logger.log_model(ref.if_detector._model, IF_NAME)


def _flatten_params(settings: Settings) -> dict[str, Any]:
    d = settings.data
    mi = settings.model.isolation_forest
    ml = settings.model.lstm_autoencoder
    mf = settings.model.lstm_forecaster
    fu = settings.detection.fusion
    return {
        "seed": settings.seed,
        "window_size": d.window_size,
        "n_points": d.n_points,
        "if_n_estimators": mi.n_estimators,
        "if_contamination": mi.contamination,
        "lstm_units_1": ml.units_1,
        "lstm_units_2": ml.units_2,
        "lstm_dropout": ml.dropout,
        "lstm_epochs": ml.epochs,
        "lstm_patience": ml.patience,
        "lstm_threshold_k": ml.threshold_k,
        "fcst_units": mf.units,
        "fcst_dropout": mf.dropout,
        "fcst_epochs": mf.epochs,
        "fcst_patience": mf.patience,
        "fcst_threshold_k": mf.threshold_k,
        "weight_if": fu.weight_if,
        "weight_lstm": fu.weight_lstm,
    }


_DETECTOR_PREFIX = {
    IF_NAME: "if",
    LSTM_NAME: "lstm_ae",
    FORECAST_NAME: "lstm_fcst",
    "fused": "fused",
}


def _flatten_metrics(metrics: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for det, prefix in _DETECTOR_PREFIX.items():
        m = metrics.get(det, {})
        for k in ("f1", "precision", "recall", "roc_auc"):
            v = m.get(k)
            if isinstance(v, float):
                out[f"{prefix}_{k}"] = v
    return out


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
