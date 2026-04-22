"""NAB (Numenta Anomaly Benchmark) data loading and evaluation.

Downloads a curated subset of NAB, adapts the univariate series to the
anomaliz three-feature format, then trains and evaluates the full detector
stack using the same architecture as an existing bundle.

Usage::

    python -m anomaliz.data.nab --bundle artifacts/dev/ --cache-dir .nab_cache
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from ..detection.scorer import fuse
from ..models.isolation_forest import IFDetector
from ..models.lstm_autoencoder import LSTMAutoencoder
from ..models.lstm_forecaster import LSTMForecaster
from ..preprocessing.normalizer import MinMaxNormalizer
from ..preprocessing.windowing import flatten_windows, make_windows, select_normal_windows

_NAB_BASE = "https://raw.githubusercontent.com/numenta/NAB/master"
_LABELS_URL = f"{_NAB_BASE}/labels/combined_windows.json"

# Curated CPU-centric series with clear, labelled anomaly windows.
SERIES: dict[str, str] = {
    "ec2_cpu_ac20cd": "realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv",
    "cpu_asg": "realKnownCause/cpu_utilization_asg_misconfiguration.csv",
}

_FEATURES = ["cpu", "memory", "latency"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _fetch(url: str, dest: Path) -> None:
    """Download url → dest unless already cached."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"  Downloading {url} ...")
        urllib.request.urlretrieve(url, dest)


def load_series(
    key: str,
    cache_dir: Path = Path(".nab_cache"),
    seed: int = 42,
) -> pd.DataFrame:
    """Load a NAB series and return a DataFrame[cpu, memory, latency, label].

    The raw univariate value is normalised to [0, 1] and used as ``cpu``.
    ``memory`` is a smoothed, correlated signal; ``latency`` is an affine
    mapping into a ~[5, 45] ms range. Labels come from the NAB anomaly windows.
    """
    if key not in SERIES:
        raise ValueError(f"Unknown series {key!r}. Available: {list(SERIES)}")

    nab_path = SERIES[key]
    _fetch(f"{_NAB_BASE}/data/{nab_path}", cache_dir / f"{key}.csv")
    _fetch(_LABELS_URL, cache_dir / "labels.json")

    df_raw = pd.read_csv(cache_dir / f"{key}.csv")
    labels_data: dict[str, list] = json.loads((cache_dir / "labels.json").read_text())

    timestamps = pd.to_datetime(df_raw["timestamp"])
    label = np.zeros(len(df_raw), dtype=np.int64)
    for win_start_str, win_end_str in labels_data.get(nab_path, []):
        mask = (timestamps >= pd.Timestamp(win_start_str)) & (
            timestamps <= pd.Timestamp(win_end_str)
        )
        label[mask] = 1

    rng = np.random.default_rng(seed)
    v = df_raw["value"].to_numpy(dtype=float)

    v_min, v_max = float(v.min()), float(v.max())
    cpu = np.clip((v - v_min) / (v_max - v_min if v_max > v_min else 1.0), 0.0, 1.0)

    smoothed = np.convolve(cpu, np.ones(5) / 5, mode="same")
    memory = np.clip(smoothed * 0.7 + 0.15 + rng.normal(0, 0.02, size=len(cpu)), 0.0, 1.0)
    latency = np.clip(5.0 + cpu * 40.0 + rng.normal(0, 1.5, size=len(cpu)), 0.0, None)

    return pd.DataFrame({"cpu": cpu, "memory": memory, "latency": latency, "label": label})


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _tune(scores: np.ndarray, labels: np.ndarray, fallback: float = 0.5) -> float:
    """Return the threshold that maximises F1 on the given validation set."""
    if labels.sum() == 0:
        return fallback
    best_t, best_f1 = fallback, 0.0
    for t in np.linspace(0.0, 1.0, 101):
        f = float(f1_score(labels, (scores > t).astype(int), zero_division=0))
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t


def _compute_metrics(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> dict[str, Any]:
    preds = (scores > threshold).astype(int)
    return {
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)) if labels.max() > 0 else None,
    }


def evaluate(
    key: str,
    bundle_dir: Path,
    cache_dir: Path = Path(".nab_cache"),
    seed: int = 42,
) -> dict[str, Any]:
    """Train and evaluate on a NAB series using the bundle's architecture config.

    Fits a fresh normaliser and models on the NAB training split so that
    results reflect architecture generalisation to real data independent of
    the scale of the synthetic training distribution.
    """
    meta = json.loads((bundle_dir / "metadata.json").read_text())
    cfg = meta["config"]
    window_size = int(cfg["data"]["window_size"])
    train_ratio = float(cfg["data"]["train_ratio"])
    val_ratio = float(cfg["data"]["val_ratio"])
    w_if = float(cfg["detection"]["fusion"]["weight_if"])
    w_lstm = float(cfg["detection"]["fusion"]["weight_lstm"])

    df = load_series(key, cache_dir=cache_dir, seed=seed)
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val :].reset_index(drop=True)

    norm = MinMaxNormalizer().fit(train_df[_FEATURES].to_numpy())
    X_train = norm.transform(train_df[_FEATURES].to_numpy())
    X_val = norm.transform(val_df[_FEATURES].to_numpy())
    X_test = norm.transform(test_df[_FEATURES].to_numpy())

    W_train, y_train = make_windows(X_train, train_df["label"].to_numpy(), window_size)
    W_val, y_val = make_windows(X_val, val_df["label"].to_numpy(), window_size)
    W_test, y_test = make_windows(X_test, test_df["label"].to_numpy(), window_size)

    W_normal = select_normal_windows(W_train, y_train)
    if W_normal.shape[0] == 0:
        raise RuntimeError(f"No normal training windows in series '{key}'")

    # Isolation Forest
    c_if = cfg["model"]["isolation_forest"]
    if_det = IFDetector(
        n_estimators=c_if["n_estimators"],
        contamination=c_if["contamination"],
        random_state=seed,
    )
    if_det.fit(flatten_windows(W_train))

    # LSTM Autoencoder
    c_ae = cfg["model"]["lstm_autoencoder"]
    ae = LSTMAutoencoder(
        units_1=c_ae["units_1"], units_2=c_ae["units_2"],
        dropout=c_ae["dropout"], recurrent_dropout=c_ae["recurrent_dropout"],
        learning_rate=c_ae["learning_rate"], epochs=c_ae["epochs"],
        batch_size=c_ae["batch_size"], patience=c_ae["patience"],
        val_split=c_ae["val_split"], random_state=seed,
    )
    ae.fit(W_normal)
    ae_val_err = ae.reconstruction_error(W_val)
    ae.set_error_range(float(ae_val_err.min()), float(ae_val_err.max()))

    # LSTM Forecaster
    c_fc = cfg["model"]["lstm_forecaster"]
    fc = LSTMForecaster(
        units=c_fc["units"], dropout=c_fc["dropout"],
        recurrent_dropout=c_fc["recurrent_dropout"], learning_rate=c_fc["learning_rate"],
        epochs=c_fc["epochs"], batch_size=c_fc["batch_size"],
        patience=c_fc["patience"], val_split=c_fc["val_split"], random_state=seed,
    )
    fc.fit(W_normal)
    fc_val_err = fc.forecast_residuals(W_val)
    fc.set_error_range(float(fc_val_err.min()), float(fc_val_err.max()))

    # Threshold tuning on validation scores
    if_val = if_det.score(flatten_windows(W_val))
    ae_val = ae.score(W_val)
    fc_val = fc.score(W_val)
    fused_val = fuse(if_val, ae_val, w_if, w_lstm)

    # Test scores
    if_test = if_det.score(flatten_windows(W_test))
    ae_test = ae.score(W_test)
    fc_test = fc.score(W_test)
    fused_test = fuse(if_test, ae_test, w_if, w_lstm)

    return {
        "series": key,
        "n_test_windows": int(len(y_test)),
        "n_test_anomalies": int(y_test.sum()),
        "anomaly_rate": round(float(y_test.mean()), 4),
        "isolation_forest": _compute_metrics(if_test, y_test, _tune(if_val, y_val)),
        "lstm_autoencoder": _compute_metrics(ae_test, y_test, _tune(ae_val, y_val)),
        "lstm_forecaster": _compute_metrics(fc_test, y_test, _tune(fc_val, y_val)),
        "fused": _compute_metrics(fused_test, y_test, _tune(fused_val, y_val)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Anomaliz models on NAB real-world time-series data."
    )
    parser.add_argument("--bundle", type=Path, required=True, help="Trained bundle directory.")
    parser.add_argument(
        "--series",
        nargs="+",
        default=list(SERIES.keys()),
        choices=list(SERIES.keys()),
        help="NAB series to evaluate (default: all).",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".nab_cache"), dest="cache_dir",
        help="Local directory for cached NAB files (default: .nab_cache).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    # Load synthetic benchmark for comparison
    synth_fused: dict[str, Any] = {}
    try:
        synth = json.loads((args.bundle / "metrics.json").read_text())
        synth_fused = synth.get("fused") or {k: synth.get(k) for k in ("f1", "roc_auc")}
    except FileNotFoundError:
        pass

    if synth_fused.get("f1") is not None:
        f1_s = synth_fused["f1"]
        auc_s = synth_fused.get("roc_auc") or 0.0
        print(f"\nSynthetic benchmark  (fused): F1={f1_s:.3f}  AUC={auc_s:.3f}")

    header = f"\n  {'Detector':<22} {'F1':>6} {'Prec':>6} {'Recall':>6} {'AUC':>6}"
    sep = "  " + "-" * 54

    for key in args.series:
        print(f"\n{'=' * 58}\nSeries: {key}")
        print("  Training and evaluating (this may take a minute) ...")
        try:
            result = evaluate(key, args.bundle, cache_dir=args.cache_dir, seed=args.seed)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

        n_anom = result["n_test_anomalies"]
        n_win = result["n_test_windows"]
        rate = result["anomaly_rate"]
        print(f"  Test windows: {n_win}  anomalies: {n_anom} ({rate:.1%})")
        print(header)
        print(sep)
        for det in ("isolation_forest", "lstm_autoencoder", "lstm_forecaster", "fused"):
            m = result[det]
            auc_str = f"{m['roc_auc']:.3f}" if m["roc_auc"] is not None else "  N/A"
            name = det.replace("_", " ").title()
            print(
                f"  {name:<22} {m['f1']:>6.3f} {m['precision']:>6.3f}"
                f" {m['recall']:>6.3f} {auc_str:>6}"
            )


if __name__ == "__main__":
    main()
