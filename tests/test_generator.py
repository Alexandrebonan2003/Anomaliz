from __future__ import annotations

import numpy as np

from anomaliz.config.settings import load_settings
from anomaliz.data.generator import generate_series


def test_seed_determinism():
    cfg = load_settings().data
    a = generate_series(cfg, np.random.default_rng(0))
    b = generate_series(cfg, np.random.default_rng(0))
    assert (a.values == b.values).all()


def test_schema_and_domain():
    cfg = load_settings().data
    df = generate_series(cfg, np.random.default_rng(1))
    assert list(df.columns) == ["timestamp", "cpu", "memory", "latency", "label"]
    assert len(df) == cfg.n_points
    assert df["label"].isin([0, 1]).all()
    assert df["cpu"].between(0, 1).all()
    assert df["memory"].between(0, 1).all()
    assert (df["latency"] >= 0).all()


def test_anomalies_are_injected():
    cfg = load_settings().data
    cfg.anomaly_probability = 0.2
    df = generate_series(cfg, np.random.default_rng(2))
    assert df["label"].sum() > 0
