"""Tests for NAB data loading — offline, no network required."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from anomaliz.data.nab import SERIES, load_series


@pytest.fixture()
def nab_cache(tmp_path: Path) -> Path:
    """Fake NAB cache: one series CSV + labels.json, no network needed."""
    cache = tmp_path / "nab"
    cache.mkdir()

    key = "ec2_cpu_ac20cd"
    nab_path = SERIES[key]
    n = 80

    timestamps = [
        str(pd.Timestamp("2014-01-01") + pd.Timedelta(minutes=5 * i)) for i in range(n)
    ]
    values = np.linspace(10.0, 90.0, n)
    pd.DataFrame({"timestamp": timestamps, "value": values}).to_csv(
        cache / f"{key}.csv", index=False
    )

    # Anomaly window: rows 65–79
    (cache / "labels.json").write_text(
        json.dumps({nab_path: [[timestamps[65], timestamps[79]]]})
    )
    return cache


def test_columns(nab_cache: Path) -> None:
    df = load_series("ec2_cpu_ac20cd", cache_dir=nab_cache)
    assert {"cpu", "memory", "latency", "label"}.issubset(df.columns)


def test_value_ranges(nab_cache: Path) -> None:
    df = load_series("ec2_cpu_ac20cd", cache_dir=nab_cache)
    assert df["cpu"].between(0.0, 1.0).all()
    assert df["memory"].between(0.0, 1.0).all()
    assert (df["latency"] >= 0.0).all()


def test_labels_assigned(nab_cache: Path) -> None:
    df = load_series("ec2_cpu_ac20cd", cache_dir=nab_cache)
    assert df["label"].sum() > 0, "expected at least one anomaly label"
    assert df["label"].sum() < len(df), "not all rows should be anomalies"


def test_length_preserved(nab_cache: Path) -> None:
    df = load_series("ec2_cpu_ac20cd", cache_dir=nab_cache)
    assert len(df) == 80


def test_unknown_key_raises(nab_cache: Path) -> None:
    with pytest.raises(ValueError, match="Unknown series"):
        load_series("nonexistent_key", cache_dir=nab_cache)


def test_deterministic_with_same_seed(nab_cache: Path) -> None:
    df1 = load_series("ec2_cpu_ac20cd", cache_dir=nab_cache, seed=7)
    df2 = load_series("ec2_cpu_ac20cd", cache_dir=nab_cache, seed=7)
    pd.testing.assert_frame_equal(df1, df2)
