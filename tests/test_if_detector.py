from __future__ import annotations

import numpy as np

from anomaliz.models.isolation_forest import IFDetector


def test_scores_in_unit_interval_and_outliers_rank_higher():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6))
    X[:10] += 5.0
    d = IFDetector(random_state=0).fit(X)
    s = d.score(X)
    assert s.min() >= 0.0
    assert s.max() <= 1.0
    assert s[:10].mean() > s[10:].mean()


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 4))
    d = IFDetector(random_state=1).fit(X)
    d.save(tmp_path / "if")
    d2 = IFDetector.load(tmp_path / "if")
    assert np.allclose(d.score(X), d2.score(X))
