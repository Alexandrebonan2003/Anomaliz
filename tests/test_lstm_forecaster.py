from __future__ import annotations

import numpy as np

from anomaliz.models.lstm_forecaster import LSTMForecaster


def _tiny() -> LSTMForecaster:
    return LSTMForecaster(
        units=4,
        dropout=0.0,
        learning_rate=1e-2,
        epochs=2,
        batch_size=8,
        patience=0,
        val_split=0.0,
        random_state=0,
    )


def test_scores_in_unit_interval():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(40, 5, 3)).astype(np.float32)
    d = _tiny().fit(W)
    s = d.score(W)
    assert s.shape == (40,)
    assert s.min() >= 0.0
    assert s.max() <= 1.0


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    W = rng.normal(size=(30, 4, 3)).astype(np.float32)
    d = _tiny().fit(W)
    d.save(tmp_path / "fcst")
    d2 = LSTMForecaster.load(tmp_path / "fcst")
    assert np.allclose(d.score(W), d2.score(W), atol=1e-5)


def test_rejects_window_size_below_two():
    W = np.zeros((10, 1, 3), dtype=np.float32)
    try:
        _tiny().fit(W)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for window_size < 2")
