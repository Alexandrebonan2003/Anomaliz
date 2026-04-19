from __future__ import annotations

import numpy as np

from anomaliz.models.lstm_autoencoder import LSTMAutoencoder


def _tiny_detector() -> LSTMAutoencoder:
    return LSTMAutoencoder(
        units_1=4,
        units_2=2,
        dropout=0.0,
        learning_rate=1e-2,
        epochs=2,
        batch_size=8,
        patience=0,
        random_state=0,
    )


def test_scores_in_unit_interval():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5, 3)).astype(np.float32)
    d = _tiny_detector().fit(X)
    s = d.score(X)
    assert s.shape == (40,)
    assert s.min() >= 0.0
    assert s.max() <= 1.0


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 4, 3)).astype(np.float32)
    d = _tiny_detector().fit(X)
    d.save(tmp_path / "lstm")
    d2 = LSTMAutoencoder.load(tmp_path / "lstm")
    assert np.allclose(d.score(X), d2.score(X), atol=1e-5)
