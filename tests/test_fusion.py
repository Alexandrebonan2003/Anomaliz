from __future__ import annotations

import numpy as np

from anomaliz.detection.scorer import fuse


def test_fuse_scalar_weighted_average():
    out = fuse(0.0, 1.0, weight_if=0.3, weight_lstm=0.7)
    assert np.isclose(float(out), 0.7)


def test_fuse_vectorized_output_in_unit_interval():
    if_scores = np.array([0.0, 0.5, 1.0])
    lstm_scores = np.array([1.0, 0.5, 0.0])
    out = fuse(if_scores, lstm_scores, 0.4, 0.6)
    assert out.shape == if_scores.shape
    assert out.min() >= 0.0
    assert out.max() <= 1.0
