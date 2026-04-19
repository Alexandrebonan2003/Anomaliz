from __future__ import annotations

import numpy as np

from anomaliz.preprocessing.normalizer import MinMaxNormalizer
from anomaliz.preprocessing.windowing import (
    flatten_windows,
    make_windows,
    select_normal_windows,
)


def test_normalizer_fit_transform_roundtrip(tmp_path):
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    norm = MinMaxNormalizer().fit(X)
    Y = norm.transform(X)
    assert Y.min() == 0.0
    assert Y.max() == 1.0

    path = tmp_path / "norm.json"
    norm.save(path)
    reloaded = MinMaxNormalizer.load(path)
    assert np.allclose(reloaded.transform(X), Y)


def test_normalizer_constant_feature_does_not_divide_by_zero():
    X = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]])
    Y = MinMaxNormalizer().fit(X).transform(X)
    assert np.all(np.isfinite(Y))


def test_windowing_labels_use_last_point():
    X = np.arange(15, dtype=float).reshape(5, 3)
    y = np.array([0, 0, 1, 0, 0])
    W, yl = make_windows(X, y, window_size=3)
    assert W.shape == (3, 3, 3)
    # windows end at indices 2, 3, 4 → labels mirror y at those positions
    assert list(yl) == [1, 0, 0]
    assert flatten_windows(W).shape == (3, 9)


def test_windowing_all_normal_yields_zero_labels():
    X = np.zeros((5, 3))
    y = np.zeros(5, dtype=int)
    _, yl = make_windows(X, y, window_size=3)
    assert list(yl) == [0, 0, 0]


def test_select_normal_windows_keeps_only_label_zero():
    X = np.arange(18, dtype=float).reshape(6, 3)
    y = np.array([0, 1, 0, 0, 1, 0])
    W, yl = make_windows(X, y, window_size=2)
    normal = select_normal_windows(W, yl)
    assert normal.shape[0] == int((yl == 0).sum())
    assert normal.shape[1:] == W.shape[1:]
