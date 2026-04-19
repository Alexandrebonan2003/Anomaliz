from __future__ import annotations

import numpy as np


def make_windows(
    X: np.ndarray, y: np.ndarray, window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same length")
    if X.shape[0] < window_size:
        raise ValueError(f"Need at least {window_size} points, got {X.shape[0]}")

    num = X.shape[0] - window_size + 1
    windows = np.stack([X[i : i + window_size] for i in range(num)])
    # Label each window by its final point: the window represents a short history
    # whose "current" timestamp is the last entry. This also makes it trivial to
    # select strictly-normal windows for future forecasting/autoencoder training.
    labels = y[window_size - 1 : window_size - 1 + num].astype(np.int64)
    return windows, labels


def flatten_windows(W: np.ndarray) -> np.ndarray:
    return W.reshape(W.shape[0], -1)


def select_normal_windows(W: np.ndarray, y: np.ndarray) -> np.ndarray:
    return W[np.asarray(y) == 0]
