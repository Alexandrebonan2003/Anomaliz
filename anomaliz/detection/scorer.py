from __future__ import annotations

import numpy as np


def decide(score: float, threshold: float) -> bool:
    return bool(score > threshold)


def fuse(
    if_score: np.ndarray | float,
    lstm_score: np.ndarray | float,
    weight_if: float,
    weight_lstm: float,
) -> np.ndarray:
    if_arr = np.asarray(if_score, dtype=float)
    lstm_arr = np.asarray(lstm_score, dtype=float)
    return np.clip(weight_if * if_arr + weight_lstm * lstm_arr, 0.0, 1.0)
