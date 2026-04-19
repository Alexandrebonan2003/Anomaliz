from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class MinMaxNormalizer:
    def __init__(self) -> None:
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MinMaxNormalizer":
        X = np.asarray(X, dtype=float)
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("Normalizer must be fit before transform")
        X = np.asarray(X, dtype=float)
        span = self.max_ - self.min_
        span = np.where(span == 0, 1.0, span)
        return (X - self.min_) / span

    def save(self, path: Path) -> None:
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("Cannot save an unfit normalizer")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"min": self.min_.tolist(), "max": self.max_.tolist()}))

    @classmethod
    def load(cls, path: Path) -> "MinMaxNormalizer":
        data = json.loads(Path(path).read_text())
        inst = cls()
        inst.min_ = np.array(data["min"], dtype=float)
        inst.max_ = np.array(data["max"], dtype=float)
        return inst
