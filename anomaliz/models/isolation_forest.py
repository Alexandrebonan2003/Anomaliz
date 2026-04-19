from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest


class IFDetector:
    def __init__(
        self, n_estimators: int = 100, contamination: float = 0.05, random_state: int = 42
    ) -> None:
        self.params = {
            "n_estimators": n_estimators,
            "contamination": contamination,
            "random_state": random_state,
        }
        self._model: IsolationForest | None = None
        self._score_min: float | None = None
        self._score_max: float | None = None

    def fit(self, X: np.ndarray) -> "IFDetector":
        self._model = IsolationForest(**self.params).fit(X)
        raw = -self._model.decision_function(X)
        self._score_min = float(raw.min())
        self._score_max = float(raw.max())
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or self._score_min is None or self._score_max is None:
            raise RuntimeError("Detector must be fit before scoring")
        raw = -self._model.decision_function(X)
        span = self._score_max - self._score_min
        if span <= 0:
            span = 1.0
        return np.clip((raw - self._score_min) / span, 0.0, 1.0)

    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("Cannot save an unfit detector")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(self._model, f)
        (path / "score_stats.json").write_text(
            json.dumps(
                {"min": self._score_min, "max": self._score_max, "params": self.params}
            )
        )

    @classmethod
    def load(cls, path: Path) -> "IFDetector":
        path = Path(path)
        stats = json.loads((path / "score_stats.json").read_text())
        inst = cls(**stats["params"])
        with open(path / "model.pkl", "rb") as f:
            inst._model = pickle.load(f)
        inst._score_min = stats["min"]
        inst._score_max = stats["max"]
        return inst
