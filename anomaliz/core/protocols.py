from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Detector(Protocol):
    def fit(self, X: np.ndarray) -> "Detector": ...

    def score(self, X: np.ndarray) -> np.ndarray: ...

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> "Detector": ...
