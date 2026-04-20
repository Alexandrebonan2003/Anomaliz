from __future__ import annotations

from typing import TypedDict


class AnomalyState(TypedDict):
    cpu: float
    memory: float
    latency: float
    score: float
    threshold: float
    model_used: str
    analysis: str
    severity: str
    recommendation: str
