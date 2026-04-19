from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    cpu: list[float] = Field(..., min_length=1)
    memory: list[float] = Field(..., min_length=1)
    latency: list[float] = Field(..., min_length=1)


class AnalyzeResponse(BaseModel):
    anomaly: bool
    score: float
    threshold: float
    model_used: str
    analysis: str | None = None
    severity: str | None = None
    recommendation: str | None = None
