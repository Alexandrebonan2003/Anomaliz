from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, HTTPException

from ..detection.scorer import decide, fuse
from .deps import Bundle, get_bundle, load_bundle
from .schemas import AnalyzeRequest, AnalyzeResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle_dir = os.environ.get("ANOMALIZ_ARTIFACT_DIR")
    app.state.bundle = load_bundle(Path(bundle_dir)) if bundle_dir else None
    yield


app = FastAPI(title="Anomaliz", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics(bundle: Bundle = Depends(get_bundle)) -> dict:
    return bundle.metrics


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    req: AnalyzeRequest, bundle: Bundle = Depends(get_bundle)
) -> AnalyzeResponse:
    lengths = {len(req.cpu), len(req.memory), len(req.latency)}
    if len(lengths) != 1:
        raise HTTPException(
            status_code=422, detail="cpu, memory, latency must have equal length"
        )
    (length,) = lengths
    if length != bundle.window_size:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {bundle.window_size} values per feature, got {length}",
        )

    raw = np.column_stack([req.cpu, req.memory, req.latency]).astype(float)
    X_norm = bundle.normalizer.transform(raw)
    X_flat = X_norm.reshape(1, -1)
    X_seq = X_norm.reshape(1, bundle.window_size, -1)

    if_score = float(bundle.detector.score(X_flat)[0])
    lstm_score = float(bundle.lstm_detector.score(X_seq)[0])
    fused_score = float(
        fuse(if_score, lstm_score, bundle.weight_if, bundle.weight_lstm)
    )

    if_contribution = bundle.weight_if * if_score
    lstm_contribution = bundle.weight_lstm * lstm_score
    model_used = (
        "isolation_forest" if if_contribution >= lstm_contribution else "lstm_autoencoder"
    )

    return AnalyzeResponse(
        anomaly=decide(fused_score, bundle.threshold),
        score=fused_score,
        threshold=bundle.threshold,
        model_used=model_used,
        analysis=None,
        severity=None,
        recommendation=None,
    )
