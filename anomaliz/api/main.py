from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, HTTPException

from ..agent.graph import build_graph, invoke_agent
from ..agent.llm import build_backend
from ..config.settings import load_settings
from ..detection.scorer import decide, fuse
from .deps import Bundle, get_agent_graph, get_bundle, load_bundle
from .schemas import AnalyzeRequest, AnalyzeResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle_dir = os.environ.get("ANOMALIZ_ARTIFACT_DIR")
    app.state.bundle = load_bundle(Path(bundle_dir)) if bundle_dir else None

    settings = load_settings()
    backend = build_backend(settings.agent)
    app.state.agent_graph = build_graph(backend) if backend is not None else None

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
    req: AnalyzeRequest,
    bundle: Bundle = Depends(get_bundle),
    agent_graph=Depends(get_agent_graph),
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

    is_anomaly = decide(fused_score, bundle.threshold)
    analysis = severity = recommendation = None

    if is_anomaly and agent_graph is not None:
        cpu_mean = float(np.mean(req.cpu))
        memory_mean = float(np.mean(req.memory))
        latency_mean = float(np.mean(req.latency))
        try:
            analysis, severity, recommendation = invoke_agent(
                agent_graph,
                cpu=cpu_mean * 100,
                memory=memory_mean * 100,
                latency=latency_mean,
                score=fused_score,
                threshold=bundle.threshold,
                model_used=model_used,
            )
        except Exception:
            logger.warning(
                "Agent invocation failed (score=%.3f, model=%s); degrading gracefully",
                fused_score,
                model_used,
                exc_info=True,
            )

    return AnalyzeResponse(
        anomaly=is_anomaly,
        score=fused_score,
        threshold=bundle.threshold,
        model_used=model_used,
        analysis=analysis,
        severity=severity,
        recommendation=recommendation,
    )
