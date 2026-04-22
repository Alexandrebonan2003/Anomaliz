from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request

from ..models.isolation_forest import IFDetector
from ..models.lstm_autoencoder import LSTMAutoencoder
from ..preprocessing.normalizer import MinMaxNormalizer

# CompiledGraph is imported at call-time to avoid mandatory langgraph import at module load


@dataclass
class Bundle:
    detector: IFDetector
    lstm_detector: LSTMAutoencoder
    normalizer: MinMaxNormalizer
    threshold: float
    window_size: int
    weight_if: float
    weight_lstm: float
    metadata: dict[str, Any]
    metrics: dict[str, Any]


def load_bundle(bundle_dir: Path) -> Bundle:
    bundle_dir = Path(bundle_dir)
    detector = IFDetector.load(bundle_dir / "isolation_forest")
    lstm_detector = LSTMAutoencoder.load(bundle_dir / "lstm_autoencoder")
    normalizer = MinMaxNormalizer.load(bundle_dir / "normalizer.json")
    threshold_payload = json.loads((bundle_dir / "threshold.json").read_text())
    metadata = json.loads((bundle_dir / "metadata.json").read_text())
    metrics = json.loads((bundle_dir / "metrics.json").read_text())
    window_size = int(metadata["config"]["data"]["window_size"])
    fusion = threshold_payload.get("fusion") or metadata["config"]["detection"]["fusion"]
    return Bundle(
        detector=detector,
        lstm_detector=lstm_detector,
        normalizer=normalizer,
        threshold=float(threshold_payload["decision_threshold"]),
        window_size=window_size,
        weight_if=float(fusion["weight_if"]),
        weight_lstm=float(fusion["weight_lstm"]),
        metadata=metadata,
        metrics=metrics,
    )


def get_bundle(request: Request) -> Bundle:
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Bundle not loaded")
    return bundle


def get_agent_graph(request: Request):
    """Return the compiled LangGraph agent, or None when agent is disabled."""
    return getattr(request.app.state, "agent_graph", None)
