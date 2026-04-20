from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

DEFAULTS_PATH = Path(__file__).parent / "defaults.yaml"
ENV_PREFIX = "ANOMALIZ__"


class DataConfig(BaseModel):
    n_points: int = 8000
    window_size: int = 10
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    anomaly_probability: float = 0.005
    min_anomaly_rate: float = 0.03
    max_anomaly_rate: float = 0.25


class IFConfig(BaseModel):
    n_estimators: int = 100
    contamination: float = 0.05
    random_state: int = 42


class LSTMConfig(BaseModel):
    units_1: int = 64
    units_2: int = 32
    dropout: float = 0.2
    recurrent_dropout: float = 0.0
    learning_rate: float = 1e-3
    epochs: int = 20
    batch_size: int = 32
    patience: int = 5
    val_split: float = 0.1
    threshold_k: float = 2.5


class LSTMForecasterConfig(BaseModel):
    units: int = 32
    dropout: float = 0.2
    recurrent_dropout: float = 0.0
    learning_rate: float = 1e-3
    epochs: int = 20
    batch_size: int = 32
    patience: int = 5
    val_split: float = 0.1
    threshold_k: float = 2.5


class ModelConfig(BaseModel):
    isolation_forest: IFConfig = Field(default_factory=IFConfig)
    lstm_autoencoder: LSTMConfig = Field(default_factory=LSTMConfig)
    lstm_forecaster: LSTMForecasterConfig = Field(default_factory=LSTMForecasterConfig)


class AblationConfig(BaseModel):
    units_2: list[int] = Field(default_factory=lambda: [16, 32, 64])
    window_size: list[int] = Field(default_factory=lambda: [10, 20])


class EvaluationConfig(BaseModel):
    seeds: list[int] = Field(default_factory=lambda: [42, 43, 44, 45, 46])
    ablation: AblationConfig = Field(default_factory=AblationConfig)


class ThresholdTuningConfig(BaseModel):
    n_thresholds: int = 101


class FusionConfig(BaseModel):
    weight_if: float = 0.3
    weight_lstm: float = 0.7


class DetectionConfig(BaseModel):
    threshold: float = 0.5
    tuning: ThresholdTuningConfig = Field(default_factory=ThresholdTuningConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class AgentConfig(BaseModel):
    backend: str = "disabled"  # "openai" | "ollama" | "disabled"
    openai_model: str = "gpt-4o-mini"
    ollama_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434"


class TrackingConfig(BaseModel):
    experiment_name: str = "anomaliz"
    tracking_uri: str | None = None


class Settings(BaseModel):
    seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)


def load_settings(config_path: Path | None = None) -> Settings:
    data = _load_yaml(DEFAULTS_PATH)
    if config_path is not None:
        data = _deep_merge(data, _load_yaml(Path(config_path)))
    data = _deep_merge(data, _env_overrides())
    return Settings.model_validate(data)


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _env_overrides() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw_key, raw_val in os.environ.items():
        if not raw_key.startswith(ENV_PREFIX):
            continue
        path = raw_key[len(ENV_PREFIX):].lower().split("__")
        cursor = result
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise ValueError(f"Env var {raw_key} conflicts with an earlier scalar override")
        cursor[path[-1]] = _coerce(raw_val)
    return result


def _coerce(s: str) -> Any:
    lower = s.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    for parser in (int, float):
        try:
            return parser(s)
        except ValueError:
            continue
    return s
