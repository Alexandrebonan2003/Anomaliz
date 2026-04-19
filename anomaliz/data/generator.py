from __future__ import annotations

import numpy as np
import pandas as pd

from ..config.settings import DataConfig

ANOMALY_TYPES = ("cpu_spike", "memory_leak", "latency_drift", "system_crash")


def generate_series(cfg: DataConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = cfg.n_points
    cpu = rng.normal(0.30, 0.05, size=n).clip(0.0, 1.0)
    memory = rng.normal(0.50, 0.02, size=n).clip(0.0, 1.0)
    latency = rng.normal(15.0, 2.0, size=n).clip(0.0, None)
    label = np.zeros(n, dtype=np.int64)

    warmup = min(50, n // 10)
    i = warmup
    tail = max(warmup, 1)
    while i < n - tail:
        if rng.random() < cfg.anomaly_probability:
            kind = ANOMALY_TYPES[int(rng.integers(0, len(ANOMALY_TYPES)))]
            i = _inject(kind, i, cpu, memory, latency, label, rng, n)
        else:
            i += 1

    return pd.DataFrame(
        {
            "timestamp": np.arange(n),
            "cpu": cpu,
            "memory": memory,
            "latency": latency,
            "label": label,
        }
    )


def _inject(
    kind: str,
    start: int,
    cpu: np.ndarray,
    memory: np.ndarray,
    latency: np.ndarray,
    label: np.ndarray,
    rng: np.random.Generator,
    n: int,
) -> int:
    if kind == "cpu_spike":
        dur = int(rng.integers(2, 6))
        end = min(start + dur, n)
        cpu[start:end] = rng.uniform(0.90, 0.95, size=end - start)
    elif kind == "memory_leak":
        dur = int(rng.integers(20, 51))
        end = min(start + dur, n)
        peak = float(rng.uniform(0.85, 0.95))
        memory[start:end] = np.linspace(memory[start], peak, end - start)
    elif kind == "latency_drift":
        dur = int(rng.integers(10, 31))
        end = min(start + dur, n)
        peak = float(rng.uniform(80.0, 150.0))
        latency[start:end] = np.linspace(latency[start], peak, end - start)
    elif kind == "system_crash":
        dur = int(rng.integers(3, 9))
        end = min(start + dur, n)
        cpu[start:end] = rng.normal(0.0, 0.01, size=end - start).clip(0.0, 1.0)
        memory[start:end] = rng.normal(0.0, 0.01, size=end - start).clip(0.0, 1.0)
        latency[start:end] = rng.normal(0.0, 0.1, size=end - start).clip(0.0, None)
    else:
        return start + 1

    label[start:end] = 1
    return end
