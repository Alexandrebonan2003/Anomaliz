from __future__ import annotations

from pathlib import Path

import pytest

from anomaliz.config.settings import load_settings
from anomaliz.training.pipeline import run_training


@pytest.fixture(scope="session")
def small_settings():
    s = load_settings()
    s.data.n_points = 400
    s.data.anomaly_probability = 0.1
    s.seed = 7
    s.model.lstm_autoencoder.units_1 = 8
    s.model.lstm_autoencoder.units_2 = 4
    s.model.lstm_autoencoder.epochs = 2
    s.model.lstm_autoencoder.batch_size = 16
    s.model.lstm_autoencoder.patience = 0
    return s


@pytest.fixture(scope="session")
def trained_bundle_dir(tmp_path_factory, small_settings) -> Path:
    out = tmp_path_factory.mktemp("bundle")
    result = run_training(small_settings, out)
    return result.bundle_dir
