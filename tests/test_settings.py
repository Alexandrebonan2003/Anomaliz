from __future__ import annotations

from anomaliz.config.settings import load_settings


def test_defaults_load():
    s = load_settings()
    assert s.data.window_size == 10
    assert s.model.isolation_forest.n_estimators == 100
    assert s.detection.threshold == 0.5


def test_env_override_propagates(monkeypatch):
    monkeypatch.setenv("ANOMALIZ__DATA__WINDOW_SIZE", "15")
    monkeypatch.setenv("ANOMALIZ__MODEL__ISOLATION_FOREST__N_ESTIMATORS", "42")
    s = load_settings()
    assert s.data.window_size == 15
    assert s.model.isolation_forest.n_estimators == 42
