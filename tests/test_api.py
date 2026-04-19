from __future__ import annotations

from fastapi.testclient import TestClient

from anomaliz.api.deps import get_bundle, load_bundle
from anomaliz.api.main import app


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_metrics_and_analyze(trained_bundle_dir):
    bundle = load_bundle(trained_bundle_dir)
    app.dependency_overrides[get_bundle] = lambda: bundle
    try:
        client = TestClient(app)

        r = client.get("/metrics")
        assert r.status_code == 200
        assert "f1" in r.json()

        ws = bundle.window_size
        payload = {
            "cpu": [0.30] * ws,
            "memory": [0.50] * ws,
            "latency": [15.0] * ws,
        }
        r = client.post("/analyze", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["model_used"] in ("isolation_forest", "lstm_autoencoder")
        assert 0.0 <= body["score"] <= 1.0
        assert body["threshold"] == bundle.threshold
        assert body["analysis"] is None
        assert body["severity"] is None
        assert body["recommendation"] is None
        assert isinstance(body["anomaly"], bool)
    finally:
        app.dependency_overrides.clear()


def test_analyze_rejects_wrong_window_length(trained_bundle_dir):
    bundle = load_bundle(trained_bundle_dir)
    app.dependency_overrides[get_bundle] = lambda: bundle
    try:
        client = TestClient(app)
        bad = {"cpu": [0.3, 0.3], "memory": [0.5, 0.5], "latency": [15.0, 15.0]}
        r = client.post("/analyze", json=bad)
        assert r.status_code == 422
    finally:
        app.dependency_overrides.clear()
