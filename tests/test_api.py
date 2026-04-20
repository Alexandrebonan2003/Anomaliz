from __future__ import annotations

from fastapi.testclient import TestClient

from anomaliz.agent.graph import build_graph
from anomaliz.agent.llm import MockLLMBackend
from anomaliz.api.deps import get_agent_graph, get_bundle, load_bundle
from anomaliz.api.main import app


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_metrics_and_analyze_no_agent(trained_bundle_dir):
    bundle = load_bundle(trained_bundle_dir)
    app.dependency_overrides[get_bundle] = lambda: bundle
    app.dependency_overrides[get_agent_graph] = lambda: None
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


def test_analyze_with_agent_on_anomaly(trained_bundle_dir):
    bundle = load_bundle(trained_bundle_dir)
    mock_graph = build_graph(MockLLMBackend())
    app.dependency_overrides[get_bundle] = lambda: bundle
    app.dependency_overrides[get_agent_graph] = lambda: mock_graph
    try:
        client = TestClient(app)
        ws = bundle.window_size
        # High-stress metrics to maximize the chance of triggering anomaly=True
        payload = {
            "cpu": [0.95] * ws,
            "memory": [0.95] * ws,
            "latency": [500.0] * ws,
        }
        r = client.post("/analyze", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body["anomaly"], bool)
        if body["anomaly"]:
            assert isinstance(body["analysis"], str) and body["analysis"]
            assert body["severity"] in ("low", "medium", "critical")
            assert isinstance(body["recommendation"], str) and body["recommendation"]
        else:
            assert body["analysis"] is None
    finally:
        app.dependency_overrides.clear()


def test_analyze_rejects_wrong_window_length(trained_bundle_dir):
    bundle = load_bundle(trained_bundle_dir)
    app.dependency_overrides[get_bundle] = lambda: bundle
    app.dependency_overrides[get_agent_graph] = lambda: None
    try:
        client = TestClient(app)
        bad = {"cpu": [0.3, 0.3], "memory": [0.5, 0.5], "latency": [15.0, 15.0]}
        r = client.post("/analyze", json=bad)
        assert r.status_code == 422
    finally:
        app.dependency_overrides.clear()
