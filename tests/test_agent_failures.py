from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from anomaliz.agent.graph import build_graph
from anomaliz.agent.llm import LLMBackend, MockLLMBackend
from anomaliz.api.deps import get_agent_graph, get_bundle, load_bundle
from anomaliz.api.main import app


class _AlwaysFailingBackend:
    """Raises on every LLM call."""

    def invoke(self, prompt: str) -> str:
        raise ConnectionError("Ollama unreachable")


class _FailOnNthCallBackend:
    """Succeeds for the first N-1 calls, then raises — simulates a mid-graph failure."""

    def __init__(self, fail_on: int, delegate: LLMBackend = MockLLMBackend()) -> None:
        self._fail_on = fail_on
        self._delegate = delegate
        self._calls = 0

    def invoke(self, prompt: str) -> str:
        self._calls += 1
        if self._calls >= self._fail_on:
            raise TimeoutError("LLM request timed out")
        return self._delegate.invoke(prompt)


@pytest.fixture
def _bundle(trained_bundle_dir):
    return load_bundle(trained_bundle_dir)


def _high_stress_payload(window_size: int) -> dict:
    return {
        "cpu": [0.95] * window_size,
        "memory": [0.95] * window_size,
        "latency": [500.0] * window_size,
    }


def test_api_survives_agent_connection_error(_bundle):
    failing_graph = build_graph(_AlwaysFailingBackend())
    app.dependency_overrides[get_bundle] = lambda: _bundle
    app.dependency_overrides[get_agent_graph] = lambda: failing_graph
    try:
        r = TestClient(app).post("/analyze", json=_high_stress_payload(_bundle.window_size))
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body["anomaly"], bool)
        assert 0.0 <= body["score"] <= 1.0
        assert body["analysis"] is None
        assert body["severity"] is None
        assert body["recommendation"] is None
    finally:
        app.dependency_overrides.clear()


def test_api_survives_mid_graph_timeout(_bundle):
    """A failure partway through the graph (e.g., on the severity node) is also caught."""
    partial_graph = build_graph(_FailOnNthCallBackend(fail_on=2))
    app.dependency_overrides[get_bundle] = lambda: _bundle
    app.dependency_overrides[get_agent_graph] = lambda: partial_graph
    try:
        r = TestClient(app).post("/analyze", json=_high_stress_payload(_bundle.window_size))
        assert r.status_code == 200
        body = r.json()
        assert body["analysis"] is None
        assert body["severity"] is None
        assert body["recommendation"] is None
    finally:
        app.dependency_overrides.clear()


def test_detection_fields_unaffected_by_agent_failure(_bundle):
    """Core detection result (anomaly, score, threshold, model_used) must be correct regardless."""
    # Get the expected values with no agent, then compare to the failing-agent run.
    app.dependency_overrides[get_bundle] = lambda: _bundle
    app.dependency_overrides[get_agent_graph] = lambda: None
    payload = _high_stress_payload(_bundle.window_size)
    try:
        baseline = TestClient(app).post("/analyze", json=payload).json()
    finally:
        app.dependency_overrides.clear()

    failing_graph = build_graph(_AlwaysFailingBackend())
    app.dependency_overrides[get_bundle] = lambda: _bundle
    app.dependency_overrides[get_agent_graph] = lambda: failing_graph
    try:
        degraded = TestClient(app).post("/analyze", json=payload).json()
    finally:
        app.dependency_overrides.clear()

    assert degraded["anomaly"] == baseline["anomaly"]
    assert degraded["score"] == baseline["score"]
    assert degraded["threshold"] == baseline["threshold"]
    assert degraded["model_used"] == baseline["model_used"]


def test_agent_failure_is_logged(_bundle, caplog):
    failing_graph = build_graph(_AlwaysFailingBackend())
    app.dependency_overrides[get_bundle] = lambda: _bundle
    app.dependency_overrides[get_agent_graph] = lambda: failing_graph
    payload = _high_stress_payload(_bundle.window_size)
    try:
        with caplog.at_level(logging.WARNING, logger="anomaliz.api.main"):
            r = TestClient(app).post("/analyze", json=payload)
        assert r.status_code == 200
        # A warning is only emitted when the anomaly path is actually triggered
        if r.json()["anomaly"] is False:
            pytest.skip("payload did not trigger anomaly; logging path not exercised")
        assert any("Agent invocation failed" in m for m in caplog.messages)
    finally:
        app.dependency_overrides.clear()
