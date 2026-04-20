from __future__ import annotations

import pytest

from anomaliz.agent.graph import build_graph, invoke_agent
from anomaliz.agent.llm import MockLLMBackend
from anomaliz.agent.state import AnomalyState


@pytest.fixture
def mock_graph():
    return build_graph(MockLLMBackend())


def _base_state() -> AnomalyState:
    return {
        "cpu": 92.0,
        "memory": 65.0,
        "latency": 180.0,
        "score": 0.87,
        "threshold": 0.65,
        "model_used": "lstm_forecaster",
        "analysis": "",
        "severity": "",
        "recommendation": "",
    }


def test_graph_populates_all_fields(mock_graph):
    result = mock_graph.invoke(_base_state())
    assert result["analysis"] != ""
    assert result["severity"] in ("low", "medium", "critical")
    assert result["recommendation"] != ""


def test_severity_normalization():
    llm = MockLLMBackend()
    # Override to return an invalid severity keyword
    original_invoke = llm.invoke
    calls = []

    def patched(prompt: str) -> str:
        calls.append(prompt)
        if "classify" in prompt.lower() or "severity" in prompt.lower() and "classify" in prompt.lower():
            return "UNKNOWN_GARBAGE"
        return original_invoke(prompt)

    llm.invoke = patched  # type: ignore[method-assign]
    graph = build_graph(llm)
    result = graph.invoke(_base_state())
    assert result["severity"] == "medium"


def test_invoke_agent_helper():
    graph = build_graph(MockLLMBackend())
    analysis, severity, recommendation = invoke_agent(
        graph,
        cpu=92.0,
        memory=65.0,
        latency=180.0,
        score=0.87,
        threshold=0.65,
        model_used="isolation_forest",
    )
    assert isinstance(analysis, str) and analysis
    assert severity in ("low", "medium", "critical")
    assert isinstance(recommendation, str) and recommendation


def test_graph_is_reusable(mock_graph):
    state = _base_state()
    r1 = mock_graph.invoke(state)
    r2 = mock_graph.invoke(state)
    assert r1["severity"] == r2["severity"]
    assert r1["analysis"] == r2["analysis"]
