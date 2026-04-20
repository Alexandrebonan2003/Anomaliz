from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .llm import LLMBackend
from .nodes import make_analyze_node, make_recommend_node, make_severity_node
from .state import AnomalyState


def build_graph(llm: LLMBackend):
    """Compile the anomaly analysis graph with the given LLM backend."""
    graph = StateGraph(AnomalyState)
    graph.add_node("analyze", make_analyze_node(llm))
    graph.add_node("severity", make_severity_node(llm))
    graph.add_node("recommend", make_recommend_node(llm))
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "severity")
    graph.add_edge("severity", "recommend")
    graph.add_edge("recommend", END)
    return graph.compile()


def invoke_agent(
    graph,
    cpu: float,
    memory: float,
    latency: float,
    score: float,
    threshold: float,
    model_used: str,
) -> tuple[str, str, str]:
    """Run the graph for one anomaly event; returns (analysis, severity, recommendation)."""
    initial: AnomalyState = {
        "cpu": cpu,
        "memory": memory,
        "latency": latency,
        "score": score,
        "threshold": threshold,
        "model_used": model_used,
        "analysis": "",
        "severity": "",
        "recommendation": "",
    }
    result = graph.invoke(initial)
    return result["analysis"], result["severity"], result["recommendation"]
