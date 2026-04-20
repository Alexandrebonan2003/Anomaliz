from __future__ import annotations

from typing import Callable

from .llm import LLMBackend
from .state import AnomalyState

_VALID_SEVERITIES = frozenset({"low", "medium", "critical"})

_ANALYZE_TMPL = (
    "You are a senior SRE. Given these system metrics at time T:\n"
    "CPU: {cpu:.1f}%, Memory: {memory:.1f}%, Latency: {latency:.1f}ms\n"
    "Anomaly score: {score:.2f} (threshold: {threshold:.2f}), detected by: {model_used}\n"
    "In 2 sentences, identify the probable anomaly type and its likely cause."
)

_SEVERITY_TMPL = (
    "Given this analysis: {analysis}\n"
    "And anomaly score: {score:.2f}\n"
    "Classify severity as exactly one of: low / medium / critical\n"
    "Reply with the keyword only."
)

_RECOMMEND_TMPL = (
    "Anomaly severity: {severity}. Analysis: {analysis}\n"
    "Generate a concrete 1–2 sentence action recommendation for a DevOps engineer."
)


def make_analyze_node(llm: LLMBackend) -> Callable[[AnomalyState], dict]:
    def analyze_node(state: AnomalyState) -> dict:
        prompt = _ANALYZE_TMPL.format(**state)
        return {"analysis": llm.invoke(prompt)}

    return analyze_node


def make_severity_node(llm: LLMBackend) -> Callable[[AnomalyState], dict]:
    def severity_node(state: AnomalyState) -> dict:
        prompt = _SEVERITY_TMPL.format(analysis=state["analysis"], score=state["score"])
        raw = llm.invoke(prompt).strip().lower()
        severity = raw if raw in _VALID_SEVERITIES else "medium"
        return {"severity": severity}

    return severity_node


def make_recommend_node(llm: LLMBackend) -> Callable[[AnomalyState], dict]:
    def recommend_node(state: AnomalyState) -> dict:
        prompt = _RECOMMEND_TMPL.format(
            severity=state["severity"], analysis=state["analysis"]
        )
        return {"recommendation": llm.invoke(prompt)}

    return recommend_node
