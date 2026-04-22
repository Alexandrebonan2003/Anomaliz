from __future__ import annotations

import os
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..config.settings import AgentConfig


@runtime_checkable
class LLMBackend(Protocol):
    def invoke(self, prompt: str) -> str: ...


class OpenAIBackend:
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        from langchain_openai import ChatOpenAI

        self._llm = ChatOpenAI(model=model, api_key=api_key)

    def invoke(self, prompt: str) -> str:
        from langchain_core.messages import HumanMessage

        return str(self._llm.invoke([HumanMessage(content=prompt)]).content)


class OllamaBackend:
    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
    ) -> None:
        from langchain_community.chat_models import ChatOllama

        self._llm = ChatOllama(model=model, base_url=base_url)

    def invoke(self, prompt: str) -> str:
        from langchain_core.messages import HumanMessage

        return str(self._llm.invoke([HumanMessage(content=prompt)]).content)


class MockLLMBackend:
    """Deterministic backend for tests — responses are matched by prompt keyword."""

    def invoke(self, prompt: str) -> str:
        p = prompt.lower()
        if "classify severity" in p or ("classify" in p and "severity" in p):
            return "critical"
        if "action recommendation" in p or "devops engineer" in p:
            return "Investigate the affected service immediately and consider scaling or restarting the impacted component."
        return "An anomaly was detected indicating abnormal resource usage, possibly a CPU spike or memory leak."


def build_backend(cfg: AgentConfig) -> LLMBackend | None:
    """Return the configured LLM backend, or None when agent is disabled."""
    backend = cfg.backend.lower()
    if backend == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        return OpenAIBackend(model=cfg.openai_model, api_key=api_key)
    if backend == "ollama":
        return OllamaBackend(model=cfg.ollama_model, base_url=cfg.ollama_base_url)
    return None
