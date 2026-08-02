from __future__ import annotations

from functools import lru_cache

from src.agents.orchestrator_agent import OrchestratorAgent
from src.llm.language_model import LanguageModelProtocol
from src.llm.language_model_factory import (
    LanguageModelProvider,
    create_language_model,
)


@lru_cache(maxsize=1)
def get_orchestrator_agent() -> OrchestratorAgent:
    """Return the shared Orchestrator Agent."""

    return OrchestratorAgent()


@lru_cache(maxsize=1)
def get_language_model() -> LanguageModelProtocol:
    """Return the configured language-model provider."""

    return create_language_model(
        provider=LanguageModelProvider.GEMINI,
    )
