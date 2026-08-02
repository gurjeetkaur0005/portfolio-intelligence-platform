from __future__ import annotations

from functools import lru_cache

from src.agents.orchestrator_agent import OrchestratorAgent


@lru_cache(maxsize=1)
def get_orchestrator_agent() -> OrchestratorAgent:
    """
    Return the shared Orchestrator Agent instance.

    FastAPI calls this dependency when an endpoint needs access to the
    portfolio workflow.
    """

    return OrchestratorAgent()