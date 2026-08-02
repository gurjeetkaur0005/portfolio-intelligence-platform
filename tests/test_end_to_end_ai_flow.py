from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field

from src.agents.explanation_agent import ExplanationAgent
from src.agents.orchestrator_agent import (
    AgentExecutionStatus,
    OrchestratorAgent,
    OrchestratorRequest,
)
from src.llm.fake_language_model import FakeLanguageModel
from src.llm.gemini_language_model import GeminiLanguageModel
from src.llm.prompt_builder import PromptBuilder
from src.optimization.optimization_models import OptimizationResult


class FakePortfolioOptimizer:
    """Optimizer stand-in for deterministic end-to-end integration."""

    def optimize(
        self,
        current_weights,
        target_weights,
        covariance_matrix,
    ) -> OptimizationResult:
        trade_weights = target_weights - current_weights

        return OptimizationResult(
            status="optimal",
            trade_weights=trade_weights,
            post_trade_weights=target_weights,
            tracking_error_before=0.0,
            tracking_error_after=0.0,
            turnover=float(abs(trade_weights).sum()),
            objective_value=0.0,
            message="Fake optimization completed successfully.",
        )


@dataclass
class FakeGeminiUsageMetadata:
    prompt_token_count: int | None = 20
    candidates_token_count: int | None = 6


@dataclass
class FakeGeminiResponse:
    text: str | None = "Gemini generated client explanation."
    usage_metadata: FakeGeminiUsageMetadata | None = field(
        default_factory=FakeGeminiUsageMetadata
    )


class FakeGeminiModels:
    """Gemini models fake used by the provider adapter."""

    def __init__(self) -> None:
        self.call_count = 0

    def generate_content(
        self,
        *,
        model: str,
        contents: str,
        config,
    ) -> FakeGeminiResponse:
        self.call_count += 1

        return FakeGeminiResponse()


@dataclass
class FakeGeminiClient:
    models: FakeGeminiModels


def test_orchestrator_pipeline_agent_prompt_llm_flow(
    monkeypatch,
) -> None:
    fake_optimizer_module = types.SimpleNamespace(
        PortfolioOptimizer=FakePortfolioOptimizer,
    )
    monkeypatch.setitem(
        sys.modules,
        "src.optimization.portfolio_optimizer",
        fake_optimizer_module,
    )

    fake_model = FakeLanguageModel(
        responses=[
            "AI generated client explanation.",
        ]
    )
    explanation_agent = ExplanationAgent(
        prompt_builder=PromptBuilder(),
        language_model=fake_model,
    )
    orchestrator = OrchestratorAgent(
        explanation_agent=explanation_agent,
    )

    response = orchestrator.execute_rebalance_with_explanations(
        OrchestratorRequest(number_of_clients=1),
    )

    assert response.status == AgentExecutionStatus.SUCCESS
    assert response.result is not None
    assert not response.result.empty
    assert {"audit_id", "approval_status"}.issubset(
        response.result.columns
    )
    assert response.analyses is not None
    assert len(response.analyses) == 1
    assert response.explanations is not None
    assert len(response.explanations) == 1
    assert (
        response.explanations[0].client_summary
        == "AI generated client explanation."
    )
    assert fake_model.call_count == 1
    assert len(fake_model.requests) == 1
    assert "Supplied facts" in fake_model.requests[0].user_prompt


def test_orchestrator_pipeline_agent_prompt_gemini_flow(
    monkeypatch,
) -> None:
    fake_optimizer_module = types.SimpleNamespace(
        PortfolioOptimizer=FakePortfolioOptimizer,
    )
    monkeypatch.setitem(
        sys.modules,
        "src.optimization.portfolio_optimizer",
        fake_optimizer_module,
    )

    gemini_models = FakeGeminiModels()
    gemini_model = GeminiLanguageModel(
        client=FakeGeminiClient(models=gemini_models),
    )
    explanation_agent = ExplanationAgent(
        prompt_builder=PromptBuilder(),
        language_model=gemini_model,
    )
    orchestrator = OrchestratorAgent(
        explanation_agent=explanation_agent,
    )

    response = orchestrator.execute_rebalance_with_explanations(
        OrchestratorRequest(number_of_clients=1),
    )

    assert response.status == AgentExecutionStatus.SUCCESS
    assert response.explanations is not None
    assert (
        response.explanations[0].client_summary
        == "Gemini generated client explanation."
    )
    assert gemini_models.call_count == 1
