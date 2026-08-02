from __future__ import annotations

from datetime import date

from fastapi.testclient import TestClient

from src.agents.explanation_agent import PortfolioExplanation
from src.agents.orchestrator_agent import (
    AgentExecutionStatus,
    OrchestratorRequest,
)
from src.api.dependencies import (
    get_language_model,
    get_orchestrator_agent,
)
from src.api.main import app
from src.llm.fake_language_model import FakeLanguageModel


client = TestClient(app)


class FakeExplanationResult:
    """Represent a successful fake orchestration result."""

    status = AgentExecutionStatus.SUCCESS
    workflow_name = "portfolio_rebalancing_with_explanations"
    message = "Workflow completed successfully."

    explanations = [
        PortfolioExplanation(
            portfolio_id="P00001",
            client_summary="Client summary.",
            advisor_summary="Advisor summary.",
            compliance_summary="Compliance summary.",
        )
    ]


class FakeExplanationOrchestrator:
    """Return a deterministic portfolio explanation result."""

    def execute_rebalance_with_explanations(
        self,
        request: OrchestratorRequest,
        language_model: object,
    ) -> FakeExplanationResult:
        assert request.number_of_clients == 1
        assert request.evaluation_date == date(2026, 8, 3)
        assert request.portfolio_value == 1_000_000.0
        assert request.transaction_cost_rate == 0.002

        return FakeExplanationResult()


def get_fake_orchestrator() -> FakeExplanationOrchestrator:
    """Return the fake orchestrator dependency."""

    return FakeExplanationOrchestrator()


def get_fake_language_model() -> FakeLanguageModel:
    """Return a deterministic fake language model."""

    return FakeLanguageModel(
        responses=[
            "This response is not called directly by the route."
        ]
    )


def test_rebalance_explain_returns_one_portfolio_summary() -> None:
    app.dependency_overrides[
        get_orchestrator_agent
    ] = get_fake_orchestrator

    app.dependency_overrides[
        get_language_model
    ] = get_fake_language_model

    try:
        response = client.post(
            "/rebalance/explain",
            json={
                "number_of_clients": 1,
                "evaluation_date": "2026-08-03",
                "portfolio_value": 1_000_000.0,
                "transaction_cost_rate": 0.002,
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "success"
    assert body["portfolio_count"] == 1
    assert len(body["explanations"]) == 1

    explanation = body["explanations"][0]

    assert explanation == {
        "portfolio_id": "P00001",
        "client_summary": "Client summary.",
        "advisor_summary": "Advisor summary.",
        "compliance_summary": "Compliance summary.",
    }