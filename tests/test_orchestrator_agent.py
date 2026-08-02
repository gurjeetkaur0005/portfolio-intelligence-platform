from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.agents.orchestrator_agent import (
    AgentExecutionStatus,
    OrchestratorAgent,
    OrchestratorRequest,
)
from src.agents.explanation_agent import (
    PortfolioExplanation,
)
from src.agents.portfolio_analyst_agent import (
    PortfolioAnalysis,
)


class FakePortfolioAnalyst:
    """Portfolio analyst spy for orchestration tests."""

    def __init__(self) -> None:
        self.received_trade_list: pd.DataFrame | None = None

    def analyze(
        self,
        trade_list: pd.DataFrame,
    ) -> list[PortfolioAnalysis]:
        self.received_trade_list = trade_list.copy()

        return [
            _build_portfolio_analysis(),
        ]


class FakeExplanationAgent:
    """Explanation agent spy for orchestration tests."""

    def __init__(self) -> None:
        self.received_analyses: list[PortfolioAnalysis] | None = None

    def explain(
        self,
        analyses: list[PortfolioAnalysis],
    ) -> list[PortfolioExplanation]:
        self.received_analyses = list(analyses)

        return [
            PortfolioExplanation(
                portfolio_id="PORTFOLIO-001",
                client_summary="AI generated explanation.",
                advisor_summary="Advisor explanation.",
                compliance_summary="Compliance explanation.",
            ),
        ]


def _build_portfolio_analysis() -> PortfolioAnalysis:
    """Build a valid portfolio analysis object."""

    return PortfolioAnalysis(
        portfolio_id="PORTFOLIO-001",
        rebalance_required=True,
        highest_threshold_severity="high",
        threshold_breached=True,
        threshold_breach_count=1,
        assets_to_buy=("domestic_equity",),
        assets_to_sell=("fixed_income",),
        assets_to_hold=("cash",),
        total_transaction_cost=100.0,
        total_estimated_tax_liability=25.0,
        client_explanations=("Client trade explanation.",),
        advisor_explanations=("Advisor trade explanation.",),
        compliance_explanations=("Compliance trade explanation.",),
    )


def test_execute_rebalance_returns_success_response() -> None:
    expected_result = pd.DataFrame(
        {
            "portfolio_id": ["PORTFOLIO-001"],
            "asset": ["domestic_equity"],
            "action": ["BUY"],
        }
    )

    def fake_pipeline(
        number_of_clients: int = 1,
        evaluation_date: date | None = None,
        portfolio_value: float = 1_000_000.0,
        transaction_cost_rate: float = 0.002,
    ) -> pd.DataFrame:
        assert number_of_clients == 2
        assert evaluation_date == date(2026, 7, 29)
        assert portfolio_value == 500_000.0
        assert transaction_cost_rate == 0.001

        return expected_result

    agent = OrchestratorAgent(
        rebalance_pipeline=fake_pipeline,
    )

    request = OrchestratorRequest(
        number_of_clients=2,
        evaluation_date=date(2026, 7, 29),
        portfolio_value=500_000.0,
        transaction_cost_rate=0.001,
    )

    response = agent.execute_rebalance(request)

    assert response.status == AgentExecutionStatus.SUCCESS
    assert response.workflow_name == "portfolio_rebalancing"
    assert response.result is not None
    pd.testing.assert_frame_equal(
        response.result,
        expected_result,
    )


def test_execute_rebalance_returns_result_copy() -> None:
    pipeline_result = pd.DataFrame(
        {
            "portfolio_id": ["PORTFOLIO-001"],
        }
    )

    def fake_pipeline(
        number_of_clients: int = 1,
        evaluation_date: date | None = None,
        portfolio_value: float = 1_000_000.0,
        transaction_cost_rate: float = 0.002,
    ) -> pd.DataFrame:
        return pipeline_result

    agent = OrchestratorAgent(
        rebalance_pipeline=fake_pipeline,
    )

    response = agent.execute_rebalance(
        OrchestratorRequest(),
    )

    assert response.result is not None
    assert response.result is not pipeline_result

    pd.testing.assert_frame_equal(
        response.result,
        pipeline_result,
    )


def test_pipeline_failure_returns_failed_response() -> None:
    def failing_pipeline(
        number_of_clients: int = 1,
        evaluation_date: date | None = None,
        portfolio_value: float = 1_000_000.0,
        transaction_cost_rate: float = 0.002,
    ) -> pd.DataFrame:
        raise RuntimeError(
            "Optimizer failed to find a solution.",
        )

    agent = OrchestratorAgent(
        rebalance_pipeline=failing_pipeline,
    )

    response = agent.execute_rebalance(
        OrchestratorRequest(),
    )

    assert response.status == AgentExecutionStatus.FAILED
    assert response.result is None
    assert (
        "Optimizer failed to find a solution"
        in response.message
    )


def test_zero_clients_raises_value_error() -> None:
    agent = OrchestratorAgent()

    request = OrchestratorRequest(
        number_of_clients=0,
    )

    with pytest.raises(
        ValueError,
        match="number_of_clients must be greater than zero",
    ):
        agent.execute_rebalance(request)


def test_negative_portfolio_value_raises_value_error() -> None:
    agent = OrchestratorAgent()

    request = OrchestratorRequest(
        portfolio_value=-100_000.0,
    )

    with pytest.raises(
        ValueError,
        match="portfolio_value must be greater than zero",
    ):
        agent.execute_rebalance(request)


def test_negative_transaction_cost_rate_raises_error() -> None:
    agent = OrchestratorAgent()

    request = OrchestratorRequest(
        transaction_cost_rate=-0.01,
    )

    with pytest.raises(
        ValueError,
        match="transaction_cost_rate cannot be negative",
    ):
        agent.execute_rebalance(request)


def test_transaction_cost_rate_above_one_raises_error() -> None:
    agent = OrchestratorAgent()

    request = OrchestratorRequest(
        transaction_cost_rate=1.01,
    )

    with pytest.raises(
        ValueError,
        match="transaction_cost_rate cannot exceed 1.0",
    ):
        agent.execute_rebalance(request)


def test_non_dataframe_pipeline_result_raises_type_error() -> None:
    def invalid_pipeline(  # type: ignore[no-untyped-def]
        **kwargs,
    ) -> list[str]:
        return ["invalid result"]

    agent = OrchestratorAgent(
        rebalance_pipeline=invalid_pipeline,  # type: ignore[arg-type]
    )

    with pytest.raises(
        TypeError,
        match="must return a pandas DataFrame",
    ):
        agent.execute_rebalance(
            OrchestratorRequest(),
        )


def test_execute_rebalance_with_explanations_coordinates_agents() -> None:
    pipeline_result = pd.DataFrame(
        {
            "portfolio_id": ["PORTFOLIO-001"],
            "asset": ["domestic_equity"],
            "action": ["BUY"],
        }
    )

    def fake_pipeline(
        number_of_clients: int = 1,
        evaluation_date: date | None = None,
        portfolio_value: float = 1_000_000.0,
        transaction_cost_rate: float = 0.002,
    ) -> pd.DataFrame:
        return pipeline_result

    portfolio_analyst = FakePortfolioAnalyst()
    explanation_agent = FakeExplanationAgent()
    agent = OrchestratorAgent(
        rebalance_pipeline=fake_pipeline,
        portfolio_analyst=portfolio_analyst,
        explanation_agent=explanation_agent,
    )

    response = agent.execute_rebalance_with_explanations(
        OrchestratorRequest(),
    )

    assert response.status == AgentExecutionStatus.SUCCESS
    assert response.result is not None
    assert response.analyses == [_build_portfolio_analysis()]
    assert response.explanations == [
        PortfolioExplanation(
            portfolio_id="PORTFOLIO-001",
            client_summary="AI generated explanation.",
            advisor_summary="Advisor explanation.",
            compliance_summary="Compliance explanation.",
        ),
    ]
    assert portfolio_analyst.received_trade_list is not None
    pd.testing.assert_frame_equal(
        portfolio_analyst.received_trade_list,
        pipeline_result,
    )
    assert explanation_agent.received_analyses == [
        _build_portfolio_analysis(),
    ]


def test_execute_rebalance_with_explanations_handles_empty_trades() -> None:
    def fake_pipeline(
        number_of_clients: int = 1,
        evaluation_date: date | None = None,
        portfolio_value: float = 1_000_000.0,
        transaction_cost_rate: float = 0.002,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    agent = OrchestratorAgent(
        rebalance_pipeline=fake_pipeline,
    )

    response = agent.execute_rebalance_with_explanations(
        OrchestratorRequest(),
    )

    assert response.status == AgentExecutionStatus.SUCCESS
    assert response.analyses == []
    assert response.explanations == []
