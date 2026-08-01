from __future__ import annotations

import pytest

from src.agents.explanation_agent import (
    ExplanationAgent,
)
from src.agents.portfolio_analyst_agent import (
    PortfolioAnalysis,
)


def _build_analysis() -> PortfolioAnalysis:
    return PortfolioAnalysis(
        portfolio_id="portfolio_1",
        rebalance_required=True,
        highest_threshold_severity="high",
        threshold_breached=True,
        threshold_breach_count=2,
        assets_to_buy=("fixed_income",),
        assets_to_sell=("domestic_equity",),
        assets_to_hold=("cash",),
        total_transaction_cost=180.0,
        total_estimated_tax_liability=250.0,
        client_explanations=(
            "Domestic equity will be reduced.",
            "Fixed income will be increased.",
            "Cash remains unchanged.",
        ),
        advisor_explanations=(
            "Sell domestic equity.",
            "Buy fixed income.",
            "Hold cash.",
        ),
        compliance_explanations=(
            "SELL recommendation recorded.",
            "BUY recommendation recorded.",
            "HOLD recommendation recorded.",
        ),
    )


def test_explain_returns_portfolio_explanation() -> None:
    agent = ExplanationAgent()

    results = agent.explain(
        [_build_analysis()]
    )

    assert len(results) == 1
    assert results[0].portfolio_id == "portfolio_1"


def test_client_summary_contains_portfolio_actions() -> None:
    agent = ExplanationAgent()

    result = agent.explain(
        [_build_analysis()]
    )[0]

    assert "requires rebalancing" in result.client_summary
    assert "fixed income" in result.client_summary
    assert "domestic equity" in result.client_summary


def test_advisor_summary_contains_analysis_facts() -> None:
    agent = ExplanationAgent()

    result = agent.explain(
        [_build_analysis()]
    )[0]

    assert "Threshold breach count: 2" in result.advisor_summary
    assert "Highest threshold severity: high" in (
        result.advisor_summary
    )
    assert "$180.00" in result.advisor_summary
    assert "$250.00" in result.advisor_summary


def test_compliance_summary_states_no_calculations() -> None:
    agent = ExplanationAgent()

    result = agent.explain(
        [_build_analysis()]
    )[0]

    assert (
        "No financial calculations or trade decisions"
        in result.compliance_summary
    )


def test_empty_analysis_list_returns_empty_list() -> None:
    agent = ExplanationAgent()

    assert agent.explain([]) == []


def test_non_list_input_raises_type_error() -> None:
    agent = ExplanationAgent()

    with pytest.raises(
        TypeError,
        match="must be provided as a list",
    ):
        agent.explain(  # type: ignore[arg-type]
            _build_analysis()
        )


def test_invalid_list_item_raises_type_error() -> None:
    agent = ExplanationAgent()

    with pytest.raises(
        TypeError,
        match="must contain PortfolioAnalysis objects",
    ):
        agent.explain(  # type: ignore[list-item]
            ["invalid"]
        )