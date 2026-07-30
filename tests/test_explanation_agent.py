from __future__ import annotations

import pandas as pd
import pytest

from src.agents.explanation_agent import (
    ExplanationAgent,
    PortfolioExplanation,
)


def _build_explained_trades() -> pd.DataFrame:
    """Build explained trade rows for two portfolios."""

    return pd.DataFrame(
        [
            {
                "portfolio_id": "portfolio_1",
                "asset": "domestic_equity",
                "action": "SELL",
                "threshold_breached": True,
                "threshold_severity": "high",
                "transaction_cost": 100.0,
                "estimated_tax_liability": 250.0,
                "client_explanation": "Sell domestic equity.",
                "advisor_explanation": "Advisor sell explanation.",
                "compliance_explanation": "Compliance sell record.",
            },
            {
                "portfolio_id": "portfolio_1",
                "asset": "fixed_income",
                "action": "BUY",
                "threshold_breached": True,
                "threshold_severity": "medium",
                "transaction_cost": 80.0,
                "estimated_tax_liability": 0.0,
                "client_explanation": "Buy fixed income.",
                "advisor_explanation": "Advisor buy explanation.",
                "compliance_explanation": "Compliance buy record.",
            },
            {
                "portfolio_id": "portfolio_1",
                "asset": "cash",
                "action": "HOLD",
                "threshold_breached": False,
                "threshold_severity": "none",
                "transaction_cost": 0.0,
                "estimated_tax_liability": 0.0,
                "client_explanation": "Hold cash.",
                "advisor_explanation": "Advisor hold explanation.",
                "compliance_explanation": "Compliance hold record.",
            },
            {
                "portfolio_id": "portfolio_2",
                "asset": "cash",
                "action": "HOLD",
                "threshold_breached": False,
                "threshold_severity": "none",
                "transaction_cost": 0.0,
                "estimated_tax_liability": 0.0,
                "client_explanation": "Hold cash.",
                "advisor_explanation": "Advisor hold explanation.",
                "compliance_explanation": "Compliance hold record.",
            },
        ]
    )


def _build_raw_hold_trade() -> pd.DataFrame:
    """Build one raw trade row accepted by the explanation generator."""

    return pd.DataFrame(
        [
            {
                "portfolio_id": "portfolio_1",
                "asset": "cash",
                "action": "HOLD",
                "current_weight": 0.05,
                "trade_weight": 0.0,
                "post_trade_weight": 0.05,
                "trade_value": 0.0,
                "transaction_cost": 0.0,
                "estimated_tax_liability": 0.0,
                "threshold_breached": False,
                "threshold_severity": "none",
                "breach_ratio": 1.0,
            }
        ]
    )


def test_explanation_columns_already_exist_are_reused() -> None:
    called = False

    def fake_generator(trades: pd.DataFrame) -> pd.DataFrame:
        nonlocal called
        called = True
        return trades

    agent = ExplanationAgent(
        trade_explanation_generator=fake_generator,
    )

    result = agent.explain(_build_explained_trades())

    assert called is False
    assert result[0].trade_count == 3


def test_missing_explanation_columns_call_generator() -> None:
    called = False

    def fake_generator(trades: pd.DataFrame) -> pd.DataFrame:
        nonlocal called
        called = True
        result = trades.copy()
        result["client_explanation"] = "Client explanation."
        result["advisor_explanation"] = "Advisor explanation."
        result["compliance_explanation"] = "Compliance explanation."
        return result

    agent = ExplanationAgent(
        trade_explanation_generator=fake_generator,
    )

    result = agent.explain(_build_raw_hold_trade())

    assert called is True
    assert result[0].hold_count == 1


def test_multiple_portfolios_return_multiple_packages() -> None:
    result = ExplanationAgent().explain(_build_explained_trades())

    assert [item.portfolio_id for item in result] == [
        "portfolio_1",
        "portfolio_2",
    ]


def test_single_portfolio_returns_one_package() -> None:
    trades = _build_explained_trades().iloc[:3].copy()

    result = ExplanationAgent().explain(trades)

    assert len(result) == 1
    assert isinstance(result[0], PortfolioExplanation)


def test_zero_transaction_costs_are_supported() -> None:
    trades = _build_explained_trades()
    trades.loc[:, "transaction_cost"] = 0.0

    result = ExplanationAgent().explain(trades)

    assert result[0].total_transaction_cost == 0.0


def test_zero_taxes_are_supported() -> None:
    trades = _build_explained_trades()
    trades.loc[:, "estimated_tax_liability"] = 0.0

    result = ExplanationAgent().explain(trades)

    assert result[0].total_estimated_tax == 0.0


def test_explain_does_not_mutate_input() -> None:
    trades = _build_raw_hold_trade()
    original = trades.copy(deep=True)

    ExplanationAgent().explain(trades)

    pd.testing.assert_frame_equal(trades, original)


def test_missing_required_columns_raise_value_error() -> None:
    trades = _build_explained_trades().drop(
        columns=["transaction_cost"]
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        ExplanationAgent().explain(trades)


def test_invalid_actions_raise_value_error() -> None:
    trades = _build_explained_trades()
    trades.loc[0, "action"] = "WAIT"

    with pytest.raises(
        ValueError,
        match="BUY, SELL, or HOLD",
    ):
        ExplanationAgent().explain(trades)


def test_invalid_severity_raises_value_error() -> None:
    trades = _build_explained_trades()
    trades.loc[0, "threshold_severity"] = "urgent"

    with pytest.raises(
        ValueError,
        match="none, medium, high, or critical",
    ):
        ExplanationAgent().explain(trades)


def test_output_is_deterministic() -> None:
    trades = _build_explained_trades()
    agent = ExplanationAgent()

    first_result = agent.explain(trades)
    second_result = agent.explain(trades)

    assert first_result == second_result


def test_empty_dataframe_returns_empty_list() -> None:
    assert ExplanationAgent().explain(pd.DataFrame()) == []


def test_buy_only_package() -> None:
    trades = _build_explained_trades().iloc[[1]].copy()

    result = ExplanationAgent().explain(trades)[0]

    assert result.buy_count == 1
    assert result.sell_count == 0
    assert result.hold_count == 0


def test_sell_only_package() -> None:
    trades = _build_explained_trades().iloc[[0]].copy()

    result = ExplanationAgent().explain(trades)[0]

    assert result.buy_count == 0
    assert result.sell_count == 1
    assert result.hold_count == 0


def test_hold_only_package() -> None:
    trades = _build_explained_trades().iloc[[2]].copy()

    result = ExplanationAgent().explain(trades)[0]

    assert result.buy_count == 0
    assert result.sell_count == 0
    assert result.hold_count == 1
    assert "does not require trading" in result.client_summary


def test_summary_contains_required_communication() -> None:
    result = ExplanationAgent().explain(_build_explained_trades())[0]

    assert "drifted away" in result.client_summary
    assert "Portfolio portfolio_1" in result.advisor_summary
    assert "No financial calculations" in result.compliance_summary


def test_generator_must_return_dataframe() -> None:
    def invalid_generator(trades: pd.DataFrame) -> pd.DataFrame:
        return "invalid"  # type: ignore[return-value]

    agent = ExplanationAgent(
        trade_explanation_generator=invalid_generator,
    )

    with pytest.raises(
        TypeError,
        match="must return a pandas DataFrame",
    ):
        agent.explain(_build_raw_hold_trade())
