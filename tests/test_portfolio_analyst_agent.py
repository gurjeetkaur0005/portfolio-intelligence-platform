from __future__ import annotations

import pandas as pd
import pytest

from src.agents.portfolio_analyst_agent import (
    PortfolioAnalystAgent,
)


def _build_explained_trades() -> pd.DataFrame:
    """Build valid explained trades for two portfolios."""

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
                "client_explanation": (
                    "Domestic equity will be reduced."
                ),
                "advisor_explanation": (
                    "Sell domestic equity."
                ),
                "compliance_explanation": (
                    "SELL recommendation recorded."
                ),
            },
            {
                "portfolio_id": "portfolio_1",
                "asset": "fixed_income",
                "action": "BUY",
                "threshold_breached": True,
                "threshold_severity": "medium",
                "transaction_cost": 80.0,
                "estimated_tax_liability": 0.0,
                "client_explanation": (
                    "Fixed income will be increased."
                ),
                "advisor_explanation": (
                    "Buy fixed income."
                ),
                "compliance_explanation": (
                    "BUY recommendation recorded."
                ),
            },
            {
                "portfolio_id": "portfolio_1",
                "asset": "cash",
                "action": "HOLD",
                "threshold_breached": False,
                "threshold_severity": "none",
                "transaction_cost": 0.0,
                "estimated_tax_liability": 0.0,
                "client_explanation": (
                    "Cash remains unchanged."
                ),
                "advisor_explanation": (
                    "Hold cash."
                ),
                "compliance_explanation": (
                    "HOLD recommendation recorded."
                ),
            },
            {
                "portfolio_id": "portfolio_2",
                "asset": "domestic_equity",
                "action": "HOLD",
                "threshold_breached": False,
                "threshold_severity": "none",
                "transaction_cost": 0.0,
                "estimated_tax_liability": 0.0,
                "client_explanation": (
                    "Domestic equity remains unchanged."
                ),
                "advisor_explanation": (
                    "Hold domestic equity."
                ),
                "compliance_explanation": (
                    "HOLD recommendation recorded."
                ),
            },
        ]
    )


def test_analyze_returns_one_result_per_portfolio() -> None:
    agent = PortfolioAnalystAgent()

    results = agent.analyze(
        _build_explained_trades()
    )

    assert len(results) == 2
    assert results[0].portfolio_id == "portfolio_1"
    assert results[1].portfolio_id == "portfolio_2"


def test_analysis_identifies_buy_sell_and_hold_assets() -> None:
    agent = PortfolioAnalystAgent()

    result = agent.analyze(
        _build_explained_trades()
    )[0]

    assert result.assets_to_buy == ("fixed_income",)
    assert result.assets_to_sell == ("domestic_equity",)
    assert result.assets_to_hold == ("cash",)


def test_analysis_recommends_rebalancing_for_trades() -> None:
    agent = PortfolioAnalystAgent()

    result = agent.analyze(
        _build_explained_trades()
    )[0]

    assert result.rebalance_required is True


def test_analysis_does_not_recommend_hold_only_portfolio() -> None:
    agent = PortfolioAnalystAgent()

    result = agent.analyze(
        _build_explained_trades()
    )[1]

    assert result.rebalance_required is False
    assert result.assets_to_buy == ()
    assert result.assets_to_sell == ()
    assert result.assets_to_hold == (
        "domestic_equity",
    )


def test_analysis_uses_highest_existing_severity() -> None:
    agent = PortfolioAnalystAgent()

    result = agent.analyze(
        _build_explained_trades()
    )[0]

    assert result.threshold_breached is True
    assert result.highest_threshold_severity == "high"


def test_analysis_aggregates_costs_and_taxes() -> None:
    agent = PortfolioAnalystAgent()

    result = agent.analyze(
        _build_explained_trades()
    )[0]

    assert result.total_transaction_cost == pytest.approx(
        180.0
    )
    assert (
        result.total_estimated_tax_liability
        == pytest.approx(250.0)
    )


def test_analysis_reuses_existing_explanations() -> None:
    explanation_generator_called = False

    def fake_explanation_generator(
        trade_list: pd.DataFrame,
    ) -> pd.DataFrame:
        nonlocal explanation_generator_called
        explanation_generator_called = True
        return trade_list

    agent = PortfolioAnalystAgent(
        explanation_generator=fake_explanation_generator
    )

    result = agent.analyze(
        _build_explained_trades()
    )[0]

    assert explanation_generator_called is False
    assert result.client_explanations == (
        "Domestic equity will be reduced.",
        "Fixed income will be increased.",
        "Cash remains unchanged.",
    )


def test_missing_explanations_calls_generator() -> None:
    raw_trades = pd.DataFrame(
        [
            {
                "portfolio_id": "portfolio_1",
                "asset": "cash",
                "action": "HOLD",
                "threshold_breached": False,
                "threshold_severity": "none",
                "transaction_cost": 0.0,
                "estimated_tax_liability": 0.0,
            }
        ]
    )

    generator_called = False

    def fake_explanation_generator(
        trade_list: pd.DataFrame,
    ) -> pd.DataFrame:
        nonlocal generator_called
        generator_called = True

        result = trade_list.copy()
        result["client_explanation"] = "Client explanation."
        result["advisor_explanation"] = "Advisor explanation."
        result["compliance_explanation"] = (
            "Compliance explanation."
        )
        return result

    agent = PortfolioAnalystAgent(
        explanation_generator=fake_explanation_generator
    )

    results = agent.analyze(raw_trades)

    assert generator_called is True
    assert len(results) == 1
    assert results[0].client_explanations == (
        "Client explanation.",
    )


def test_analyze_does_not_mutate_input() -> None:
    trades = _build_explained_trades()
    original = trades.copy(deep=True)

    agent = PortfolioAnalystAgent()
    agent.analyze(trades)

    pd.testing.assert_frame_equal(
        trades,
        original,
    )


def test_empty_dataframe_returns_empty_list() -> None:
    agent = PortfolioAnalystAgent()

    results = agent.analyze(pd.DataFrame())

    assert results == []


def test_non_dataframe_raises_type_error() -> None:
    agent = PortfolioAnalystAgent()

    with pytest.raises(
        TypeError,
        match="must be a pandas DataFrame",
    ):
        agent.analyze([])  # type: ignore[arg-type]


def test_missing_required_column_raises_value_error() -> None:
    trades = _build_explained_trades().drop(
        columns=["transaction_cost"]
    )

    agent = PortfolioAnalystAgent()

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        agent.analyze(trades)


def test_invalid_action_raises_value_error() -> None:
    trades = _build_explained_trades()
    trades.loc[0, "action"] = "WAIT"

    agent = PortfolioAnalystAgent()

    with pytest.raises(
        ValueError,
        match="BUY, SELL, or HOLD",
    ):
        agent.analyze(trades)


def test_invalid_severity_raises_value_error() -> None:
    trades = _build_explained_trades()
    trades.loc[0, "threshold_severity"] = "extreme"

    agent = PortfolioAnalystAgent()

    with pytest.raises(
        ValueError,
        match="none, medium, high, or critical",
    ):
        agent.analyze(trades)


def test_negative_transaction_cost_raises_value_error() -> None:
    trades = _build_explained_trades()
    trades.loc[0, "transaction_cost"] = -1.0

    agent = PortfolioAnalystAgent()

    with pytest.raises(
        ValueError,
        match="must not be negative",
    ):
        agent.analyze(trades)


def test_generator_must_return_dataframe() -> None:
    raw_trades = pd.DataFrame(
        {
            "portfolio_id": ["portfolio_1"],
        }
    )

    def invalid_generator(
        trade_list: pd.DataFrame,
    ) -> pd.DataFrame:
        return "invalid"  # type: ignore[return-value]

    agent = PortfolioAnalystAgent(
        explanation_generator=invalid_generator
    )

    with pytest.raises(
        TypeError,
        match="must return a pandas DataFrame",
    ):
        agent.analyze(raw_trades)