from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting.backtest_engine import (
    BacktestResult,
    apply_market_returns,
    calculate_current_weights,
    run_buy_and_hold_backtest,
    run_threshold_rebalancing_backtest,
)
from src.data.portfolio_generator import ASSET_CLASSES


@pytest.fixture
def sample_market_returns() -> pd.DataFrame:
    """
    Create sample monthly market returns.
    """

    return pd.DataFrame(
        {
            "domestic_equity": [
                0.02,
                -0.01,
                0.03,
            ],
            "fixed_income": [
                0.005,
                0.004,
                0.006,
            ],
            "cash": [
                0.001,
                0.001,
                0.001,
            ],
        },
        index=pd.date_range(
            start="2026-01-01",
            periods=3,
            freq="ME",
        ),
    )


@pytest.fixture
def threshold_market_returns() -> pd.DataFrame:
    """
    Create returns using the configured asset-class schema.
    """

    return pd.DataFrame(
        {
            "domestic_equity": [
                0.20,
                -0.04,
                0.03,
            ],
            "international_equity": [
                -0.08,
                0.02,
                0.01,
            ],
            "fixed_income": [
                0.01,
                0.01,
                0.01,
            ],
            "real_estate": [
                -0.03,
                0.04,
                0.02,
            ],
            "commodities": [
                0.06,
                -0.02,
                0.01,
            ],
            "cash": [
                0.001,
                0.001,
                0.001,
            ],
        },
        index=pd.date_range(
            start="2026-01-01",
            periods=3,
            freq="ME",
        ),
    )


def test_apply_market_returns() -> None:
    """
    Asset values should update correctly.
    """

    asset_values = np.array(
        [
            50_000.0,
            30_000.0,
            20_000.0,
        ]
    )

    returns = np.array(
        [
            0.10,
            0.02,
            -0.05,
        ]
    )

    result = apply_market_returns(
        asset_values=asset_values,
        period_returns=returns,
    )

    expected = np.array(
        [
            55_000.0,
            30_600.0,
            19_000.0,
        ]
    )

    assert np.allclose(
        result,
        expected,
    )


def test_calculate_current_weights() -> None:
    """
    Portfolio weights should sum to one.
    """

    asset_values = np.array(
        [
            60_000.0,
            30_000.0,
            10_000.0,
        ]
    )

    weights = calculate_current_weights(
        asset_values,
    )

    assert np.isclose(
        np.sum(weights),
        1.0,
    )

    assert np.allclose(
        weights,
        [
            0.60,
            0.30,
            0.10,
        ],
    )


def test_buy_and_hold_backtest_runs(
    sample_market_returns: pd.DataFrame,
) -> None:
    """
    Buy-and-hold backtest should execute successfully.
    """

    result = run_buy_and_hold_backtest(
        initial_weights=[
            0.60,
            0.30,
            0.10,
        ],
        market_returns=sample_market_returns,
        initial_portfolio_value=100_000.0,
        periods_per_year=12,
    )

    assert not result.portfolio_history.empty

    assert result.total_return > -1.0

    assert result.volatility >= 0.0


def test_history_contains_expected_columns(
    sample_market_returns: pd.DataFrame,
) -> None:
    """
    Portfolio history should contain required columns.
    """

    result = run_buy_and_hold_backtest(
        initial_weights=[
            0.60,
            0.30,
            0.10,
        ],
        market_returns=sample_market_returns,
        periods_per_year=12,
    )

    expected_columns = {
        "date",
        "portfolio_value",
        "domestic_equity_value",
        "fixed_income_value",
        "cash_value",
        "domestic_equity_weight",
        "fixed_income_weight",
        "cash_weight",
    }

    assert expected_columns.issubset(
        result.portfolio_history.columns,
    )


def test_buy_and_hold_backtest_returns_drawdown_history(
    sample_market_returns: pd.DataFrame,
) -> None:
    """Backtest results include backend-calculated drawdown history."""

    result = run_buy_and_hold_backtest(
        initial_weights=[
            0.60,
            0.30,
            0.10,
        ],
        market_returns=sample_market_returns,
        periods_per_year=12,
    )

    assert len(result.drawdown_history) == len(result.portfolio_history)
    assert list(result.drawdown_history.columns) == [
        "period",
        "drawdown",
        "date",
    ]
    assert result.drawdown_history.iloc[0]["drawdown"] == pytest.approx(
        0.0
    )
    assert (
        result.drawdown_history["drawdown"].max()
        <= 0.0
    )
    assert result.maximum_drawdown == pytest.approx(
        result.drawdown_history["drawdown"].min()
    )


def test_invalid_weights_raise_error(
    sample_market_returns: pd.DataFrame,
) -> None:
    """
    Weights must sum to one.
    """

    with pytest.raises(
        ValueError,
        match="Initial weights must sum to one.",
    ):
        run_buy_and_hold_backtest(
            initial_weights=[
                0.50,
                0.30,
                0.30,
            ],
            market_returns=sample_market_returns,
        )


def test_empty_market_returns_raise_error() -> None:
    """
    Empty market returns should not be accepted.
    """

    empty_returns = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="Market returns cannot be empty.",
    ):
        run_buy_and_hold_backtest(
            initial_weights=[
                1.0,
            ],
            market_returns=empty_returns,
        )


def test_asset_value_and_return_length_mismatch() -> None:
    """
    Asset values and returns must have matching lengths.
    """

    with pytest.raises(
        ValueError,
        match=(
            "Asset values and period returns "
            "must have the same length."
        ),
    ):
        apply_market_returns(
            asset_values=np.array(
                [
                    1.0,
                    2.0,
                ]
            ),
            period_returns=np.array(
                [
                    0.10,
                ]
            ),
        )


def test_current_weights_sum_to_one_after_backtest(
    sample_market_returns: pd.DataFrame,
) -> None:
    """
    Final weights should still sum to one.
    """

    result = run_buy_and_hold_backtest(
        initial_weights=[
            0.60,
            0.30,
            0.10,
        ],
        market_returns=sample_market_returns,
        periods_per_year=12,
    )

    final_row = result.portfolio_history.iloc[-1]

    total_weight = (
        final_row["domestic_equity_weight"]
        + final_row["fixed_income_weight"]
        + final_row["cash_weight"]
    )

    assert np.isclose(
        total_weight,
        1.0,
    )


def test_threshold_rebalancing_backtest_runs(
    threshold_market_returns: pd.DataFrame,
) -> None:
    """
    Threshold rebalancing should produce a BacktestResult.
    """

    result = run_threshold_rebalancing_backtest(
        initial_weights=[
            0.35,
            0.20,
            0.25,
            0.10,
            0.05,
            0.05,
        ],
        target_weights=[
            0.30,
            0.20,
            0.30,
            0.10,
            0.05,
            0.05,
        ],
        market_returns=threshold_market_returns,
        initial_portfolio_value=100_000.0,
        drift_band=0.01,
        transaction_cost_rate=0.001,
        tax_rate=0.20,
        turnover_budget=0.30,
        periods_per_year=12,
    )

    assert isinstance(
        result,
        BacktestResult,
    )
    assert not result.portfolio_history.empty
    assert result.total_return > -1.0
    assert result.volatility >= 0.0


def test_threshold_rebalancing_history_contains_event_columns(
    threshold_market_returns: pd.DataFrame,
) -> None:
    """
    Threshold history should record trigger and execution metadata.
    """

    result = run_threshold_rebalancing_backtest(
        initial_weights=[
            0.35,
            0.20,
            0.25,
            0.10,
            0.05,
            0.05,
        ],
        target_weights=[
            0.30,
            0.20,
            0.30,
            0.10,
            0.05,
            0.05,
        ],
        market_returns=threshold_market_returns,
        drift_band=0.01,
        transaction_cost_rate=0.001,
        turnover_budget=0.30,
        periods_per_year=12,
    )

    expected_columns = {
        "rebalanced",
        "threshold_breached",
        "trigger_severity",
        "breach_ratio",
        "transaction_cost",
        "estimated_tax_liability",
        "trade_count",
        "turnover",
    }

    assert expected_columns.issubset(
        result.portfolio_history.columns,
    )


def test_threshold_rebalancing_backtest_returns_drawdown_history(
    threshold_market_returns: pd.DataFrame,
) -> None:
    """Threshold backtests include aligned drawdown history."""

    result = run_threshold_rebalancing_backtest(
        initial_weights=[
            0.35,
            0.20,
            0.25,
            0.10,
            0.05,
            0.05,
        ],
        target_weights=[
            0.30,
            0.20,
            0.30,
            0.10,
            0.05,
            0.05,
        ],
        market_returns=threshold_market_returns,
        initial_portfolio_value=100_000.0,
        drift_band=0.01,
        transaction_cost_rate=0.001,
        tax_rate=0.20,
        turnover_budget=0.30,
        periods_per_year=12,
    )

    assert len(result.drawdown_history) == len(result.portfolio_history)
    assert result.drawdown_history.iloc[0]["drawdown"] == pytest.approx(
        0.0
    )
    assert (
        result.drawdown_history["drawdown"].max()
        <= 0.0
    )
    assert result.maximum_drawdown == pytest.approx(
        result.drawdown_history["drawdown"].min()
    )
    assert result.portfolio_history["rebalanced"].any()
    assert (
        result.portfolio_history["transaction_cost"] >= 0.0
    ).all()
    assert (
        result.portfolio_history["estimated_tax_liability"] >= 0.0
    ).all()


def test_threshold_rebalancing_final_weights_sum_to_one(
    threshold_market_returns: pd.DataFrame,
) -> None:
    """
    Executed trades and costs should still leave valid weights.
    """

    result = run_threshold_rebalancing_backtest(
        initial_weights=[
            0.35,
            0.20,
            0.25,
            0.10,
            0.05,
            0.05,
        ],
        target_weights=[
            0.30,
            0.20,
            0.30,
            0.10,
            0.05,
            0.05,
        ],
        market_returns=threshold_market_returns,
        drift_band=0.01,
        turnover_budget=0.30,
        periods_per_year=12,
    )
    final_row = result.portfolio_history.iloc[-1]

    total_weight = sum(
        final_row[f"{asset_name}_weight"]
        for asset_name in ASSET_CLASSES
    )

    assert np.isclose(
        total_weight,
        1.0,
    )


def test_threshold_rebalancing_requires_configured_asset_schema() -> None:
    """
    Threshold rebalancing should reject unsupported return schemas.
    """

    unsupported_returns = pd.DataFrame(
        {
            "domestic_equity": [
                0.01,
            ],
            "fixed_income": [
                0.01,
            ],
            "cash": [
                0.001,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="Market-return columns must match configured asset classes.",
    ):
        run_threshold_rebalancing_backtest(
            initial_weights=[
                0.60,
                0.30,
                0.10,
            ],
            target_weights=[
                0.60,
                0.30,
                0.10,
            ],
            market_returns=unsupported_returns,
        )
