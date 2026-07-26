from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting.backtest_engine import (
    apply_market_returns,
    calculate_current_weights,
    run_buy_and_hold_backtest,
)


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