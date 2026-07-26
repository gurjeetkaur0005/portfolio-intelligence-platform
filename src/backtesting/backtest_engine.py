from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src.backtesting.performance_metrics import (
    calculate_annualized_return,
    calculate_maximum_drawdown,
    calculate_sharpe_ratio,
    calculate_total_return,
    calculate_volatility,
)

@dataclass(frozen=True)
class BacktestResult:
    """
    Store the results produced by a portfolio backtest.
    """

    portfolio_history: pd.DataFrame
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    maximum_drawdown: float

def apply_market_returns(
    asset_values: np.ndarray,
    period_returns: np.ndarray,
) -> np.ndarray:
    """
    Apply one period of market returns to the asset values.

    Parameters
    ----------
    asset_values:
        Current monetary value of each asset.

    period_returns:
        Return earned by each asset during the current period.

    Returns
    -------
    np.ndarray
        Updated asset values after applying market returns.
    """

    asset_values = np.asarray(
        asset_values,
        dtype=float,
    )

    period_returns = np.asarray(
        period_returns,
        dtype=float,
    )

    if asset_values.ndim != 1:
        raise ValueError(
            "Asset values must be a one-dimensional array.",
        )

    if period_returns.ndim != 1:
        raise ValueError(
            "Period returns must be a one-dimensional array.",
        )

    if len(asset_values) != len(period_returns):
        raise ValueError(
            "Asset values and period returns must have the same length.",
        )

    if np.any(asset_values < 0):
        raise ValueError(
            "Asset values cannot be negative.",
        )

    if np.any(period_returns < -1.0):
        raise ValueError(
            "Period returns cannot be less than -100%.",
        )

    updated_asset_values = asset_values * (
        1.0 + period_returns
    )

    return updated_asset_values

def calculate_current_weights(
    asset_values: np.ndarray,
) -> np.ndarray:
    """
    Calculate the current portfolio weights from asset values.
    """

    asset_values = np.asarray(
        asset_values,
        dtype=float,
    )

    if asset_values.ndim != 1:
        raise ValueError(
            "Asset values must be a one-dimensional array.",
        )

    if np.any(asset_values < 0):
        raise ValueError(
            "Asset values cannot be negative.",
        )

    total_portfolio_value = float(
        np.sum(asset_values)
    )

    if total_portfolio_value <= 0:
        raise ValueError(
            "Total portfolio value must be positive.",
        )

    current_weights = (
        asset_values / total_portfolio_value
    )

    return current_weights

def run_buy_and_hold_backtest(
    initial_weights: Sequence[float],
    market_returns: pd.DataFrame,
    initial_portfolio_value: float = 100_000.0,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> BacktestResult:
    """
    Run a buy-and-hold portfolio backtest.

    The portfolio is initialized using the supplied asset weights.
    Market returns are applied period by period without rebalancing.
    """

    _validate_backtest_inputs(
        initial_weights=initial_weights,
        market_returns=market_returns,
        initial_portfolio_value=initial_portfolio_value,
        periods_per_year=periods_per_year,
    )

    weights = np.asarray(
        initial_weights,
        dtype=float,
    )

    asset_names = list(
        market_returns.columns
    )

    asset_values = (
        weights * initial_portfolio_value
    )

    history_records: list[dict[str, object]] = []

    initial_record = _build_history_record(
        date="initial",
        asset_names=asset_names,
        asset_values=asset_values,
    )

    history_records.append(
        initial_record
    )

    for date, return_row in market_returns.iterrows():
        period_returns = return_row.to_numpy(
            dtype=float,
        )

        asset_values = apply_market_returns(
            asset_values=asset_values,
            period_returns=period_returns,
        )

        history_record = _build_history_record(
            date=date,
            asset_names=asset_names,
            asset_values=asset_values,
        )

        history_records.append(
            history_record
        )

    portfolio_history = pd.DataFrame(
        history_records
    )

    portfolio_values = portfolio_history[
        "portfolio_value"
    ].tolist()

    number_of_periods = len(
        market_returns
    )

    years = (
        number_of_periods / periods_per_year
    )

    total_return = calculate_total_return(
        portfolio_values
    )

    annualized_return = calculate_annualized_return(
        portfolio_values=portfolio_values,
        years=years,
    )

    volatility = calculate_volatility(
        portfolio_values=portfolio_values,
        periods_per_year=periods_per_year,
    )

    sharpe_ratio = calculate_sharpe_ratio(
        annualized_return=annualized_return,
        annualized_volatility=volatility,
        risk_free_rate=risk_free_rate,
    )

    maximum_drawdown = calculate_maximum_drawdown(
        portfolio_values
    )

    return BacktestResult(
        portfolio_history=portfolio_history,
        total_return=total_return,
        annualized_return=annualized_return,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        maximum_drawdown=maximum_drawdown,
    )


def _validate_backtest_inputs(
    initial_weights: Sequence[float],
    market_returns: pd.DataFrame,
    initial_portfolio_value: float,
    periods_per_year: int,
) -> None:
    """
    Validate the inputs required by the backtest.
    """

    weights = np.asarray(
        initial_weights,
        dtype=float,
    )

    if weights.ndim != 1:
        raise ValueError(
            "Initial weights must be one-dimensional.",
        )

    if len(weights) == 0:
        raise ValueError(
            "At least one initial weight is required.",
        )

    if initial_portfolio_value <= 0:
        raise ValueError(
            "Initial portfolio value must be positive.",
        )

    if periods_per_year <= 0:
        raise ValueError(
            "Periods per year must be positive.",
        )

    if market_returns.empty:
        raise ValueError(
            "Market returns cannot be empty.",
        )

    if len(weights) != len(
        market_returns.columns
    ):
        raise ValueError(
            "The number of initial weights must match "
            "the number of market-return columns.",
        )

    if np.any(weights < 0):
        raise ValueError(
            "Initial weights cannot be negative.",
        )

    if not np.isclose(
        np.sum(weights),
        1.0,
        atol=1e-6,
    ):
        raise ValueError(
            "Initial weights must sum to one.",
        )

    return_values = market_returns.to_numpy(
        dtype=float,
    )

    if not np.all(
        np.isfinite(return_values)
    ):
        raise ValueError(
            "Market returns must contain only finite numeric values.",
        )

    if np.any(return_values < -1.0):
        raise ValueError(
            "Market returns cannot be less than -100%.",
        )

def _build_history_record(
    date: object,
    asset_names: Sequence[str],
    asset_values: np.ndarray,
) -> dict[str, object]:
    """
    Build one portfolio-history record.
    """

    portfolio_value = float(
        np.sum(asset_values)
    )

    current_weights = calculate_current_weights(
        asset_values
    )

    record: dict[str, object] = {
        "date": date,
        "portfolio_value": portfolio_value,
    }

    for asset_name, asset_value, weight in zip(
        asset_names,
        asset_values,
        current_weights,
        strict=True,
    ):
        record[f"{asset_name}_value"] = float(
            asset_value
        )

        record[f"{asset_name}_weight"] = float(
            weight
        )

    return record

if __name__ == "__main__":
    sample_returns = pd.DataFrame(
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

    result = run_buy_and_hold_backtest(
        initial_weights=[
            0.60,
            0.30,
            0.10,
        ],
        market_returns=sample_returns,
        initial_portfolio_value=100_000.0,
        periods_per_year=12,
    )

    print(result.portfolio_history)
    print()
    print(f"Total return: {result.total_return:.4f}")
    print(
        "Annualized return: "
        f"{result.annualized_return:.4f}"
    )
    print(f"Volatility: {result.volatility:.4f}")
    print(f"Sharpe ratio: {result.sharpe_ratio:.4f}")
    print(
        "Maximum drawdown: "
        f"{result.maximum_drawdown:.4f}"
    )
