from __future__ import annotations

from typing import Sequence

import numpy as np


def calculate_total_return(
    portfolio_values: Sequence[float],
) -> float:
    """
    Calculate the cumulative portfolio return over the
    entire backtest period.
    """
    if len(portfolio_values) < 2:
        raise ValueError(
            "At least two portfolio values are required.",
        )

    starting_value = portfolio_values[0]
    ending_value = portfolio_values[-1]

    if starting_value <= 0:
        raise ValueError(
            "Starting portfolio value must be positive.",
        )

    total_return = (
        ending_value - starting_value
    ) / starting_value

    return float(total_return)


def calculate_annualized_return(
    portfolio_values: Sequence[float],
    years: float,
) -> float:
    """
    Calculate the annualized portfolio return (CAGR).
    """

    if len(portfolio_values) < 2:
        raise ValueError(
            "At least two portfolio values are required.",
        )

    if years <= 0:
        raise ValueError(
            "Investment period must be positive.",
        )

    starting_value = portfolio_values[0]
    ending_value = portfolio_values[-1]

    if starting_value <= 0:
        raise ValueError(
            "Starting portfolio value must be positive.",
        )

    annualized_return = (
        ending_value / starting_value
    ) ** (1 / years) - 1

    return float(annualized_return)


def calculate_volatility(
    portfolio_values: Sequence[float],
    periods_per_year: int = 252,
) -> float:
    """
    Calculate the annualized portfolio volatility.
    """

    if len(portfolio_values) < 2:
        raise ValueError(
            "At least two portfolio values are required.",
        )

    if periods_per_year <= 0:
        raise ValueError(
            "Periods per year must be positive.",
        )

    portfolio_value_array = np.asarray(
        portfolio_values,
        dtype=float,
    )

    period_returns = (
        portfolio_value_array[1:]
        / portfolio_value_array[:-1]
    ) - 1

    volatility = (
        np.std(
            period_returns,
            ddof=1,
        )
        * np.sqrt(periods_per_year)
    )

    return float(volatility)


def calculate_sharpe_ratio(
    annualized_return: float,
    annualized_volatility: float,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Calculate the portfolio Sharpe ratio.
    """

    if annualized_volatility < 0:
        raise ValueError(
            "Annualized volatility cannot be negative.",
        )

    if annualized_volatility == 0:
        raise ValueError(
            "Sharpe ratio cannot be calculated when volatility is zero.",
        )

    excess_return = (
        annualized_return - risk_free_rate
    )

    sharpe_ratio = (
        excess_return / annualized_volatility
    )

    return float(sharpe_ratio)


def calculate_maximum_drawdown(
    portfolio_values: Sequence[float],
) -> float:
    """
    Calculate the maximum drawdown of a portfolio.
    """

    if len(portfolio_values) < 2:
        raise ValueError(
            "At least two portfolio values are required.",
        )

    peak = portfolio_values[0]
    maximum_drawdown = 0.0

    for value in portfolio_values:

        if value > peak:
            peak = value

        drawdown = (
            peak - value
        ) / peak

        maximum_drawdown = max(
            maximum_drawdown,
            drawdown,
        )

    return float(maximum_drawdown)
