from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math


BUY_AND_HOLD_NAME = "Buy & Hold"
THRESHOLD_REBALANCING_NAME = "Threshold Rebalancing"
SIMILARITY_TOLERANCE = 1e-12


@dataclass(frozen=True)
class StrategyComparisonMetrics:
    """
    Store normalized performance metrics for one strategy.
    """

    strategy_name: str
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    maximum_drawdown: float
    transaction_costs: float = 0.0
    taxes_paid: float = 0.0
    number_of_rebalances: int = 0

    @property
    def total_implementation_cost(self) -> float:
        """
        Return transaction costs plus taxes paid.
        """

        return self.transaction_costs + self.taxes_paid


@dataclass(frozen=True)
class StrategyComparisonResult:
    """
    Store the side-by-side strategy comparison.
    """

    buy_and_hold: StrategyComparisonMetrics
    threshold_rebalancing: StrategyComparisonMetrics
    performance_summary: str


def compare_backtest_results(
    buy_and_hold_result: Any,
    threshold_rebalancing_result: Any,
) -> StrategyComparisonResult:
    """
    Compare buy-and-hold and threshold-rebalancing backtest results.

    Core performance metrics are read from the supplied backtest result
    objects. Threshold implementation costs are read from explicit
    aggregate fields when available, or derived from threshold backtest
    history columns produced by the backtest engine.
    """

    buy_and_hold = StrategyComparisonMetrics(
        strategy_name=BUY_AND_HOLD_NAME,
        total_return=_get_required_float(
            buy_and_hold_result,
            "total_return",
        ),
        annualized_return=_get_required_float(
            buy_and_hold_result,
            "annualized_return",
        ),
        volatility=_get_required_float(
            buy_and_hold_result,
            "volatility",
        ),
        sharpe_ratio=_get_required_float(
            buy_and_hold_result,
            "sharpe_ratio",
        ),
        maximum_drawdown=_get_required_float(
            buy_and_hold_result,
            "maximum_drawdown",
        ),
    )

    threshold_rebalancing = StrategyComparisonMetrics(
        strategy_name=THRESHOLD_REBALANCING_NAME,
        total_return=_get_required_float(
            threshold_rebalancing_result,
            "total_return",
        ),
        annualized_return=_get_required_float(
            threshold_rebalancing_result,
            "annualized_return",
        ),
        volatility=_get_required_float(
            threshold_rebalancing_result,
            "volatility",
        ),
        sharpe_ratio=_get_required_float(
            threshold_rebalancing_result,
            "sharpe_ratio",
        ),
        maximum_drawdown=_get_required_float(
            threshold_rebalancing_result,
            "maximum_drawdown",
        ),
        transaction_costs=_get_transaction_costs(
            threshold_rebalancing_result
        ),
        taxes_paid=_get_taxes_paid(threshold_rebalancing_result),
        number_of_rebalances=_get_number_of_rebalances(
            threshold_rebalancing_result
        ),
    )

    _validate_strategy_metrics(buy_and_hold)
    _validate_strategy_metrics(threshold_rebalancing)

    return StrategyComparisonResult(
        buy_and_hold=buy_and_hold,
        threshold_rebalancing=threshold_rebalancing,
        performance_summary=_build_performance_summary(
            buy_and_hold=buy_and_hold,
            threshold_rebalancing=threshold_rebalancing,
        ),
    )


def _get_required_float(
    result: Any,
    field_name: str,
) -> float:
    """Read and validate a required numeric result field."""

    if not hasattr(result, field_name):
        raise ValueError(f"Missing required field: {field_name}.")

    return _to_finite_float(
        value=getattr(result, field_name),
        field_name=field_name,
    )


def _get_optional_float(
    result: Any,
    field_name: str,
    history_column: str,
) -> float:
    """Read an optional aggregate field or sum its history column."""

    if hasattr(result, field_name):
        return _to_finite_float(
            value=getattr(result, field_name),
            field_name=field_name,
        )

    portfolio_history = getattr(result, "portfolio_history", None)
    if (
        portfolio_history is not None
        and history_column in portfolio_history
    ):
        return _to_finite_float(
            value=portfolio_history[history_column].sum(),
            field_name=history_column,
        )

    return 0.0


def _get_transaction_costs(result: Any) -> float:
    """Return total transaction costs from a threshold result."""

    return _get_optional_float(
        result=result,
        field_name="total_transaction_costs",
        history_column="transaction_cost",
    )


def _get_taxes_paid(result: Any) -> float:
    """Return total taxes paid from a threshold result."""

    return _get_optional_float(
        result=result,
        field_name="total_taxes_paid",
        history_column="estimated_tax_liability",
    )


def _get_number_of_rebalances(result: Any) -> int:
    """Return the number of rebalances from a threshold result."""

    if hasattr(result, "number_of_rebalances"):
        value = getattr(result, "number_of_rebalances")
        return _to_non_negative_integer(
            value=value,
            field_name="number_of_rebalances",
        )

    portfolio_history = getattr(result, "portfolio_history", None)
    if portfolio_history is not None and "rebalanced" in portfolio_history:
        return _to_non_negative_integer(
            value=int(portfolio_history["rebalanced"].sum()),
            field_name="number_of_rebalances",
        )

    return 0


def _to_finite_float(
    value: Any,
    field_name: str,
) -> float:
    """Convert a value to a finite float."""

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number.")

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field_name} must be a finite number."
        ) from error

    if not math.isfinite(numeric_value):
        raise ValueError(f"{field_name} must be a finite number.")

    return numeric_value


def _to_non_negative_integer(
    value: Any,
    field_name: str,
) -> int:
    """Convert and validate a non-negative integer field."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")

    if value < 0:
        raise ValueError(f"{field_name} cannot be negative.")

    return value


def _validate_strategy_metrics(
    metrics: StrategyComparisonMetrics,
) -> None:
    """Validate normalized strategy metrics."""

    if metrics.volatility < 0:
        raise ValueError("volatility cannot be negative.")

    if metrics.transaction_costs < 0:
        raise ValueError("transaction_costs cannot be negative.")

    if metrics.taxes_paid < 0:
        raise ValueError("taxes_paid cannot be negative.")

    if metrics.number_of_rebalances < 0:
        raise ValueError("number_of_rebalances cannot be negative.")


def _build_performance_summary(
    buy_and_hold: StrategyComparisonMetrics,
    threshold_rebalancing: StrategyComparisonMetrics,
) -> str:
    """Build a concise human-readable comparison summary."""

    return " ".join(
        [
            _compare_higher_is_better(
                metric_name="total return",
                buy_and_hold_value=buy_and_hold.total_return,
                threshold_value=threshold_rebalancing.total_return,
            ),
            _compare_lower_absolute_is_better(
                metric_name="volatility",
                buy_and_hold_value=buy_and_hold.volatility,
                threshold_value=threshold_rebalancing.volatility,
            ),
            _compare_lower_absolute_is_better(
                metric_name="maximum drawdown",
                buy_and_hold_value=buy_and_hold.maximum_drawdown,
                threshold_value=threshold_rebalancing.maximum_drawdown,
            ),
            (
                f"{THRESHOLD_REBALANCING_NAME} executed "
                f"{threshold_rebalancing.number_of_rebalances} "
                f"rebalances, with "
                f"{_format_currency(threshold_rebalancing.transaction_costs)} "
                "in transaction costs and "
                f"{_format_currency(threshold_rebalancing.taxes_paid)} "
                "in taxes paid."
            ),
        ]
    )


def _compare_higher_is_better(
    metric_name: str,
    buy_and_hold_value: float,
    threshold_value: float,
) -> str:
    """Compare a metric where the larger value is better."""

    if _values_are_similar(buy_and_hold_value, threshold_value):
        return f"Both strategies produced similar {metric_name}."

    if threshold_value > buy_and_hold_value:
        return (
            f"{THRESHOLD_REBALANCING_NAME} produced higher "
            f"{metric_name}."
        )

    return f"{BUY_AND_HOLD_NAME} produced higher {metric_name}."


def _compare_lower_absolute_is_better(
    metric_name: str,
    buy_and_hold_value: float,
    threshold_value: float,
) -> str:
    """Compare a risk metric where lower magnitude is better."""

    buy_and_hold_magnitude = abs(buy_and_hold_value)
    threshold_magnitude = abs(threshold_value)

    if _values_are_similar(
        buy_and_hold_magnitude,
        threshold_magnitude,
    ):
        return f"Both strategies produced similar {metric_name}."

    if threshold_magnitude < buy_and_hold_magnitude:
        return (
            f"{THRESHOLD_REBALANCING_NAME} produced lower "
            f"{metric_name}."
        )

    return f"{BUY_AND_HOLD_NAME} produced lower {metric_name}."


def _values_are_similar(
    left_value: float,
    right_value: float,
) -> bool:
    """Return whether two values are effectively equal."""

    return math.isclose(
        left_value,
        right_value,
        rel_tol=SIMILARITY_TOLERANCE,
        abs_tol=SIMILARITY_TOLERANCE,
    )


def _format_currency(value: float) -> str:
    """Format a monetary value for summary text."""

    return f"${value:,.2f}"
