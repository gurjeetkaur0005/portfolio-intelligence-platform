from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.backtesting.performance_metrics import (
    calculate_annualized_return,
    calculate_maximum_drawdown,
    calculate_sharpe_ratio,
    calculate_total_return,
    calculate_volatility,
)
from src.data.portfolio_generator import ASSET_CLASSES
from src.execution.trade_list_generator import generate_trade_list
from src.execution.transaction_cost_estimator import (
    estimate_transaction_costs,
)
from src.explanations.explanation_generator import (
    generate_trade_explanations,
)
from src.monitoring.drift_calculator import calculate_drift
from src.optimization.portfolio_optimizer import PortfolioOptimizer
from src.pipeline.tax_adapter import (
    estimate_taxes_allowing_zero_holding_buys,
)
from src.pipeline.trade_enrichment import enrich_trade_data
from src.triggers.threshold_trigger import evaluate_threshold_triggers


ZERO_TOLERANCE = 1e-12
FloatArray: TypeAlias = NDArray[np.float64]


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

    weights: FloatArray = np.asarray(
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


def run_threshold_rebalancing_backtest(
    initial_weights: Sequence[float],
    target_weights: Sequence[float],
    market_returns: pd.DataFrame,
    initial_portfolio_value: float = 100_000.0,
    drift_band: float = 0.05,
    transaction_cost_rate: float = 0.002,
    tax_rate: float = 0.20,
    turnover_budget: float = 0.10,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    portfolio_id: str = "BACKTEST",
    covariance_matrix: np.ndarray | None = None,
) -> BacktestResult:
    """
    Run a threshold-triggered rebalancing backtest.

    The engine orchestrates the existing monitoring, trigger, optimizer,
    execution, tax, and explanation modules. Trades are executed against
    current asset values; asset values are never reset to target weights.
    """

    _validate_threshold_backtest_inputs(
        initial_weights=initial_weights,
        target_weights=target_weights,
        market_returns=market_returns,
        initial_portfolio_value=initial_portfolio_value,
        drift_band=drift_band,
        transaction_cost_rate=transaction_cost_rate,
        tax_rate=tax_rate,
        turnover_budget=turnover_budget,
        periods_per_year=periods_per_year,
        covariance_matrix=covariance_matrix,
    )

    asset_names = list(market_returns.columns)
    current_asset_values: FloatArray = (
        np.asarray(initial_weights, dtype=float)
        * initial_portfolio_value
    )
    target_weight_array: FloatArray = np.asarray(
        target_weights,
        dtype=float,
    )
    optimizer = PortfolioOptimizer(
        turnover_budget=turnover_budget,
    )
    covariance = _resolve_covariance_matrix(
        covariance_matrix=covariance_matrix,
        number_of_assets=len(asset_names),
    )
    history_records: list[dict[str, object]] = []

    initial_record = _build_history_record(
        date="initial",
        asset_names=asset_names,
        asset_values=current_asset_values,
    )
    initial_record.update(
        _build_backtest_event_fields(
            rebalanced=False,
            threshold_breached=False,
            trigger_severity="none",
            breach_ratio=0.0,
            transaction_cost=0.0,
            estimated_tax_liability=0.0,
            trade_count=0,
            turnover=0.0,
        )
    )
    history_records.append(initial_record)

    for date, return_row in market_returns.iterrows():
        period_returns = return_row.to_numpy(dtype=float)

        # Apply market returns.
        current_asset_values = apply_market_returns(
            asset_values=current_asset_values,
            period_returns=period_returns,
        )

        # Calculate current weights.
        current_weights = calculate_current_weights(
            current_asset_values
        )
        portfolio_value = float(
            np.sum(current_asset_values)
        )

        # Calculate drift.
        portfolio_frame = _build_portfolio_frame(
            portfolio_id=portfolio_id,
            asset_names=asset_names,
            current_weights=current_weights,
            target_weights=target_weight_array,
            drift_band=drift_band,
        )
        drift_results = calculate_drift(portfolio_frame)

        # Evaluate threshold triggers.
        threshold_results = evaluate_threshold_triggers(
            drift_results
        )
        trigger_row = threshold_results.iloc[0]

        rebalanced = False
        total_transaction_cost = 0.0
        total_tax_liability = 0.0
        trade_count = 0
        turnover = 0.0

        if bool(trigger_row["threshold_breached"]):
            # Optimize portfolio.
            optimization_result = optimizer.optimize(
                current_weights=current_weights,
                target_weights=target_weight_array,
                covariance_matrix=covariance,
            )

            if (
                optimization_result.trade_weights is None
                or optimization_result.post_trade_weights is None
            ):
                raise ValueError(optimization_result.message)

            # Generate trade list.
            trade_list = generate_trade_list(
                asset_names=asset_names,
                current_weights=current_weights,
                trade_weights=optimization_result.trade_weights,
                post_trade_weights=optimization_result.post_trade_weights,
            )

            # Estimate transaction costs.
            costed_trades = estimate_transaction_costs(
                trade_list=trade_list,
                portfolio_value=portfolio_value,
                transaction_cost_rate=transaction_cost_rate,
            )

            # Enrich trades for tax estimation.
            enriched_trades = enrich_trade_data(
                trade_list=costed_trades,
                portfolio_id=portfolio_id,
                portfolio_value=portfolio_value,
                tax_rate=tax_rate,
            )

            # Estimate trade taxes.
            tax_aware_trades = estimate_taxes_allowing_zero_holding_buys(
                enriched_trades
            )
            explained_trade_inputs = _add_threshold_context(
                trade_list=tax_aware_trades,
                threshold_row=trigger_row,
            )

            # Generate trade explanations.
            explained_trades = generate_trade_explanations(
                explained_trade_inputs
            )

            # Execute trades.
            current_asset_values = _execute_trades(
                asset_values=current_asset_values,
                asset_names=asset_names,
                trades=explained_trades,
            )

            # Update asset values for transaction costs and taxes.
            total_transaction_cost = float(
                explained_trades["transaction_cost"].sum()
            )
            total_tax_liability = float(
                explained_trades["estimated_tax_liability"].sum()
            )
            current_asset_values = _deduct_portfolio_costs(
                asset_values=current_asset_values,
                asset_names=asset_names,
                total_cost=(
                    total_transaction_cost
                    + total_tax_liability
                ),
            )

            rebalanced = _has_executable_trades(explained_trades)
            trade_count = int(
                (explained_trades["action"] != "HOLD").sum()
            )
            turnover = float(
                np.sum(
                    np.abs(optimization_result.trade_weights)
                )
            )

        # Save history.
        history_record = _build_history_record(
            date=date,
            asset_names=asset_names,
            asset_values=current_asset_values,
        )
        history_record.update(
            _build_backtest_event_fields(
                rebalanced=rebalanced,
                threshold_breached=bool(
                    trigger_row["threshold_breached"]
                ),
                trigger_severity=str(
                    trigger_row["trigger_severity"]
                ),
                breach_ratio=float(trigger_row["breach_ratio"]),
                transaction_cost=total_transaction_cost,
                estimated_tax_liability=total_tax_liability,
                trade_count=trade_count,
                turnover=turnover,
            )
        )
        history_records.append(history_record)

    portfolio_history = pd.DataFrame(history_records)

    # Compute performance metrics.
    return _build_backtest_result(
        portfolio_history=portfolio_history,
        number_of_periods=len(market_returns),
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
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

    weights: FloatArray = np.asarray(
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


def _validate_threshold_backtest_inputs(
    initial_weights: Sequence[float],
    target_weights: Sequence[float],
    market_returns: pd.DataFrame,
    initial_portfolio_value: float,
    drift_band: float,
    transaction_cost_rate: float,
    tax_rate: float,
    turnover_budget: float,
    periods_per_year: int,
    covariance_matrix: np.ndarray | None,
) -> None:
    """Validate inputs required by threshold rebalancing."""

    _validate_backtest_inputs(
        initial_weights=initial_weights,
        market_returns=market_returns,
        initial_portfolio_value=initial_portfolio_value,
        periods_per_year=periods_per_year,
    )

    if list(market_returns.columns) != ASSET_CLASSES:
        raise ValueError(
            "Market-return columns must match configured asset classes."
        )

    targets: FloatArray = np.asarray(
        target_weights,
        dtype=float,
    )

    if targets.ndim != 1:
        raise ValueError(
            "Target weights must be one-dimensional.",
        )

    if len(targets) != len(market_returns.columns):
        raise ValueError(
            "The number of target weights must match "
            "the number of market-return columns.",
        )

    if np.any(targets < 0):
        raise ValueError(
            "Target weights cannot be negative.",
        )

    if not np.isclose(
        np.sum(targets),
        1.0,
        atol=1e-6,
    ):
        raise ValueError(
            "Target weights must sum to one.",
        )

    if drift_band <= 0:
        raise ValueError(
            "Drift band must be positive.",
        )

    if transaction_cost_rate < 0:
        raise ValueError(
            "Transaction cost rate cannot be negative.",
        )

    if not 0 <= tax_rate <= 1:
        raise ValueError(
            "Tax rate must be between 0 and 1.",
        )

    if turnover_budget <= 0:
        raise ValueError(
            "Turnover budget must be positive.",
        )

    if covariance_matrix is None:
        return

    covariance = np.asarray(
        covariance_matrix,
        dtype=float,
    )
    number_of_assets = len(market_returns.columns)

    if covariance.shape != (
        number_of_assets,
        number_of_assets,
    ):
        raise ValueError(
            "Covariance matrix shape must match "
            "the number of assets.",
        )

    if not np.all(np.isfinite(covariance)):
        raise ValueError(
            "Covariance matrix must contain finite values.",
        )


def _resolve_covariance_matrix(
    covariance_matrix: np.ndarray | None,
    number_of_assets: int,
) -> np.ndarray:
    """Return a covariance matrix for optimization."""

    if covariance_matrix is None:
        return np.eye(
            number_of_assets,
            dtype=float,
        )

    return np.asarray(
        covariance_matrix,
        dtype=float,
    )


def _build_portfolio_frame(
    portfolio_id: str,
    asset_names: Sequence[str],
    current_weights: np.ndarray,
    target_weights: np.ndarray,
    drift_band: float,
) -> pd.DataFrame:
    """Build the one-row portfolio frame required by drift calculation."""

    record: dict[str, object] = {
        "portfolio_id": portfolio_id,
        "risk_category": "backtest",
        "drift_band": drift_band,
    }

    for asset_name, current_weight, target_weight in zip(
        asset_names,
        current_weights,
        target_weights,
        strict=True,
    ):
        record[f"current_{asset_name}"] = float(current_weight)
        record[f"target_{asset_name}"] = float(target_weight)

    return pd.DataFrame([record])


def _add_threshold_context(
    trade_list: pd.DataFrame,
    threshold_row: pd.Series,
) -> pd.DataFrame:
    """Add explanation threshold fields from trigger output."""

    result = trade_list.copy()

    result["threshold_breached"] = bool(
        threshold_row["threshold_breached"]
    )
    result["threshold_severity"] = str(
        threshold_row["trigger_severity"]
    )
    result["breach_ratio"] = float(
        threshold_row["breach_ratio"]
    )

    return result


def _execute_trades(
    asset_values: np.ndarray,
    asset_names: Sequence[str],
    trades: pd.DataFrame,
) -> np.ndarray:
    """Execute signed trade values against asset values."""

    updated_asset_values = asset_values.copy()

    for _, trade in trades.iterrows():
        asset_index = asset_names.index(
            str(trade["asset"])
        )
        updated_asset_values[asset_index] += float(
            trade["trade_value"]
        )

    updated_asset_values[
        np.abs(updated_asset_values) < ZERO_TOLERANCE
    ] = 0.0

    if np.any(updated_asset_values < -ZERO_TOLERANCE):
        raise ValueError(
            "Trade execution produced negative asset values."
        )

    return updated_asset_values


def _deduct_portfolio_costs(
    asset_values: np.ndarray,
    asset_names: Sequence[str],
    total_cost: float,
) -> np.ndarray:
    """Deduct transaction costs and taxes from portfolio values."""

    if total_cost <= ZERO_TOLERANCE:
        return asset_values

    updated_asset_values = asset_values.copy()

    if "cash" in asset_names:
        cash_index = asset_names.index("cash")
        cash_deduction = min(
            float(updated_asset_values[cash_index]),
            total_cost,
        )
        updated_asset_values[cash_index] -= cash_deduction
        total_cost -= cash_deduction

    if total_cost <= ZERO_TOLERANCE:
        return updated_asset_values

    available_value = float(
        np.sum(updated_asset_values)
    )

    if available_value <= total_cost:
        raise ValueError(
            "Portfolio value is insufficient to cover costs."
        )

    deduction_weights = (
        updated_asset_values / available_value
    )
    updated_asset_values = (
        updated_asset_values
        - (deduction_weights * total_cost)
    )

    updated_asset_values[
        np.abs(updated_asset_values) < ZERO_TOLERANCE
    ] = 0.0

    return updated_asset_values


def _has_executable_trades(
    trades: pd.DataFrame,
) -> bool:
    """Return whether a trade list contains BUY or SELL rows."""

    return bool(
        (trades["action"] != "HOLD").any()
    )


def _build_backtest_event_fields(
    rebalanced: bool,
    threshold_breached: bool,
    trigger_severity: str,
    breach_ratio: float,
    transaction_cost: float,
    estimated_tax_liability: float,
    trade_count: int,
    turnover: float,
) -> dict[str, object]:
    """Build history fields specific to the rebalancing strategy."""

    return {
        "rebalanced": rebalanced,
        "threshold_breached": threshold_breached,
        "trigger_severity": trigger_severity,
        "breach_ratio": breach_ratio,
        "transaction_cost": transaction_cost,
        "estimated_tax_liability": estimated_tax_liability,
        "trade_count": trade_count,
        "turnover": turnover,
    }


def _build_backtest_result(
    portfolio_history: pd.DataFrame,
    number_of_periods: int,
    periods_per_year: int,
    risk_free_rate: float,
) -> BacktestResult:
    """Compute performance metrics and return a BacktestResult."""

    portfolio_values = portfolio_history[
        "portfolio_value"
    ].tolist()
    years = number_of_periods / periods_per_year

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
