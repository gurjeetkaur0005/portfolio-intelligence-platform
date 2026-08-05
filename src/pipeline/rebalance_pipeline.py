from __future__ import annotations

from datetime import date
from typing import Any, Protocol

import numpy as np
import pandas as pd

from src.audit.audit_trail import create_audit_trail
from config.asset_classes import ASSET_CLASSES
from config.settings import RANDOM_SEED
from src.data.client_profile_generator import generate_client_profiles
from src.data.portfolio_generator import generate_portfolios
from src.execution.trade_list_generator import generate_trade_list
from src.execution.transaction_cost_estimator import (
    estimate_transaction_costs,
)
from src.explanations.explanation_generator import (
    generate_trade_explanations,
)
from src.monitoring.drift_calculator import calculate_drift
from src.optimization.optimization_models import OptimizationResult
from src.override.human_approval_engine import evaluate_trade_approvals
from src.pipeline.tax_adapter import (
    estimate_taxes_allowing_zero_holding_buys,
)
from src.pipeline.trade_enrichment import enrich_trade_data
from src.triggers.calendar_trigger import evaluate_calendar_triggers
from src.triggers.event_trigger import evaluate_event_triggers
from src.triggers.threshold_trigger import evaluate_threshold_triggers
from src.triggers.trigger_consolidator import consolidate_triggers


class PortfolioOptimizerProtocol(Protocol):
    """Optimizer interface required by the rebalance pipeline."""

    def optimize(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> OptimizationResult:
        """Optimize a portfolio allocation."""


def run_rebalance_pipeline(
    number_of_clients: int = 1,
    evaluation_date: date | None = None,
    portfolio_value: float = 1_000_000.0,
    transaction_cost_rate: float = 0.002,
    market_drop: float = 0.0,
    regulatory_change: bool = False,
    client_life_event: bool = False,
    corporate_action: bool = False,
    large_cash_flow: bool = False,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Run the complete portfolio rebalancing workflow.

    The pipeline orchestrates existing modules in order and adds only the
    enrichment required to pass tax fields into the tax-aware optimizer.
    """

    if evaluation_date is None:
        evaluation_date = date(2026, 1, 2)

    client_profiles = generate_client_profiles(
        number_of_clients=number_of_clients,
        seed=seed,
    )
    portfolios = generate_portfolios(
        client_profiles=client_profiles,
        seed=seed,
    )

    return run_rebalance_pipeline_for_inputs(
        client_profiles=client_profiles,
        portfolios=portfolios,
        evaluation_date=evaluation_date,
        portfolio_value=portfolio_value,
        transaction_cost_rate=transaction_cost_rate,
        market_drop=market_drop,
        regulatory_change=regulatory_change,
        client_life_event=client_life_event,
        corporate_action=corporate_action,
        large_cash_flow=large_cash_flow,
    )


def run_rebalance_pipeline_for_inputs(
    *,
    client_profiles: pd.DataFrame,
    portfolios: pd.DataFrame,
    evaluation_date: date | None = None,
    portfolio_value: float = 1_000_000.0,
    transaction_cost_rate: float = 0.002,
    market_drop: float = 0.0,
    regulatory_change: bool = False,
    client_life_event: bool = False,
    corporate_action: bool = False,
    large_cash_flow: bool = False,
) -> pd.DataFrame:
    """
    Run the rebalance workflow for supplied deterministic inputs.

    This entry point preserves the existing deterministic pipeline while
    allowing production callers to provide persisted portfolio input
    instead of synthetic generator output.
    """

    if evaluation_date is None:
        evaluation_date = date(2026, 1, 2)

    from src.optimization.portfolio_optimizer import PortfolioOptimizer

    drift_results = calculate_drift(portfolios)
    threshold_results = evaluate_threshold_triggers(drift_results)
    calendar_results = evaluate_calendar_triggers(
        portfolios=portfolios,
        evaluation_date=evaluation_date,
    )
    event_results = evaluate_event_triggers(
        portfolios=portfolios,
        evaluation_date=evaluation_date,
        market_drop=market_drop,
        regulatory_change=regulatory_change,
        client_life_event=client_life_event,
        corporate_action=corporate_action,
        large_cash_flow=large_cash_flow,
    )
    trigger_results = consolidate_triggers(
        threshold_results=threshold_results,
        calendar_results=calendar_results,
        event_results=event_results,
    )

    optimizer = PortfolioOptimizer()
    covariance_matrix = _build_covariance_matrix()
    trade_frames: list[pd.DataFrame] = []

    for portfolio in portfolios.itertuples(index=False):
        trigger_row = trigger_results.loc[
            trigger_results["portfolio_id"] == portfolio.portfolio_id
        ].iloc[0]

        if not bool(trigger_row["any_triggered"]):
            continue

        trade_frame = _rebalance_portfolio(
            portfolio=portfolio,
            trigger_row=trigger_row,
            client_profiles=client_profiles,
            optimizer=optimizer,
            covariance_matrix=covariance_matrix,
            portfolio_value=portfolio_value,
            transaction_cost_rate=transaction_cost_rate,
        )
        trade_frames.append(trade_frame)

    if not trade_frames:
        return pd.DataFrame()

    return pd.concat(
        trade_frames,
        ignore_index=True,
    )


def _rebalance_portfolio(
    portfolio: Any,
    trigger_row: pd.Series,
    client_profiles: pd.DataFrame,
    optimizer: PortfolioOptimizerProtocol,
    covariance_matrix: np.ndarray,
    portfolio_value: float,
    transaction_cost_rate: float,
) -> pd.DataFrame:
    """Optimize one portfolio and return final explained trades."""

    current_weights = _extract_weights(
        portfolio=portfolio,
        prefix="current",
    )
    target_weights = _extract_weights(
        portfolio=portfolio,
        prefix="target",
    )

    optimization_result = optimizer.optimize(
        current_weights=current_weights,
        target_weights=target_weights,
        covariance_matrix=covariance_matrix,
    )

    if (
        optimization_result.trade_weights is None
        or optimization_result.post_trade_weights is None
    ):
        raise ValueError(optimization_result.message)

    trade_list = generate_trade_list(
        asset_names=ASSET_CLASSES,
        current_weights=current_weights,
        trade_weights=optimization_result.trade_weights,
        post_trade_weights=optimization_result.post_trade_weights,
    )
    trade_list = _add_persisted_holding_values(
        trade_list=trade_list,
        portfolio=portfolio,
    )
    costed_trades = estimate_transaction_costs(
        trade_list=trade_list,
        portfolio_value=portfolio_value,
        transaction_cost_rate=transaction_cost_rate,
    )
    enriched_trades = enrich_trade_data(
        trade_list=costed_trades,
        portfolio_id=str(portfolio.portfolio_id),
        portfolio_value=portfolio_value,
        tax_rate=_get_tax_rate(
            client_profiles=client_profiles,
            portfolio_id=str(portfolio.portfolio_id),
        ),
    )
    tax_aware_trades = estimate_taxes_allowing_zero_holding_buys(
        enriched_trades
    )
    explained_trades_input = _add_threshold_context(
        trade_list=tax_aware_trades,
        trigger_row=trigger_row,
    )

    explained_trades = generate_trade_explanations(
        explained_trades_input
    )
    approval_ready_trades = _add_approval_and_audit_context(
        trade_list=explained_trades,
        trigger_row=trigger_row,
        client_profiles=client_profiles,
        portfolio_id=str(portfolio.portfolio_id),
    )
    approved_trades = evaluate_trade_approvals(
        approval_ready_trades
    )

    return create_audit_trail(approved_trades)


def _extract_weights(
    portfolio: Any,
    prefix: str,
) -> np.ndarray:
    """Extract ordered asset weights from a portfolio row."""

    return np.array(
        [
            getattr(portfolio, f"{prefix}_{asset}")
            for asset in ASSET_CLASSES
        ],
        dtype=float,
    )


def _add_persisted_holding_values(
    *,
    trade_list: pd.DataFrame,
    portfolio: Any,
) -> pd.DataFrame:
    """Add persisted holding value columns when supplied."""

    result = trade_list.copy()

    for value_prefix in [
        "current_value",
        "cost_basis",
    ]:
        values: list[float] = []

        for asset in ASSET_CLASSES:
            column_name = f"{value_prefix}_{asset}"

            if not hasattr(portfolio, column_name):
                return result

            values.append(
                float(getattr(portfolio, column_name))
            )

        result[value_prefix] = values

    return result


def _get_tax_rate(
    client_profiles: pd.DataFrame,
    portfolio_id: str,
) -> float:
    """Return the tax rate associated with a portfolio."""

    return float(
        _get_client_profile_row(
            client_profiles=client_profiles,
            portfolio_id=portfolio_id,
        )["tax_bracket"]
    )


def _get_prior_approval_required(
    client_profiles: pd.DataFrame,
    portfolio_id: str,
) -> bool:
    """Return whether the client requires prior trade approval."""

    return bool(
        _get_client_profile_row(
            client_profiles=client_profiles,
            portfolio_id=portfolio_id,
        )["prior_approval_required"]
    )


def _get_client_profile_row(
    client_profiles: pd.DataFrame,
    portfolio_id: str,
) -> pd.Series:
    """Return the client profile row associated with a portfolio."""

    matching_profiles = client_profiles.loc[
        client_profiles["portfolio_id"] == portfolio_id
    ]

    if matching_profiles.empty:
        raise ValueError(
            f"No client profile found for portfolio {portfolio_id}."
        )

    return matching_profiles.iloc[0]


def _add_threshold_context(
    trade_list: pd.DataFrame,
    trigger_row: pd.Series,
) -> pd.DataFrame:
    """Add threshold context required by explanation generation."""

    result = trade_list.copy()

    result["threshold_breached"] = bool(
        trigger_row["threshold_breached"]
    )
    result["threshold_severity"] = trigger_row["trigger_severity"]
    result["breach_ratio"] = float(trigger_row["breach_ratio"])

    return result


def _add_approval_and_audit_context(
    trade_list: pd.DataFrame,
    trigger_row: pd.Series,
    client_profiles: pd.DataFrame,
    portfolio_id: str,
) -> pd.DataFrame:
    """Add approval and audit fields not produced by trade modules."""

    result = trade_list.copy()

    result["prior_approval_required"] = (
        _get_prior_approval_required(
            client_profiles=client_profiles,
            portfolio_id=portfolio_id,
        )
    )
    result["final_trigger_type"] = str(
        trigger_row["final_trigger_type"]
    )
    result["final_priority"] = str(trigger_row["final_priority"])
    result["contributing_triggers"] = str(
        trigger_row["contributing_triggers"]
    )

    return result


def _build_covariance_matrix() -> np.ndarray:
    """Build a simple positive-semidefinite covariance matrix."""

    return np.eye(len(ASSET_CLASSES), dtype=float)
