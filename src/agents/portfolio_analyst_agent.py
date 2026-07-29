from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from src.explanations.explanation_generator import (
    generate_trade_explanations,
)


REQUIRED_ANALYSIS_COLUMNS = {
    "portfolio_id",
    "asset",
    "action",
    "threshold_breached",
    "threshold_severity",
    "transaction_cost",
    "estimated_tax_liability",
    "client_explanation",
    "advisor_explanation",
    "compliance_explanation",
}

EXPLANATION_COLUMNS = {
    "client_explanation",
    "advisor_explanation",
    "compliance_explanation",
}

ALLOWED_ACTIONS = {
    "BUY",
    "SELL",
    "HOLD",
}

SEVERITY_RANK = {
    "none": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
}


class ExplanationGeneratorProtocol(Protocol):
    """Contract for a trade-explanation generator."""

    def __call__(
        self,
        trade_list: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate explanations for each trade."""
        ...


@dataclass(frozen=True, slots=True)
class PortfolioAnalysis:
    """
    Store a deterministic portfolio-level analysis.

    Financial calculations are performed by upstream modules. This object
    only summarizes their already-calculated outputs.
    """

    portfolio_id: object
    rebalance_required: bool
    highest_threshold_severity: str
    threshold_breached: bool
    assets_to_buy: tuple[str, ...]
    assets_to_sell: tuple[str, ...]
    assets_to_hold: tuple[str, ...]
    total_transaction_cost: float
    total_estimated_tax_liability: float
    client_summary: str
    advisor_summary: str
    client_explanations: tuple[str, ...]
    advisor_explanations: tuple[str, ...]
    compliance_explanations: tuple[str, ...]


class PortfolioAnalystAgent:
    """
    Produce portfolio-level analysis from deterministic trade results.

    The agent does not calculate:

    - Portfolio drift
    - Trigger decisions
    - Optimized weights
    - Trade values
    - Transaction costs
    - Tax liabilities
    - Individual trade explanations

    It reads and summarizes outputs produced by the existing financial
    engine and explanation generator.
    """

    def __init__(
        self,
        explanation_generator: ExplanationGeneratorProtocol = (
            generate_trade_explanations
        ),
    ) -> None:
        self._explanation_generator = explanation_generator

    def analyze(
        self,
        trade_list: pd.DataFrame,
    ) -> list[PortfolioAnalysis]:
        """
        Generate one analysis result for each portfolio.

        If explanation columns are already present, they are reused.
        Otherwise, the configured explanation generator is called.

        The input DataFrame is not mutated.
        """

        if not isinstance(trade_list, pd.DataFrame):
            raise TypeError(
                "Trade list must be a pandas DataFrame."
            )

        if trade_list.empty:
            return []

        explained_trades = self._prepare_explained_trades(
            trade_list
        )

        _validate_analysis_inputs(explained_trades)

        analyses: list[PortfolioAnalysis] = []

        grouped_trades = explained_trades.groupby(
            "portfolio_id",
            sort=False,
            dropna=False,
        )

        for portfolio_id, portfolio_trades in grouped_trades:
            analyses.append(
                _build_portfolio_analysis(
                    portfolio_id=portfolio_id,
                    portfolio_trades=portfolio_trades,
                )
            )

        return analyses

    def _prepare_explained_trades(
        self,
        trade_list: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return a safe DataFrame containing trade explanations."""

        if EXPLANATION_COLUMNS.issubset(trade_list.columns):
            return trade_list.copy()

        explained_trades = self._explanation_generator(
            trade_list.copy()
        )

        if not isinstance(explained_trades, pd.DataFrame):
            raise TypeError(
                "Explanation generator must return a pandas DataFrame."
            )

        return explained_trades.copy()


def _validate_analysis_inputs(
    explained_trades: pd.DataFrame,
) -> None:
    """Validate inputs required for portfolio-level analysis."""

    missing_columns = (
        REQUIRED_ANALYSIS_COLUMNS
        - set(explained_trades.columns)
    )

    if missing_columns:
        raise ValueError(
            "Explained trade list is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if explained_trades["portfolio_id"].isna().any():
        raise ValueError(
            "Portfolio IDs must not be missing."
        )

    if explained_trades["asset"].isna().any():
        raise ValueError(
            "Asset names must not be missing."
        )

    invalid_asset_names = (
        explained_trades["asset"]
        .map(lambda value: not isinstance(value, str) or not value.strip())
    )

    if invalid_asset_names.any():
        raise ValueError(
            "Asset names must be non-empty strings."
        )

    invalid_actions = ~explained_trades["action"].isin(
        ALLOWED_ACTIONS
    )

    if invalid_actions.any():
        raise ValueError(
            "Action must be one of BUY, SELL, or HOLD."
        )

    invalid_severities = (
        ~explained_trades["threshold_severity"].isin(
            SEVERITY_RANK
        )
    )

    if invalid_severities.any():
        raise ValueError(
            "Threshold severity must be one of "
            "none, medium, high, or critical."
        )

    _validate_threshold_flags(explained_trades)
    _validate_monetary_values(explained_trades)
    _validate_explanation_text(explained_trades)


def _validate_threshold_flags(
    explained_trades: pd.DataFrame,
) -> None:
    """Validate threshold-breached Boolean values."""

    invalid_flags = explained_trades[
        "threshold_breached"
    ].map(
        lambda value: not isinstance(
            value,
            (bool, np.bool_),
        )
    )

    if invalid_flags.any():
        raise ValueError(
            "Threshold breached must contain Boolean values."
        )


def _validate_monetary_values(
    explained_trades: pd.DataFrame,
) -> None:
    """Validate already-calculated costs and tax liabilities."""

    monetary_columns = [
        "transaction_cost",
        "estimated_tax_liability",
    ]

    if explained_trades[monetary_columns].isna().any().any():
        raise ValueError(
            "Transaction costs and tax liabilities "
            "must not be missing."
        )

    monetary_values = explained_trades[
        monetary_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if monetary_values.isna().any().any():
        raise ValueError(
            "Transaction costs and tax liabilities "
            "must be valid numbers."
        )

    if not np.isfinite(
        monetary_values.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Transaction costs and tax liabilities "
            "must be finite."
        )

    if (monetary_values < 0).any().any():
        raise ValueError(
            "Transaction costs and tax liabilities "
            "must not be negative."
        )


def _validate_explanation_text(
    explained_trades: pd.DataFrame,
) -> None:
    """Validate generated explanation columns."""

    for column in sorted(EXPLANATION_COLUMNS):
        invalid_values = explained_trades[column].map(
            lambda value: (
                not isinstance(value, str)
                or not value.strip()
            )
        )

        if invalid_values.any():
            raise ValueError(
                f"{column} must contain non-empty strings."
            )


def _build_portfolio_analysis(
    portfolio_id: object,
    portfolio_trades: pd.DataFrame,
) -> PortfolioAnalysis:
    """Build one portfolio-level analysis."""

    assets_to_buy = _extract_assets_by_action(
        portfolio_trades=portfolio_trades,
        action="BUY",
    )
    assets_to_sell = _extract_assets_by_action(
        portfolio_trades=portfolio_trades,
        action="SELL",
    )
    assets_to_hold = _extract_assets_by_action(
        portfolio_trades=portfolio_trades,
        action="HOLD",
    )

    rebalance_required = bool(
        assets_to_buy or assets_to_sell
    )

    threshold_breached = bool(
        portfolio_trades["threshold_breached"].any()
    )

    highest_severity = _find_highest_severity(
        portfolio_trades["threshold_severity"]
    )

    total_transaction_cost = float(
        portfolio_trades["transaction_cost"].sum()
    )

    total_estimated_tax_liability = float(
        portfolio_trades[
            "estimated_tax_liability"
        ].sum()
    )

    client_summary = _build_client_summary(
        portfolio_id=portfolio_id,
        rebalance_required=rebalance_required,
        assets_to_buy=assets_to_buy,
        assets_to_sell=assets_to_sell,
        total_transaction_cost=total_transaction_cost,
        total_estimated_tax_liability=(
            total_estimated_tax_liability
        ),
    )

    advisor_summary = _build_advisor_summary(
        portfolio_id=portfolio_id,
        rebalance_required=rebalance_required,
        threshold_breached=threshold_breached,
        highest_severity=highest_severity,
        assets_to_buy=assets_to_buy,
        assets_to_sell=assets_to_sell,
        assets_to_hold=assets_to_hold,
        total_transaction_cost=total_transaction_cost,
        total_estimated_tax_liability=(
            total_estimated_tax_liability
        ),
    )

    return PortfolioAnalysis(
        portfolio_id=portfolio_id,
        rebalance_required=rebalance_required,
        highest_threshold_severity=highest_severity,
        threshold_breached=threshold_breached,
        assets_to_buy=assets_to_buy,
        assets_to_sell=assets_to_sell,
        assets_to_hold=assets_to_hold,
        total_transaction_cost=total_transaction_cost,
        total_estimated_tax_liability=(
            total_estimated_tax_liability
        ),
        client_summary=client_summary,
        advisor_summary=advisor_summary,
        client_explanations=tuple(
            portfolio_trades[
                "client_explanation"
            ].astype(str)
        ),
        advisor_explanations=tuple(
            portfolio_trades[
                "advisor_explanation"
            ].astype(str)
        ),
        compliance_explanations=tuple(
            portfolio_trades[
                "compliance_explanation"
            ].astype(str)
        ),
    )


def _extract_assets_by_action(
    portfolio_trades: pd.DataFrame,
    action: str,
) -> tuple[str, ...]:
    """Return assets associated with a specified action."""

    matching_assets = portfolio_trades.loc[
        portfolio_trades["action"] == action,
        "asset",
    ]

    return tuple(
        str(asset)
        for asset in matching_assets
    )


def _find_highest_severity(
    severities: pd.Series,
) -> str:
    """Return the highest existing threshold severity."""

    return max(
        (str(severity) for severity in severities),
        key=SEVERITY_RANK.__getitem__,
    )


def _format_asset_list(
    assets: tuple[str, ...],
) -> str:
    """Format internal asset names for readable summaries."""

    if not assets:
        return "none"

    return ", ".join(
        asset.replace("_", " ")
        for asset in assets
    )


def _format_currency(
    value: float,
) -> str:
    """Format a monetary value using US dollars."""

    return f"${value:,.2f}"


def _build_client_summary(
    portfolio_id: object,
    rebalance_required: bool,
    assets_to_buy: tuple[str, ...],
    assets_to_sell: tuple[str, ...],
    total_transaction_cost: float,
    total_estimated_tax_liability: float,
) -> str:
    """Build a simple portfolio-level client summary."""

    if not rebalance_required:
        return (
            f"Portfolio {portfolio_id} does not require "
            "rebalancing. All positions will remain unchanged."
        )

    return (
        f"Portfolio {portfolio_id} requires rebalancing. "
        f"The plan increases {_format_asset_list(assets_to_buy)} "
        f"and decreases {_format_asset_list(assets_to_sell)}. "
        "The combined estimated transaction cost is "
        f"{_format_currency(total_transaction_cost)}, and the "
        "combined estimated tax liability is "
        f"{_format_currency(total_estimated_tax_liability)}."
    )


def _build_advisor_summary(
    portfolio_id: object,
    rebalance_required: bool,
    threshold_breached: bool,
    highest_severity: str,
    assets_to_buy: tuple[str, ...],
    assets_to_sell: tuple[str, ...],
    assets_to_hold: tuple[str, ...],
    total_transaction_cost: float,
    total_estimated_tax_liability: float,
) -> str:
    """Build a detailed portfolio-level advisor summary."""

    recommendation = (
        "Rebalancing is recommended."
        if rebalance_required
        else "Rebalancing is not required."
    )

    return (
        f"Portfolio {portfolio_id}: {recommendation} "
        f"Threshold breached: {threshold_breached}. "
        f"Highest threshold severity: {highest_severity}. "
        f"Assets to buy: {_format_asset_list(assets_to_buy)}. "
        f"Assets to sell: {_format_asset_list(assets_to_sell)}. "
        f"Assets to hold: {_format_asset_list(assets_to_hold)}. "
        "Total estimated transaction cost: "
        f"{_format_currency(total_transaction_cost)}. "
        "Total estimated tax liability: "
        f"{_format_currency(total_estimated_tax_liability)}."
    )