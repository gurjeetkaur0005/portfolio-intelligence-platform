from __future__ import annotations

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

MONETARY_COLUMNS = [
    "transaction_cost",
    "estimated_tax_liability",
]


class ExplanationGeneratorProtocol(Protocol):
    """Contract for a trade-level explanation generator."""

    def __call__(
        self,
        trade_list: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate explanations for each trade."""
        ...


@dataclass(frozen=True, slots=True)
class PortfolioAnalysis:
    """
    Store deterministic portfolio-level facts.

    Financial calculations are performed by upstream modules. This object
    only organizes and aggregates their existing outputs.

    It does not generate portfolio-level client, advisor, or compliance
    communication.
    """

    portfolio_id: object
    rebalance_required: bool
    highest_threshold_severity: str
    threshold_breached: bool
    threshold_breach_count: int
    assets_to_buy: tuple[str, ...]
    assets_to_sell: tuple[str, ...]
    assets_to_hold: tuple[str, ...]
    total_transaction_cost: float
    total_estimated_tax_liability: float
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
    - Trade actions
    - Individual trade explanations
    - Audience-specific portfolio summaries

    It only organizes and aggregates outputs produced by the existing
    deterministic financial engine.
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
        Generate one structured analysis for each portfolio.

        Existing explanation columns are reused. If they are absent, the
        configured deterministic explanation generator is called.

        The caller-owned DataFrame is never mutated.

        Args:
            trade_list:
                Trade results produced by the deterministic pipeline.

        Returns:
            One PortfolioAnalysis object per portfolio.

        Raises:
            TypeError:
                If trade_list is not a pandas DataFrame or the explanation
                generator returns an invalid type.

            ValueError:
                If required columns or values are invalid.
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
        """
        Return a safe DataFrame containing trade explanations.

        Existing explanation columns are reused to avoid unnecessary
        regeneration.
        """

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

    invalid_asset_names = explained_trades["asset"].map(
        lambda value: (
            not isinstance(value, str)
            or not value.strip()
        )
    )

    if invalid_asset_names.any():
        raise ValueError(
            "Asset names must be non-empty strings."
        )

    if explained_trades["action"].isna().any():
        raise ValueError(
            "Actions must not be missing."
        )

    invalid_actions = ~explained_trades["action"].isin(
        ALLOWED_ACTIONS
    )

    if invalid_actions.any():
        raise ValueError(
            "Action must be one of BUY, SELL, or HOLD."
        )

    if explained_trades["threshold_severity"].isna().any():
        raise ValueError(
            "Threshold severity must not be missing."
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

    if explained_trades["threshold_breached"].isna().any():
        raise ValueError(
            "Threshold breached values must not be missing."
        )

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

    if explained_trades[
        MONETARY_COLUMNS
    ].isna().any().any():
        raise ValueError(
            "Transaction costs and tax liabilities "
            "must not be missing."
        )

    monetary_values = explained_trades[
        MONETARY_COLUMNS
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

    if (monetary_values < 0.0).any().any():
        raise ValueError(
            "Transaction costs and tax liabilities "
            "must not be negative."
        )


def _validate_explanation_text(
    explained_trades: pd.DataFrame,
) -> None:
    """Validate generated or reused explanation text."""

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
    """Build one structured portfolio-level analysis."""

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

    threshold_breach_count = int(
        portfolio_trades["threshold_breached"].sum()
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

    return PortfolioAnalysis(
        portfolio_id=portfolio_id,
        rebalance_required=rebalance_required,
        highest_threshold_severity=highest_severity,
        threshold_breached=threshold_breached,
        threshold_breach_count=threshold_breach_count,
        assets_to_buy=assets_to_buy,
        assets_to_sell=assets_to_sell,
        assets_to_hold=assets_to_hold,
        total_transaction_cost=total_transaction_cost,
        total_estimated_tax_liability=(
            total_estimated_tax_liability
        ),
        client_explanations=_extract_explanations(
            portfolio_trades=portfolio_trades,
            column="client_explanation",
        ),
        advisor_explanations=_extract_explanations(
            portfolio_trades=portfolio_trades,
            column="advisor_explanation",
        ),
        compliance_explanations=_extract_explanations(
            portfolio_trades=portfolio_trades,
            column="compliance_explanation",
        ),
    )


def _extract_assets_by_action(
    portfolio_trades: pd.DataFrame,
    action: str,
) -> tuple[str, ...]:
    """Return assets associated with an existing trade action."""

    matching_assets = portfolio_trades.loc[
        portfolio_trades["action"] == action,
        "asset",
    ]

    return tuple(
        str(asset)
        for asset in matching_assets
    )


def _extract_explanations(
    portfolio_trades: pd.DataFrame,
    column: str,
) -> tuple[str, ...]:
    """Return validated explanations from one explanation column."""

    return tuple(
        str(explanation)
        for explanation in portfolio_trades[column]
    )


def _find_highest_severity(
    severities: pd.Series,
) -> str:
    """Return the highest existing threshold severity."""

    return max(
        (str(severity) for severity in severities),
        key=SEVERITY_RANK.__getitem__,
    )