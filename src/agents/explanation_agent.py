from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from src.explanations.explanation_generator import (
    generate_trade_explanations,
)


EXPLANATION_COLUMNS = {
    "client_explanation",
    "advisor_explanation",
    "compliance_explanation",
}

REQUIRED_PACKAGE_COLUMNS = {
    "portfolio_id",
    "action",
    "threshold_breached",
    "threshold_severity",
    "transaction_cost",
    "estimated_tax_liability",
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

MONEY_COLUMNS = [
    "transaction_cost",
    "estimated_tax_liability",
]


class TradeExplanationGeneratorProtocol(Protocol):
    """Contract for trade-level explanation generation."""

    def __call__(
        self,
        trade_list: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return a DataFrame containing trade explanation columns."""


@dataclass(frozen=True, slots=True)
class PortfolioExplanation:
    """
    Portfolio-level communication package.

    The package summarizes deterministic trade outputs. It does not
    calculate drift, optimization results, trade actions, costs, or taxes.
    """

    portfolio_id: object
    client_summary: str
    advisor_summary: str
    compliance_summary: str
    trade_count: int
    buy_count: int
    sell_count: int
    hold_count: int
    total_transaction_cost: float
    total_estimated_tax: float
    highest_threshold_severity: str
    threshold_breach_count: int


class ExplanationAgent:
    """
    Build portfolio-level explanations from deterministic trade outputs.

    Trade-level explanations remain owned by
    ``generate_trade_explanations``. This agent reuses those columns when
    present and delegates to the generator only when they are missing.
    """

    def __init__(
        self,
        trade_explanation_generator: (
            TradeExplanationGeneratorProtocol
        ) = generate_trade_explanations,
    ) -> None:
        self._trade_explanation_generator = trade_explanation_generator

    def explain(
        self,
        trades: pd.DataFrame,
    ) -> list[PortfolioExplanation]:
        """
        Return one portfolio-level explanation per portfolio.

        The caller-owned DataFrame is never mutated.
        """

        if not isinstance(trades, pd.DataFrame):
            raise TypeError(
                "Trades must be provided as a pandas DataFrame."
            )

        if trades.empty:
            return []

        explained_trades = self._prepare_explained_trades(trades)
        _validate_explained_trades(explained_trades)

        return [
            _build_portfolio_explanation(
                portfolio_id=portfolio_id,
                portfolio_trades=portfolio_trades,
            )
            for portfolio_id, portfolio_trades in explained_trades.groupby(
                "portfolio_id",
                sort=False,
                dropna=False,
            )
        ]

    def _prepare_explained_trades(
        self,
        trades: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return a copied DataFrame with explanation columns."""

        if EXPLANATION_COLUMNS.issubset(trades.columns):
            return trades.copy()

        explained_trades = self._trade_explanation_generator(
            trades.copy()
        )

        if not isinstance(explained_trades, pd.DataFrame):
            raise TypeError(
                "Trade explanation generator must return "
                "a pandas DataFrame."
            )

        return explained_trades.copy()


def _validate_explained_trades(
    trades: pd.DataFrame,
) -> None:
    """Validate columns and values needed for portfolio explanations."""

    missing_columns = (
        REQUIRED_PACKAGE_COLUMNS
        | EXPLANATION_COLUMNS
    ) - set(trades.columns)

    if missing_columns:
        raise ValueError(
            "Explained trades are missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if trades["portfolio_id"].isna().any():
        raise ValueError("Portfolio IDs must not be missing.")

    if trades["action"].isna().any():
        raise ValueError("Actions must not be missing.")

    invalid_actions = ~trades["action"].isin(ALLOWED_ACTIONS)
    if invalid_actions.any():
        raise ValueError("Action must be one of BUY, SELL, or HOLD.")

    if trades["threshold_severity"].isna().any():
        raise ValueError("Threshold severity must not be missing.")

    invalid_severities = ~trades["threshold_severity"].isin(
        SEVERITY_RANK
    )
    if invalid_severities.any():
        raise ValueError(
            "Threshold severity must be one of none, medium, "
            "high, or critical."
        )

    _validate_threshold_flags(trades)
    _validate_money_columns(trades)
    _validate_explanation_columns(trades)


def _validate_threshold_flags(
    trades: pd.DataFrame,
) -> None:
    """Validate threshold-breached flags."""

    invalid_flags = trades["threshold_breached"].map(
        lambda value: not isinstance(value, (bool, np.bool_))
    )

    if invalid_flags.any():
        raise ValueError(
            "Threshold breached must contain Boolean values."
        )


def _validate_money_columns(
    trades: pd.DataFrame,
) -> None:
    """Validate already-computed cost and tax columns."""

    if trades[MONEY_COLUMNS].isna().any().any():
        raise ValueError(
            "Transaction costs and estimated taxes must not be missing."
        )

    money_values = trades[MONEY_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if money_values.isna().any().any():
        raise ValueError(
            "Transaction costs and estimated taxes must be valid numbers."
        )

    if not np.isfinite(money_values.to_numpy(dtype=float)).all():
        raise ValueError(
            "Transaction costs and estimated taxes must be finite."
        )

    if (money_values < 0.0).any().any():
        raise ValueError(
            "Transaction costs and estimated taxes must not be negative."
        )


def _validate_explanation_columns(
    trades: pd.DataFrame,
) -> None:
    """Validate reused or generated explanation text columns."""

    for column in sorted(EXPLANATION_COLUMNS):
        invalid_values = trades[column].map(
            lambda value: (
                not isinstance(value, str)
                or not value.strip()
            )
        )

        if invalid_values.any():
            raise ValueError(
                f"{column} must contain non-empty strings."
            )


def _build_portfolio_explanation(
    portfolio_id: object,
    portfolio_trades: pd.DataFrame,
) -> PortfolioExplanation:
    """Build a portfolio-level explanation package."""

    trade_count = len(portfolio_trades)
    buy_count = _count_action(portfolio_trades, "BUY")
    sell_count = _count_action(portfolio_trades, "SELL")
    hold_count = _count_action(portfolio_trades, "HOLD")
    total_transaction_cost = float(
        portfolio_trades["transaction_cost"].sum()
    )
    total_estimated_tax = float(
        portfolio_trades["estimated_tax_liability"].sum()
    )
    highest_severity = _highest_severity(
        portfolio_trades["threshold_severity"]
    )
    threshold_breach_count = int(
        portfolio_trades["threshold_breached"].sum()
    )

    return PortfolioExplanation(
        portfolio_id=portfolio_id,
        client_summary=_build_client_summary(
            trade_count=trade_count,
            buy_count=buy_count,
            sell_count=sell_count,
            threshold_breach_count=threshold_breach_count,
            total_transaction_cost=total_transaction_cost,
            total_estimated_tax=total_estimated_tax,
        ),
        advisor_summary=_build_advisor_summary(
            portfolio_id=portfolio_id,
            trade_count=trade_count,
            buy_count=buy_count,
            sell_count=sell_count,
            hold_count=hold_count,
            highest_severity=highest_severity,
            threshold_breach_count=threshold_breach_count,
            total_transaction_cost=total_transaction_cost,
            total_estimated_tax=total_estimated_tax,
        ),
        compliance_summary=_build_compliance_summary(),
        trade_count=trade_count,
        buy_count=buy_count,
        sell_count=sell_count,
        hold_count=hold_count,
        total_transaction_cost=total_transaction_cost,
        total_estimated_tax=total_estimated_tax,
        highest_threshold_severity=highest_severity,
        threshold_breach_count=threshold_breach_count,
    )


def _count_action(
    portfolio_trades: pd.DataFrame,
    action: str,
) -> int:
    """Count rows with a given existing trade action."""

    return int((portfolio_trades["action"] == action).sum())


def _highest_severity(
    severities: pd.Series,
) -> str:
    """Return the highest existing threshold severity."""

    return max(
        (str(severity) for severity in severities),
        key=SEVERITY_RANK.__getitem__,
    )


def _build_client_summary(
    trade_count: int,
    buy_count: int,
    sell_count: int,
    threshold_breach_count: int,
    total_transaction_cost: float,
    total_estimated_tax: float,
) -> str:
    """Build a client-facing portfolio summary."""

    if buy_count == 0 and sell_count == 0:
        return (
            "Your portfolio does not require trading. "
            "All recommended actions are holds."
        )

    threshold_phrase = (
        "has drifted away from its intended allocation"
        if threshold_breach_count > 0
        else "remains within its threshold controls"
    )

    return (
        f"Your portfolio {threshold_phrase}. "
        f"{_format_count(trade_count, 'trade')} are reviewed, "
        f"including {_format_count(buy_count, 'buy')} and "
        f"{_format_count(sell_count, 'sell')}. "
        "Estimated transaction costs are "
        f"{_format_currency(total_transaction_cost)}, with "
        f"estimated taxes of {_format_currency(total_estimated_tax)}."
    )


def _build_advisor_summary(
    portfolio_id: object,
    trade_count: int,
    buy_count: int,
    sell_count: int,
    hold_count: int,
    highest_severity: str,
    threshold_breach_count: int,
    total_transaction_cost: float,
    total_estimated_tax: float,
) -> str:
    """Build an advisor-facing portfolio summary."""

    return (
        f"Portfolio {portfolio_id} triggered a "
        f"{highest_severity}-severity threshold state. "
        f"{_format_count(trade_count, 'trade')} are included: "
        f"{buy_count} BUY, {sell_count} SELL, and {hold_count} HOLD. "
        f"Threshold breaches: {threshold_breach_count}. "
        "Estimated transaction costs are "
        f"{_format_currency(total_transaction_cost)} with "
        f"estimated taxes of {_format_currency(total_estimated_tax)}."
    )


def _build_compliance_summary() -> str:
    """Build a compliance-facing portfolio summary."""

    return (
        "Summary generated from deterministic optimizer outputs. "
        "No financial calculations were performed inside the "
        "Explanation Agent."
    )


def _format_count(
    count: int,
    singular_label: str,
) -> str:
    """Format a count with a singular or plural label."""

    label = singular_label if count == 1 else f"{singular_label}s"
    return f"{count} {label}"


def _format_currency(
    value: float,
) -> str:
    """Format a monetary value for portfolio-level communication."""

    return f"${value:,.2f}"
