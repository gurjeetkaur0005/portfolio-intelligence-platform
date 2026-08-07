from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, cast

import pandas as pd

from config.asset_classes import ASSET_CLASSES
from config.risk_categories import RISK_CATEGORIES
from src.database.models import (
    ClientModel,
    PortfolioHoldingModel,
    PortfolioModel,
)


DEFAULT_TAX_BRACKET = 0.20
DEFAULT_AGE = 40
DEFAULT_INVESTMENT_HORIZON_YEARS = 10
DEFAULT_ANNUAL_INCOME = 0.0
DEFAULT_MONTHLY_SIP = 0.0
DEFAULT_PLANNED_WITHDRAWAL = 0.0
DEFAULT_RESTRICTED_SECURITY_COUNT = 0
DEFAULT_ESG_PREFERENCE = False
DEFAULT_PRIOR_APPROVAL_REQUIRED = False
REQUIRED_WEIGHT_TOTAL = Decimal("1.0")
WEIGHT_SUM_TOLERANCE = Decimal("0.000001")


@dataclass(frozen=True, slots=True)
class DeterministicPortfolioInput:
    """Store deterministic pipeline input frames."""

    client_profiles: pd.DataFrame
    portfolios: pd.DataFrame


class PortfolioInputAdapter:
    """
    Convert persisted ORM portfolio data into pipeline inputs.

    The adapter performs shape translation only. It does not calculate
    drift, triggers, trades, transaction costs, taxes, or explanations.
    """

    def build_input(
        self,
        portfolio: PortfolioModel,
    ) -> DeterministicPortfolioInput:
        """Return deterministic pipeline inputs for one portfolio."""

        _validate_portfolio(portfolio)

        client = portfolio.client

        return DeterministicPortfolioInput(
            client_profiles=_build_client_profiles(
                portfolio=portfolio,
                client=client,
            ),
            portfolios=_build_portfolios(portfolio),
        )


def _build_client_profiles(
    *,
    portfolio: PortfolioModel,
    client: ClientModel,
) -> pd.DataFrame:
    """Build the client profile row expected by the pipeline."""

    return pd.DataFrame(
        [
            {
                "client_id": client.client_id,
                "portfolio_id": portfolio.portfolio_id,
                "risk_category": client.risk_category,
                "age": DEFAULT_AGE,
                "investment_horizon_years": (
                    DEFAULT_INVESTMENT_HORIZON_YEARS
                ),
                "annual_income": DEFAULT_ANNUAL_INCOME,
                "tax_bracket": DEFAULT_TAX_BRACKET,
                "monthly_sip": DEFAULT_MONTHLY_SIP,
                "planned_withdrawal": (
                    DEFAULT_PLANNED_WITHDRAWAL
                ),
                "restricted_security_count": (
                    DEFAULT_RESTRICTED_SECURITY_COUNT
                ),
                "esg_preference": DEFAULT_ESG_PREFERENCE,
                "prior_approval_required": (
                    DEFAULT_PRIOR_APPROVAL_REQUIRED
                ),
            }
        ]
    )


def _build_portfolios(
    portfolio: PortfolioModel,
) -> pd.DataFrame:
    """Build the portfolio row expected by the pipeline."""

    category_config = _risk_category_config(
        portfolio.client.risk_category
    )
    target = _target_weights(category_config)
    drift_band = float(
        cast(float, category_config["drift_band"])
    )
    holdings_by_asset = _holdings_by_asset(
        portfolio.holdings
    )

    row: dict[str, object] = {
        "portfolio_id": portfolio.portfolio_id,
        "risk_category": portfolio.client.risk_category,
        "drift_band": drift_band,
    }

    for index, asset in enumerate(ASSET_CLASSES):
        holding = holdings_by_asset.get(asset)

        row[f"target_{asset}"] = float(target[index])
        row[f"current_{asset}"] = (
            float(holding.current_weight)
            if holding is not None
            else 0.0
        )
        row[f"current_value_{asset}"] = (
            float(holding.current_value)
            if holding is not None
            else 0.0
        )
        row[f"cost_basis_{asset}"] = (
            float(holding.cost_basis)
            if holding is not None
            else 0.0
        )

    return pd.DataFrame([row])


def _validate_portfolio(
    portfolio: PortfolioModel,
) -> None:
    """Validate the ORM graph needed for adaptation."""

    if not isinstance(portfolio, PortfolioModel):
        raise TypeError(
            "portfolio must be a PortfolioModel."
        )

    if not isinstance(portfolio.client, ClientModel):
        raise ValueError(
            "portfolio must include a client."
        )

    _risk_category_config(portfolio.client.risk_category)
    _holdings_by_asset(portfolio.holdings)


def _risk_category_config(
    risk_category: str,
) -> dict[str, Any]:
    """Return configured risk-category data."""

    if risk_category not in RISK_CATEGORIES:
        raise ValueError(
            f"Unsupported risk category: {risk_category!r}."
        )

    return cast(
        dict[str, Any],
        RISK_CATEGORIES[risk_category],
    )


def _target_weights(
    category_config: dict[str, Any],
) -> list[float]:
    """Return configured target weights."""

    target = category_config["target"]

    if not isinstance(target, list):
        raise ValueError(
            "Risk category target allocation must be a list."
        )

    if len(target) != len(ASSET_CLASSES):
        raise ValueError(
            "Risk category target allocation must match assets."
        )

    return [float(value) for value in target]


def _holdings_by_asset(
    holdings: list[PortfolioHoldingModel],
) -> dict[str, PortfolioHoldingModel]:
    """Return holdings keyed by asset name."""

    result: dict[str, PortfolioHoldingModel] = {}

    valid_assets = set(ASSET_CLASSES)
    weight_total = Decimal("0")

    for holding in holdings:
        if not isinstance(holding, PortfolioHoldingModel):
            raise TypeError(
                "holdings must contain PortfolioHoldingModel objects."
            )

        if holding.asset not in valid_assets:
            raise ValueError(
                f"Unsupported holding asset: {holding.asset!r}."
            )

        if holding.asset in result:
            raise ValueError(
                f"Duplicate holding asset: {holding.asset!r}."
            )

        _validate_decimal(
            holding.current_weight,
            "current_weight",
        )
        if holding.current_weight < 0:
            raise ValueError(
                "current_weight must be non-negative."
            )

        _validate_decimal(
            holding.current_value,
            "current_value",
        )
        if holding.current_value < 0:
            raise ValueError(
                "current_value must be non-negative."
            )

        _validate_decimal(
            holding.cost_basis,
            "cost_basis",
        )
        if holding.cost_basis < 0:
            raise ValueError(
                "cost_basis must be non-negative."
            )

        result[holding.asset] = holding
        weight_total += holding.current_weight

    missing_assets = valid_assets - set(result)

    if missing_assets:
        formatted_assets = ", ".join(
            sorted(missing_assets)
        )
        raise ValueError(
            "Portfolio holdings must include all required "
            f"asset classes. Missing: {formatted_assets}."
        )

    excess_assets = set(result) - valid_assets

    if excess_assets:
        formatted_assets = ", ".join(
            sorted(excess_assets)
        )
        raise ValueError(
            "Portfolio holdings contain unsupported asset "
            f"classes: {formatted_assets}."
        )

    if abs(weight_total - REQUIRED_WEIGHT_TOTAL) > WEIGHT_SUM_TOLERANCE:
        raise ValueError(
            "Portfolio holding weights must sum to 1.0."
        )

    return result


def _validate_decimal(
    value: Decimal,
    field_name: str,
) -> None:
    """Validate finite Decimal holding values."""

    if not isinstance(value, Decimal):
        raise TypeError(
            f"{field_name} must be a Decimal."
        )

    if not value.is_finite():
        raise ValueError(
            f"{field_name} must be finite."
        )
