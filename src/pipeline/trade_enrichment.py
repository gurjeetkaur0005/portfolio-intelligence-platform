from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "asset",
    "current_weight",
}


def enrich_trade_data(
    trade_list: pd.DataFrame,
    portfolio_id: str,
    portfolio_value: float,
    tax_rate: float,
) -> pd.DataFrame:
    """
    Add holding and tax fields required by tax-aware optimization.

    This helper intentionally uses a synthetic cost-basis approximation
    until tax-lot data is introduced.
    """

    _validate_enrichment_inputs(
        trade_list=trade_list,
        portfolio_value=portfolio_value,
        tax_rate=tax_rate,
    )

    result = trade_list.copy()
    result["current_weight"] = _validate_current_weights(result)

    result["portfolio_id"] = portfolio_id
    result["current_value"] = (
        result["current_weight"] * portfolio_value
    )
    result["cost_basis"] = result["current_value"] * 0.90
    result["tax_rate"] = tax_rate

    return result


def _validate_enrichment_inputs(
    trade_list: pd.DataFrame,
    portfolio_value: float,
    tax_rate: float,
) -> None:
    """Validate inputs required for trade enrichment."""

    missing_columns = REQUIRED_COLUMNS - set(trade_list.columns)
    if missing_columns:
        raise ValueError(
            "Trade list is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if portfolio_value <= 0:
        raise ValueError(
            "Portfolio value must be greater than zero."
        )

    if not 0 <= tax_rate <= 1:
        raise ValueError(
            "Tax rate must be between 0 and 1."
        )

    _validate_current_weights(trade_list)


def _validate_current_weights(
    trade_list: pd.DataFrame,
) -> pd.Series:
    """Validate and return numeric current weights."""

    if trade_list["current_weight"].isna().any():
        raise ValueError(
            "Current weights must not contain missing values."
        )

    current_weights = pd.to_numeric(
        trade_list["current_weight"],
        errors="coerce",
    )

    if current_weights.isna().any():
        raise ValueError(
            "Current weights must be valid numbers."
        )

    if not np.isfinite(current_weights.to_numpy()).all():
        raise ValueError(
            "Current weights must be finite numeric values."
        )

    return current_weights

