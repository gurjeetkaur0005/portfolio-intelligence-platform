from __future__ import annotations

import math

import pandas as pd


REQUIRED_COLUMNS = {
    "portfolio_id",
    "asset",
    "action",
    "current_weight",
    "trade_weight",
    "post_trade_weight",
    "trade_value",
    "transaction_cost",
    "estimated_tax_liability",
}

NUMERIC_COLUMNS = [
    "current_weight",
    "trade_weight",
    "post_trade_weight",
    "trade_value",
    "transaction_cost",
    "estimated_tax_liability",
]

ALLOWED_ACTIONS = {
    "BUY",
    "SELL",
    "HOLD",
}


def generate_trade_explanations(
    trade_list: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate client, advisor, and compliance explanations for trades.

    The input trade list uses a signed trade-value convention: positive
    values are buys, negative values are sells, and zero values are holds.
    The input DataFrame is not mutated.
    """

    _validate_explanation_inputs(trade_list)

    result = trade_list.copy()

    result["client_explanation"] = result.apply(
        _build_client_explanation,
        axis=1,
    )
    result["advisor_explanation"] = result.apply(
        _build_advisor_explanation,
        axis=1,
    )
    result["compliance_explanation"] = result.apply(
        _build_compliance_explanation,
        axis=1,
    )

    return result


def _validate_explanation_inputs(
    trade_list: pd.DataFrame,
) -> None:
    """Validate columns and values required for explanations."""

    missing_columns = REQUIRED_COLUMNS - set(trade_list.columns)

    if missing_columns:
        raise ValueError(
            "Trade list is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if trade_list[NUMERIC_COLUMNS].isna().any().any():
        raise ValueError(
            "Explanation inputs must not contain missing numeric values."
        )

    invalid_actions = ~trade_list["action"].isin(ALLOWED_ACTIONS)
    if invalid_actions.any():
        raise ValueError(
            "Action must be one of BUY, SELL, or HOLD."
        )

    for column in [
        "current_weight",
        "trade_weight",
        "post_trade_weight",
    ]:
        non_finite_values = trade_list[column].apply(
            lambda value: not math.isfinite(float(value))
        )
        if non_finite_values.any():
            raise ValueError(
                "Weights must be finite numeric values."
            )


def _format_asset_name(
    asset: str,
) -> str:
    """Convert an internal asset name into readable text."""

    return asset.replace("_", " ")


def _format_weight(
    weight: float,
) -> str:
    """Format a raw weight as a percentage."""

    return f"{weight:.2%}"


def _allocation_change_phrase(
    action: str,
) -> str:
    """Return plain-language allocation direction for an action."""

    if action == "BUY":
        return "increasing"

    if action == "SELL":
        return "decreasing"

    return "unchanged"


def _build_client_explanation(
    row: pd.Series,
) -> str:
    """Build a simple explanation for a client."""

    asset = _format_asset_name(str(row["asset"]))
    action = str(row["action"])
    allocation_change = _allocation_change_phrase(action)

    return (
        f"Your allocation to {asset} is {allocation_change}, "
        f"moving from {_format_weight(row['current_weight'])} "
        f"to {_format_weight(row['post_trade_weight'])}. "
        f"Estimated transaction cost is "
        f"${row['transaction_cost']:,.2f}, and estimated tax is "
        f"${row['estimated_tax_liability']:,.2f}."
    )


def _build_advisor_explanation(
    row: pd.Series,
) -> str:
    """Build a detailed explanation for an advisor."""

    asset = _format_asset_name(str(row["asset"]))

    return (
        f"Portfolio {row['portfolio_id']} | Asset: {asset} | "
        f"Action: {row['action']} | "
        f"Current weight: {_format_weight(row['current_weight'])} | "
        f"Trade weight: {_format_weight(row['trade_weight'])} | "
        f"Post-trade weight: {_format_weight(row['post_trade_weight'])} | "
        f"Signed trade value: ${row['trade_value']:,.2f} | "
        f"Transaction cost: ${row['transaction_cost']:,.2f} | "
        f"Tax liability: ${row['estimated_tax_liability']:,.2f}."
    )


def _build_compliance_explanation(
    row: pd.Series,
) -> str:
    """Build a structured, traceable explanation for compliance review."""

    asset = _format_asset_name(str(row["asset"]))

    return (
        f"Recommendation source: portfolio optimizer. "
        f"Portfolio ID: {row['portfolio_id']}. "
        f"Asset: {asset}. "
        f"Action: {row['action']}. "
        f"Current weight: {row['current_weight']:.6f}. "
        f"Trade weight: {row['trade_weight']:.6f}. "
        f"Post-trade weight: {row['post_trade_weight']:.6f}. "
        f"Signed trade value: {row['trade_value']:.2f}. "
        f"Transaction cost: {row['transaction_cost']:.2f}. "
        f"Estimated tax liability: "
        f"{row['estimated_tax_liability']:.2f}."
    )
