from __future__ import annotations

import numpy as np
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
    "threshold_breached",
    "threshold_severity",
    "breach_ratio",
}

NUMERIC_COLUMNS = [
    "current_weight",
    "trade_weight",
    "post_trade_weight",
    "trade_value",
    "transaction_cost",
    "estimated_tax_liability",
    "breach_ratio",
]

ALLOWED_ACTIONS = {
    "BUY",
    "SELL",
    "HOLD",
}

ALLOWED_THRESHOLD_SEVERITIES = {
    "none",
    "medium",
    "high",
    "critical",
}

ZERO_TOLERANCE = 1e-9


def generate_trade_explanations(
    trade_list: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate client, advisor, and compliance explanations for trades.

    The input trade list uses signed trade values and signed trade
    weights:

    - Positive values represent buys.
    - Negative values represent sells.
    - Zero values represent holds.

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

    if trade_list["threshold_breached"].isna().any():
        raise ValueError(
            "Threshold-breached values must not be missing."
        )

    if trade_list["threshold_severity"].isna().any():
        raise ValueError(
            "Threshold-severity values must not be missing."
        )

    invalid_actions = ~trade_list["action"].isin(ALLOWED_ACTIONS)

    if invalid_actions.any():
        raise ValueError(
            "Action must be one of BUY, SELL, or HOLD."
        )

    invalid_severities = ~trade_list["threshold_severity"].isin(
        ALLOWED_THRESHOLD_SEVERITIES
    )

    if invalid_severities.any():
        raise ValueError(
            "Threshold severity must be one of "
            "none, medium, high, or critical."
        )

    numeric_values = _validate_numeric_values(trade_list)
    _validate_threshold_values(
        trade_list=trade_list,
        numeric_values=numeric_values,
    )
    _validate_trade_conventions(
        trade_list=trade_list,
        numeric_values=numeric_values,
    )


def _validate_numeric_values(
    trade_list: pd.DataFrame,
) -> pd.DataFrame:
    """Validate that numeric explanation inputs are valid and finite."""

    if trade_list[NUMERIC_COLUMNS].isna().any().any():
        raise ValueError(
            "Explanation inputs must not contain missing numeric values."
        )

    numeric_values = trade_list[NUMERIC_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if numeric_values.isna().any().any():
        raise ValueError(
            "Explanation numeric inputs must be valid numbers."
        )

    if not np.isfinite(numeric_values.to_numpy()).all():
        raise ValueError(
            "Explanation numeric inputs must be finite values."
        )

    return numeric_values


def _validate_threshold_values(
    trade_list: pd.DataFrame,
    numeric_values: pd.DataFrame,
) -> None:
    """Validate threshold fields and their consistency."""

    invalid_threshold_flags = ~trade_list["threshold_breached"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    )

    if invalid_threshold_flags.any():
        raise ValueError(
            "Threshold breached must contain Boolean values."
        )

    breach_ratio = numeric_values["breach_ratio"]
    threshold_breached = trade_list["threshold_breached"]
    threshold_severity = trade_list["threshold_severity"]

    negative_breach_ratio = breach_ratio < 0

    if negative_breach_ratio.any():
        raise ValueError(
            "Breach ratio must not be negative."
        )

    expected_breached = breach_ratio > 1.0
    invalid_breached_rows = threshold_breached != expected_breached

    expected_severity = breach_ratio.apply(
        _threshold_severity_for_breach_ratio
    )
    invalid_severity_rows = threshold_severity != expected_severity

    if (
        invalid_breached_rows.any()
        or invalid_severity_rows.any()
    ):
        raise ValueError(
            "Threshold breached, severity, and breach ratio "
            "are inconsistent."
        )


def _validate_trade_conventions(
    trade_list: pd.DataFrame,
    numeric_values: pd.DataFrame,
) -> None:
    """Validate action, weight, and signed trade-value consistency."""

    buy_mask = trade_list["action"] == "BUY"
    sell_mask = trade_list["action"] == "SELL"
    hold_mask = trade_list["action"] == "HOLD"

    invalid_buy = buy_mask & (
        (numeric_values["trade_weight"] <= ZERO_TOLERANCE)
        | (numeric_values["trade_value"] <= ZERO_TOLERANCE)
    )

    invalid_sell = sell_mask & (
        (numeric_values["trade_weight"] >= -ZERO_TOLERANCE)
        | (numeric_values["trade_value"] >= -ZERO_TOLERANCE)
    )

    invalid_hold = hold_mask & (
        (numeric_values["trade_weight"].abs() > ZERO_TOLERANCE)
        | (numeric_values["trade_value"].abs() > ZERO_TOLERANCE)
    )

    if (
        invalid_buy.any()
        or invalid_sell.any()
        or invalid_hold.any()
    ):
        raise ValueError(
            "Trade action is inconsistent with trade weight "
            "and trade value."
        )


def _threshold_severity_for_breach_ratio(
    breach_ratio: float,
) -> str:
    """Return expected threshold severity for a breach ratio."""

    if breach_ratio <= 1.0:
        return "none"

    if breach_ratio < 1.5:
        return "medium"

    if breach_ratio < 2.0:
        return "high"

    return "critical"


def _format_asset_name(
    asset: str,
) -> str:
    """Convert an internal asset name into readable text."""

    return asset.replace("_", " ")


def _format_weight(
    weight: float,
) -> str:
    """Format a raw portfolio weight as a percentage."""

    return f"{float(weight):.2%}"


def _format_currency(
    value: float,
) -> str:
    """Format a currency value."""

    return f"${float(value):,.2f}"


def _format_signed_currency(
    value: float,
) -> str:
    """Format a signed currency value with a clear leading sign."""

    numeric_value = float(value)

    if numeric_value < 0:
        return f"-${abs(numeric_value):,.2f}"

    return f"${numeric_value:,.2f}"


def _allocation_change_phrase(
    action: str,
) -> str:
    """Return a plain-language allocation direction."""

    if action == "BUY":
        return "increasing"

    if action == "SELL":
        return "decreasing"

    return "remaining unchanged"


def _build_client_threshold_phrase(
    row: pd.Series,
) -> str:
    """Build a simple threshold reason for the client."""

    if not bool(row["threshold_breached"]):
        return (
            "The allocation remains within its permitted range. "
        )

    return (
        "This adjustment is recommended because the portfolio "
        "allocation moved outside its permitted range. "
    )


def _build_advisor_threshold_phrase(
    row: pd.Series,
) -> str:
    """Build a detailed threshold explanation for an advisor."""

    if not bool(row["threshold_breached"]):
        return (
            "No allocation threshold was breached. "
            f"The breach ratio is {row['breach_ratio']:.2f}. "
        )

    severity = str(row["threshold_severity"])
    breach_ratio = float(row["breach_ratio"])

    return (
        "The allocation threshold was breached with "
        f"{severity} severity and a breach ratio of "
        f"{breach_ratio:.2f}. "
    )


def _build_client_explanation(
    row: pd.Series,
) -> str:
    """Build a simple explanation for a client."""

    asset = _format_asset_name(str(row["asset"]))
    action = str(row["action"])
    threshold_reason = _build_client_threshold_phrase(row)

    if action == "HOLD":
        allocation_phrase = (
            f"Your investment in {asset} remains unchanged at "
            f"{_format_weight(row['current_weight'])}. "
        )
    else:
        allocation_change = _allocation_change_phrase(action)
        allocation_phrase = (
            f"Your investment in {asset} is {allocation_change}, "
            f"changing from {_format_weight(row['current_weight'])} "
            f"to {_format_weight(row['post_trade_weight'])}. "
        )

    return (
        f"{allocation_phrase}"
        f"{threshold_reason}"
        f"The estimated transaction cost is "
        f"{_format_currency(row['transaction_cost'])}, "
        f"and the estimated tax liability is "
        f"{_format_currency(row['estimated_tax_liability'])}."
    )


def _build_advisor_explanation(
    row: pd.Series,
) -> str:
    """Build a detailed explanation for an advisor."""

    asset = _format_asset_name(str(row["asset"]))
    threshold_reason = _build_advisor_threshold_phrase(row)

    return (
        f"Portfolio {row['portfolio_id']}: "
        f"Recommend {row['action']} for {asset}. "
        f"{threshold_reason}"
        f"Adjust the allocation from "
        f"{_format_weight(row['current_weight'])} "
        f"to {_format_weight(row['post_trade_weight'])} "
        f"through a trade of "
        f"{_format_weight(row['trade_weight'])}. "
        f"Estimated signed trade value: "
        f"{_format_signed_currency(row['trade_value'])}. "
        f"Estimated transaction cost: "
        f"{_format_currency(row['transaction_cost'])}. "
        f"Estimated tax liability: "
        f"{_format_currency(row['estimated_tax_liability'])}."
    )


def _build_compliance_explanation(
    row: pd.Series,
) -> str:
    """Build a structured, traceable compliance explanation."""

    asset = _format_asset_name(str(row["asset"]))

    return (
        "Trade recommendation generated by the portfolio optimizer. "
        f"Portfolio ID: {row['portfolio_id']}. "
        f"Asset: {asset}. "
        f"Recommended action: {row['action']}. "
        f"Threshold breached: {bool(row['threshold_breached'])}. "
        f"Threshold severity: {row['threshold_severity']}. "
        f"Breach ratio: {float(row['breach_ratio']):.6f}. "
        f"Current portfolio weight: "
        f"{float(row['current_weight']):.6f}. "
        f"Recommended trade weight: "
        f"{float(row['trade_weight']):.6f}. "
        f"Expected post-trade weight: "
        f"{float(row['post_trade_weight']):.6f}. "
        f"Signed trade value: "
        f"{_format_signed_currency(row['trade_value'])}. "
        f"Estimated transaction cost: "
        f"{_format_currency(row['transaction_cost'])}. "
        f"Estimated tax liability: "
        f"{_format_currency(row['estimated_tax_liability'])}."
    )
