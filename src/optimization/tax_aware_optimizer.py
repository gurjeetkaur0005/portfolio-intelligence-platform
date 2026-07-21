from __future__ import annotations
import pandas as pd
REQUIRED_COLUMNS = {
    "portfolio_id",
    "asset_class",
    "trade_value",
    "current_value",
    "cost_basis",
    "tax_rate",
}
def estimate_trade_taxes(
    trade_list: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate the capital-gains tax created by proposed trades.

    Positive trade values represent buy trades.
    Negative trade values represent sell trades.

    Parameters
    ----------
    trade_list:
        Proposed portfolio trades with holding and tax information.

    Returns
    -------
    pd.DataFrame
        Trade list with additional tax-estimation columns.
    """

    
    missing_columns = REQUIRED_COLUMNS - set(trade_list.columns)

    if missing_columns:
        raise ValueError(
            "Trade list is missing required columns: "
            f"{sorted(missing_columns)}"
        )
    result = trade_list.copy()
    _validate_tax_inputs(result)
    result["unrealized_gain"] = (
            result["current_value"]
            - result["cost_basis"]
        )
    result["sell_value"] = (
        -result["trade_value"].clip(upper=0.0)
    )
    result["sell_fraction"] = (
        result["sell_value"]
        / result["current_value"]
    )
    result["estimated_realized_gain"] = (
        result["sell_fraction"]
        * result["unrealized_gain"]
    )
    result["estimated_tax_liability"] = (
        result["estimated_realized_gain"]
        .clip(lower=0.0)
        * result["tax_rate"]
    )
    result["creates_tax_liability"] = (
        result["estimated_tax_liability"] > 0
    )
    result["after_tax_trade_value"] = (
        result["sell_value"]
        - result["estimated_tax_liability"]
    )

    result["portfolio_estimated_tax"] = (
        result.groupby("portfolio_id")[
            "estimated_tax_liability"
        ].transform("sum")
    )


    return result

def _validate_tax_inputs(
    trade_list: pd.DataFrame,
) -> None:
    """
    Validate values required for tax estimation.
    """

    numeric_columns = [
        "trade_value",
        "current_value",
        "cost_basis",
        "tax_rate",
    ]

    if trade_list[numeric_columns].isna().any().any():
        raise ValueError(
            "Tax estimation inputs must not contain missing values."
        )
    if (trade_list["current_value"] <= 0).any():
        raise ValueError(
            "Current value must be greater than zero."
        )
    if (trade_list["cost_basis"] < 0).any():
        raise ValueError(
            "Cost basis must be greater than or equal to zero."
        )
    invalid_tax_rates = (
        (trade_list["tax_rate"] < 0)
        | (trade_list["tax_rate"] > 1)
    )
    if invalid_tax_rates.any():
        raise ValueError(
            "Tax rate must be between 0 and 1."
        )
    sell_values = -trade_list["trade_value"].clip(
        upper=0.0,
    )

    excessive_sales = (
        sell_values > trade_list["current_value"]
    )

    if excessive_sales.any():
        raise ValueError(
            "Sell trade value cannot exceed "
            "the current holding value."
        )