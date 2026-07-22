import numpy as np
import pandas as pd


ZERO_TOLERANCE = 1e-9


def estimate_transaction_costs(
    trade_list: pd.DataFrame,
    portfolio_value: float,
    transaction_cost_rate: float = 0.002,
) -> pd.DataFrame:
    """
    Estimate transaction costs for each proposed trade.

    Trade values follow a signed convention: positive values represent
    buys, negative values represent sells, and zero values represent holds.
    Transaction costs are calculated from the absolute trade value.
    """

    required_columns = {
        "asset",
        "action",
        "trade_weight",
    }

    missing_columns = required_columns - set(
        trade_list.columns
    )

    if missing_columns:
        raise ValueError(
            "Trade list is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if portfolio_value <= 0:
        raise ValueError(
            "Portfolio value must be greater than zero."
        )

    if transaction_cost_rate < 0:
        raise ValueError(
            "Transaction cost rate cannot be negative."
        )

    result = trade_list.copy()
    result["trade_weight"] = _validate_trade_weights(result)

    result["trade_value"] = (
        result["trade_weight"]
        * portfolio_value
    )

    result["transaction_cost"] = (
        result["trade_value"].abs()
        * transaction_cost_rate
    )

    _validate_trade_conventions(result)

    return result


def _validate_trade_weights(
    trade_list: pd.DataFrame,
) -> pd.Series:
    """Validate and return numeric trade weights."""

    if trade_list["trade_weight"].isna().any():
        raise ValueError(
            "Trade weights must not contain missing values."
        )

    trade_weights = pd.to_numeric(
        trade_list["trade_weight"],
        errors="coerce",
    )

    if trade_weights.isna().any():
        raise ValueError(
            "Trade weights must be valid numbers."
        )

    if not np.isfinite(trade_weights.to_numpy()).all():
        raise ValueError(
            "Trade weights must be finite numeric values."
        )

    return trade_weights


def _validate_trade_conventions(
    trade_list: pd.DataFrame,
) -> None:
    """Validate trade action, weight, value, and cost conventions."""

    numeric_columns = [
        "trade_weight",
        "trade_value",
        "transaction_cost",
    ]

    if trade_list[numeric_columns].isna().any().any():
        raise ValueError(
            "Trade convention inputs must not contain missing values."
        )

    if not np.isfinite(
        trade_list[numeric_columns].to_numpy()
    ).all():
        raise ValueError(
            "Trade convention inputs must be finite numeric values."
        )

    buy_mask = trade_list["action"] == "BUY"
    sell_mask = trade_list["action"] == "SELL"
    hold_mask = trade_list["action"] == "HOLD"

    invalid_actions = ~(buy_mask | sell_mask | hold_mask)
    if invalid_actions.any():
        raise ValueError(
            "Action must be one of BUY, SELL, or HOLD."
        )

    invalid_buy = buy_mask & (
        (trade_list["trade_weight"] <= ZERO_TOLERANCE)
        | (trade_list["trade_value"] <= ZERO_TOLERANCE)
    )
    invalid_sell = sell_mask & (
        (trade_list["trade_weight"] >= -ZERO_TOLERANCE)
        | (trade_list["trade_value"] >= -ZERO_TOLERANCE)
    )
    invalid_hold = hold_mask & (
        (trade_list["trade_weight"].abs() > ZERO_TOLERANCE)
        | (trade_list["trade_value"].abs() > ZERO_TOLERANCE)
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

    if (trade_list["transaction_cost"] < 0).any():
        raise ValueError(
            "Transaction cost cannot be negative."
        )
