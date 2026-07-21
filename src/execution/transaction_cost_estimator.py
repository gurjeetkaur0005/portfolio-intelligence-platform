import pandas as pd

def estimate_transaction_costs(
    trade_list: pd.DataFrame,
    portfolio_value: float,
    transaction_cost_rate: float = 0.002,
) -> pd.DataFrame:
    """
    Estimate transaction costs for each trade.
    """
    required_columns = {
        "asset",
        "action",
        "trade_weight",
        }
    missing_columns = required_columns - set(trade_list.columns)
    if missing_columns:
        raise ValueError(
            f"Trade list is missing required columns: "
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
    result["trade_value"] = (
        result["trade_weight"].abs()
        * portfolio_value
    )
    result["transaction_cost"] = (
        result["trade_value"]
        * transaction_cost_rate
    )
    return result
