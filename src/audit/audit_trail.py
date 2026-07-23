from __future__ import annotations

from datetime import datetime

import pandas as pd

REQUIRED_COLUMNS = {
    "portfolio_id",
    "asset",
    "action",
    "trade_value",
    "transaction_cost",
    "estimated_tax_liability",
    "final_trigger_type",
    "final_priority",
    "approval_required",
    "approval_status",
    "approval_reason",
}

AUDIT_ID_PREFIX = "AUD"

def create_audit_trail(
    trades: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create audit records for every trade.

    The input DataFrame is not mutated.
    """

    _validate_inputs(trades)

    result = trades.copy()

    result.insert(
        0,
        "audit_id",
        _generate_audit_ids(len(result)),
    )

    result.insert(
        1,
        "audit_timestamp",
        datetime.now(),
    )

    return result

def _validate_inputs(
    trades: pd.DataFrame,
) -> None:
    """
    Validate the trade DataFrame.
    """

    if not isinstance(trades, pd.DataFrame):
        raise TypeError(
            "Trades must be provided as a pandas DataFrame."
        )

    missing_columns = REQUIRED_COLUMNS.difference(
        trades.columns,
    )

    if missing_columns:
        missing_column_list = ", ".join(
            sorted(missing_columns),
        )

        raise ValueError(
            "Trades are missing required columns: "
            f"{missing_column_list}"
        )
    
def _generate_audit_ids(
    number_of_records: int,
) -> list[str]:
    """
    Generate sequential audit identifiers.
    """

    return [
        f"{AUDIT_ID_PREFIX}{index:06d}"
        for index in range(
            1,
            number_of_records + 1,
        )
    ]