from __future__ import annotations

from typing import Any

import pandas as pd

from streamlit_app.components.charts import _format_asset_label
from streamlit_app.components.status import drift_label


def prepare_holdings_table_data(
    holdings: list[dict[str, Any]],
) -> pd.DataFrame:
    """Return display-ready holding rows."""

    dataframe = pd.DataFrame(holdings)

    preferred_columns = [
        "asset",
        "current_weight",
        "target_weight",
        "drift",
        "current_value",
        "cost_basis",
    ]
    visible_columns = [
        column
        for column in preferred_columns
        if column in dataframe.columns
    ]

    if not visible_columns:
        return pd.DataFrame()

    result = dataframe[visible_columns].copy()

    if "asset" in result.columns:
        result["asset"] = result["asset"].astype(str).map(
            _format_asset_label
        )

    if "current_weight" in result.columns:
        result["current_weight"] = pd.to_numeric(
            result["current_weight"],
            errors="coerce",
        ) * 100

    if "target_weight" in result.columns:
        result["target_weight"] = pd.to_numeric(
            result["target_weight"],
            errors="coerce",
        ) * 100

    if "drift" in result.columns:
        result["drift"] = result["drift"].map(drift_label)

    for column in (
        "current_value",
        "cost_basis",
    ):
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    return result


def prepare_portfolio_table_data(
    portfolios: list[dict[str, Any]],
) -> pd.DataFrame:
    """Return display-ready portfolio rows."""

    dataframe = pd.DataFrame(portfolios)

    preferred_columns = [
        "portfolio_id",
        "client_id",
        "portfolio_value",
        "currency",
    ]
    visible_columns = [
        column
        for column in preferred_columns
        if column in dataframe.columns
    ]

    if not visible_columns:
        return pd.DataFrame()

    result = dataframe[visible_columns].copy()

    if "portfolio_value" in result.columns:
        result["portfolio_value"] = pd.to_numeric(
            result["portfolio_value"],
            errors="coerce",
        )

    return result


def prepare_rebalance_trade_table_data(
    trades: list[dict[str, Any]],
) -> pd.DataFrame:
    """Return display-ready rebalance trade rows."""

    dataframe = pd.DataFrame(trades)

    preferred_columns = [
        "asset",
        "action",
        "current_weight",
        "trade_weight",
        "post_trade_weight",
        "trade_value",
        "estimated_transaction_cost",
        "estimated_tax",
        "threshold_severity",
        "final_trigger_type",
    ]
    visible_columns = [
        column
        for column in preferred_columns
        if column in dataframe.columns
    ]

    if not visible_columns:
        return pd.DataFrame()

    result = dataframe[visible_columns].copy()

    if "asset" in result.columns:
        result["asset"] = result["asset"].astype(str).map(
            _format_asset_label
        )

    for column in (
        "current_weight",
        "trade_weight",
        "post_trade_weight",
    ):
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            ) * 100

    for column in (
        "trade_value",
        "estimated_transaction_cost",
        "estimated_tax",
    ):
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    return result


def prepare_rebalance_audit_table_data(
    audit_records: list[dict[str, Any]],
) -> pd.DataFrame:
    """Return display-ready rebalance audit rows."""

    dataframe = pd.DataFrame(audit_records)

    preferred_columns = [
        "timestamp",
        "event_type",
        "asset",
        "action",
        "approval_status",
        "audit_message",
        "audit_id",
        "approval_reason",
        "reviewed_by",
        "reviewed_at",
    ]
    visible_columns = [
        column
        for column in preferred_columns
        if column in dataframe.columns
    ]

    if not visible_columns:
        return pd.DataFrame()

    result = dataframe[visible_columns].copy()

    if "asset" in result.columns:
        result["asset"] = result["asset"].astype(str).map(
            _format_asset_label
        )

    return result


def prepare_rebalance_run_table_data(
    runs: list[dict[str, Any]],
) -> pd.DataFrame:
    """Return display-ready historical rebalance run rows."""

    dataframe = pd.DataFrame(runs)

    preferred_columns = [
        "run_id",
        "status",
        "created_at",
        "portfolio_value",
        "transaction_cost",
    ]
    visible_columns = [
        column
        for column in preferred_columns
        if column in dataframe.columns
    ]

    if not visible_columns:
        return pd.DataFrame()

    result = dataframe[visible_columns].copy()

    for column in (
        "portfolio_value",
        "transaction_cost",
    ):
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    return result


def prepare_strategy_comparison_table_data(
    comparison: dict[str, Any],
) -> pd.DataFrame:
    """Return display-ready strategy comparison rows."""

    from streamlit_app.services.display import (
        display_currency,
        display_label,
        display_number,
        display_percentage,
    )

    buy_and_hold = comparison.get("buy_and_hold")
    threshold = comparison.get("threshold_rebalancing")

    if not isinstance(buy_and_hold, dict) or not isinstance(
        threshold,
        dict,
    ):
        return pd.DataFrame()

    metric_formatters = {
        "total_return": display_percentage,
        "annualized_return": display_percentage,
        "volatility": display_percentage,
        "maximum_drawdown": display_percentage,
        "sharpe_ratio": display_number,
        "transaction_costs": display_currency,
        "taxes_paid": display_currency,
        "total_implementation_cost": display_currency,
        "number_of_rebalances": _format_table_count,
    }
    rows: list[dict[str, str]] = []

    for metric_name, formatter in metric_formatters.items():
        rows.append(
            {
                "Metric": display_label(metric_name),
                "Buy & Hold": formatter(
                    buy_and_hold.get(metric_name)
                ),
                "Threshold Rebalancing": formatter(
                    threshold.get(metric_name)
                ),
            }
        )

    return pd.DataFrame(rows)


def _format_table_count(
    value: Any,
) -> str:
    """Format a table count value."""

    if isinstance(value, bool) or not isinstance(value, int):
        return "Not available"

    return f"{value:,}"


def render_portfolio_table(
    portfolios: list[dict[str, Any]],
) -> None:
    """Render a table containing stored portfolios."""

    import streamlit as st

    dataframe = prepare_portfolio_table_data(portfolios)

    if dataframe.empty:
        st.info("No portfolios were returned by the backend.")
        return

    st.dataframe(
        dataframe,
        width="stretch",
        hide_index=True,
        column_config={
            "portfolio_id": st.column_config.TextColumn(
                "Portfolio ID",
            ),
            "client_id": st.column_config.TextColumn(
                "Client ID",
            ),
            "portfolio_value": st.column_config.NumberColumn(
                "Portfolio Value",
                format="$%.2f",
            ),
            "currency": st.column_config.TextColumn(
                "Currency",
            ),
        },
    )


def render_holdings_table(
    holdings: list[dict[str, Any]],
) -> None:
    """Render portfolio holdings in a user-friendly table."""

    import streamlit as st

    dataframe = prepare_holdings_table_data(holdings)

    if dataframe.empty:
        st.info("No holdings were returned by the backend.")
        return

    st.dataframe(
        dataframe,
        width="stretch",
        hide_index=True,
        column_config={
            "asset": st.column_config.TextColumn(
                "Asset Class",
            ),
            "current_weight": st.column_config.NumberColumn(
                "Current Weight",
                format="%.2f%%",
            ),
            "target_weight": st.column_config.NumberColumn(
                "Target Weight",
                format="%.2f%%",
            ),
            "drift": st.column_config.TextColumn(
                "Drift",
            ),
            "current_value": st.column_config.NumberColumn(
                "Current Value",
                format="$%.2f",
            ),
            "cost_basis": st.column_config.NumberColumn(
                "Cost Basis",
                format="$%.2f",
            ),
        },
    )


def render_rebalance_trade_table(
    trades: list[dict[str, Any]],
) -> None:
    """Render the main rebalance trade recommendation table."""

    import streamlit as st

    dataframe = prepare_rebalance_trade_table_data(trades)

    if dataframe.empty:
        st.info("No trades were returned for this rebalance run.")
        return

    st.dataframe(
        dataframe,
        width="stretch",
        hide_index=True,
        column_config={
            "asset": st.column_config.TextColumn(
                "Asset",
            ),
            "action": st.column_config.TextColumn(
                "Action",
            ),
            "current_weight": st.column_config.NumberColumn(
                "Current Weight",
                format="%.2f%%",
            ),
            "trade_weight": st.column_config.NumberColumn(
                "Trade Weight",
                format="%.2f%%",
            ),
            "post_trade_weight": st.column_config.NumberColumn(
                "Post-Trade Weight",
                format="%.2f%%",
            ),
            "trade_value": st.column_config.NumberColumn(
                "Trade Value",
                format="$%.2f",
            ),
            "estimated_transaction_cost": (
                st.column_config.NumberColumn(
                    "Transaction Cost",
                    format="$%.2f",
                )
            ),
            "estimated_tax": st.column_config.NumberColumn(
                "Estimated Tax",
                format="$%.2f",
            ),
            "threshold_severity": st.column_config.TextColumn(
                "Threshold Severity",
            ),
            "final_trigger_type": st.column_config.TextColumn(
                "Final Trigger",
            ),
        },
    )


def render_rebalance_audit_table(
    audit_records: list[dict[str, Any]],
) -> None:
    """Render the rebalance audit trail table."""

    import streamlit as st

    dataframe = prepare_rebalance_audit_table_data(audit_records)

    if dataframe.empty:
        st.info(
            "No audit records were returned for this rebalance run."
        )
        return

    st.dataframe(
        dataframe,
        width="stretch",
        hide_index=True,
        column_config={
            "timestamp": st.column_config.TextColumn(
                "Timestamp",
            ),
            "event_type": st.column_config.TextColumn(
                "Event Type",
            ),
            "asset": st.column_config.TextColumn(
                "Asset",
            ),
            "action": st.column_config.TextColumn(
                "Action",
            ),
            "approval_status": st.column_config.TextColumn(
                "Approval Status",
            ),
            "audit_message": st.column_config.TextColumn(
                "Audit Message",
            ),
            "audit_id": st.column_config.TextColumn(
                "Audit ID",
            ),
            "approval_reason": st.column_config.TextColumn(
                "Approval Reason",
            ),
            "reviewed_by": st.column_config.TextColumn(
                "Reviewed By",
            ),
            "reviewed_at": st.column_config.TextColumn(
                "Reviewed At",
            ),
        },
    )


def render_rebalance_run_table(
    runs: list[dict[str, Any]],
) -> None:
    """Render historical rebalance runs."""

    import streamlit as st

    dataframe = prepare_rebalance_run_table_data(runs)

    if dataframe.empty:
        st.info("No rebalance runs exist for this portfolio.")
        return

    st.dataframe(
        dataframe,
        width="stretch",
        hide_index=True,
        column_config={
            "run_id": st.column_config.TextColumn(
                "Run ID",
            ),
            "status": st.column_config.TextColumn(
                "Status",
            ),
            "created_at": st.column_config.TextColumn(
                "Created",
            ),
            "portfolio_value": st.column_config.NumberColumn(
                "Portfolio Value",
                format="$%.2f",
            ),
            "transaction_cost": st.column_config.NumberColumn(
                "Transaction Cost",
                format="$%.2f",
            ),
        },
    )


def render_strategy_comparison_table(
    comparison: dict[str, Any],
) -> None:
    """Render a side-by-side strategy comparison table."""

    import streamlit as st

    dataframe = prepare_strategy_comparison_table_data(comparison)

    if dataframe.empty:
        st.warning("Strategy comparison data is unavailable.")
        return

    st.dataframe(
        dataframe,
        width="stretch",
        hide_index=True,
    )
