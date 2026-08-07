from __future__ import annotations

import pandas as pd
import streamlit as st

from streamlit_app.components.metrics import format_currency
from streamlit_app.components.navigation import render_sidebar
from streamlit_app.config import get_settings
from streamlit_app.services.api_client import (
    ApiClientError,
    FastApiClient,
    JsonObject,
    PaginatedResponse,
)
from streamlit_app.services.display import display_timestamp


def _build_client() -> FastApiClient:
    """Create the reusable FastAPI client."""

    settings = get_settings()

    return FastApiClient(
        base_url=settings.api_base_url,
        timeout_seconds=settings.api_timeout_seconds,
    )


def _extract_portfolio_ids(
    portfolio_page: PaginatedResponse,
) -> list[str]:
    """Extract valid portfolio identifiers."""

    portfolio_ids: list[str] = []

    for item in portfolio_page.items:
        portfolio_id = item.get("portfolio_id")

        if isinstance(portfolio_id, str) and portfolio_id.strip():
            portfolio_ids.append(portfolio_id)

    return portfolio_ids


def _extract_run_ids(
    rebalance_page: PaginatedResponse,
) -> list[str]:
    """Extract valid rebalance run identifiers."""

    run_ids: list[str] = []

    for item in rebalance_page.items:
        run_id = item.get("run_id")

        if isinstance(run_id, str) and run_id.strip():
            run_ids.append(run_id)

    return run_ids


def _render_run_history(
    rebalance_page: PaginatedResponse,
) -> None:
    """Render the portfolio's rebalance history."""

    st.subheader("Rebalance Runs")

    if not rebalance_page.items:
        st.info("No rebalance runs exist for this portfolio.")
        return

    dataframe = pd.DataFrame(rebalance_page.items)

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

    if visible_columns:
        dataframe = dataframe[visible_columns].copy()

    st.dataframe(
        dataframe,
        width="stretch",
        hide_index=True,
        column_config={
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

    st.caption(
        f"Showing {rebalance_page.count} run(s), "
        f"limit={rebalance_page.limit}, "
        f"offset={rebalance_page.offset}."
    )


def _render_run_summary(
    summary: JsonObject,
) -> None:
    """Render one historical rebalance summary."""

    st.subheader("Run Summary")

    status_column, trade_column, value_column = st.columns(3)

    with status_column:
        st.metric(
            "Status",
            str(summary.get("status", "Unknown")),
        )

    with trade_column:
        st.metric(
            "Trade Count",
            str(summary.get("trade_count", "Unknown")),
        )

    with value_column:
        st.metric(
            "Portfolio Value",
            format_currency(
                summary.get("portfolio_value")
            ),
        )

    cost_column, tax_column = st.columns(2)

    with cost_column:
        st.metric(
            "Transaction Cost",
            format_currency(
                summary.get("transaction_cost")
            ),
        )

    with tax_column:
        st.metric(
            "Estimated Tax",
            format_currency(
                summary.get("estimated_tax_liability")
            ),
        )

    approval_column, pending_column = st.columns(2)

    with approval_column:
        st.metric(
            "Approvals Required",
            str(
                summary.get(
                    "approval_required_count",
                    0,
                )
            ),
        )

    with pending_column:
        st.metric(
            "Pending Approvals",
            str(
                summary.get(
                    "pending_approval_count",
                    0,
                )
            ),
        )

    st.markdown("#### Run Information")

    left_column, right_column = st.columns(2)

    with left_column:
        st.write(
            "**Run ID:**",
            summary.get("run_id", "Unavailable"),
        )

        st.write(
            "**Created:**",
            display_timestamp(
                summary.get("created_at")
            ),
        )

        st.write(
            "**Completed:**",
            display_timestamp(
                summary.get("completed_at")
            ),
        )

    with right_column:
        st.write(
            "**Portfolio ID:**",
            summary.get("portfolio_id", "Unavailable"),
        )

        st.write(
            "**Transaction Cost Rate:**",
            summary.get(
                "transaction_cost_rate",
                "Unavailable",
            ),
        )


def _render_trades(
    trade_page: PaginatedResponse,
) -> None:
    """Render historical rebalance trades."""

    st.subheader("Trades")

    if not trade_page.items:
        st.info("No trades exist for this rebalance run.")
        return

    dataframe = pd.DataFrame(trade_page.items)

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

    if visible_columns:
        dataframe = dataframe[visible_columns].copy()

    for column in (
        "current_weight",
        "trade_weight",
        "post_trade_weight",
    ):
        if column in dataframe.columns:
            dataframe[column] = dataframe[column] * 100

    st.dataframe(
        dataframe,
        width="stretch",
        hide_index=True,
        column_config={
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
        },
    )


def _render_audit(
    audit_page: PaginatedResponse,
) -> None:
    """Render audit records for one historical run."""

    st.subheader("Audit Trail")

    if not audit_page.items:
        st.info("No audit records exist for this run.")
        return

    dataframe = pd.DataFrame(audit_page.items)

    preferred_columns = [
        "audit_id",
        "timestamp",
        "event_type",
        "asset",
        "action",
        "approval_status",
        "approval_reason",
        "reviewed_by",
        "reviewed_at",
        "audit_message",
    ]

    visible_columns = [
        column
        for column in preferred_columns
        if column in dataframe.columns
    ]

    if visible_columns:
        dataframe = dataframe[visible_columns].copy()

    st.dataframe(
        dataframe,
        width="stretch",
        hide_index=True,
    )


def main() -> None:
    """Render persisted rebalance history."""

    settings = get_settings()

    st.set_page_config(
        page_title=f"History | {settings.app_title}",
        page_icon="🕘",
        layout="wide",
    )

    render_sidebar(settings)

    st.title("Rebalance History")
    st.caption(
        "Inspect previous database-backed rebalance runs."
    )

    client = _build_client()

    try:
        portfolio_page = client.list_portfolios(
            limit=20,
            offset=0,
        )
    except ApiClientError as exc:
        st.error(
            f"Could not load portfolios: {exc}"
        )
        return

    portfolio_ids = _extract_portfolio_ids(
        portfolio_page
    )

    if not portfolio_ids:
        st.info("No stored portfolios are available.")
        return

    selected_portfolio = st.selectbox(
        "Select Portfolio",
        options=portfolio_ids,
    )

    try:
        rebalance_page = client.list_portfolio_rebalances(
            portfolio_id=selected_portfolio,
            limit=20,
            offset=0,
        )
    except ApiClientError as exc:
        st.error(
            f"Could not load rebalance history: {exc}"
        )
        return

    _render_run_history(
        rebalance_page
    )

    run_ids = _extract_run_ids(
        rebalance_page
    )

    if not run_ids:
        return

    selected_run = st.selectbox(
        "Select Rebalance Run",
        options=run_ids,
    )

    try:
        summary = client.get_rebalance(
            selected_run
        )
    except ApiClientError as exc:
        st.error(
            f"Could not load run details: {exc}"
        )
        return

    _render_run_summary(
        summary
    )

    try:
        trade_page = client.list_rebalance_trades(
            run_id=selected_run,
            limit=20,
            offset=0,
        )
    except ApiClientError as exc:
        st.warning(
            f"Could not load trades: {exc}"
        )
    else:
        _render_trades(
            trade_page
        )

    try:
        audit_page = client.list_rebalance_audit(
            run_id=selected_run,
            limit=20,
            offset=0,
        )
    except ApiClientError as exc:
        st.warning(
            f"Could not load audit records: {exc}"
        )
    else:
        _render_audit(
            audit_page
        )


if __name__ == "__main__":
    main()