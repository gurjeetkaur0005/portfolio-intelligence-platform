from __future__ import annotations

from typing import Any

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
    """Extract valid portfolio identifiers from a portfolio page."""

    portfolio_ids: list[str] = []

    for portfolio in portfolio_page.items:
        portfolio_id = portfolio.get("portfolio_id")

        if isinstance(portfolio_id, str) and portfolio_id.strip():
            portfolio_ids.append(portfolio_id)

    return portfolio_ids


def _render_command_result(
    result: JsonObject,
) -> None:
    """Render the lightweight POST rebalance response."""

    message = result.get("message")

    if isinstance(message, str) and message.strip():
        st.success(message)
    else:
        st.success("Rebalance completed successfully.")

    status_column, trade_column = st.columns(2)

    with status_column:
        st.metric(
            label="Status",
            value=str(result.get("status", "Unknown")),
        )

    with trade_column:
        st.metric(
            label="Trades Created",
            value=str(result.get("trade_count", "Unknown")),
        )


def _render_rebalance_summary(
    summary: JsonObject,
) -> None:
    """Render persisted rebalance run information."""

    st.subheader("Rebalance Summary")

    first_column, second_column, third_column = st.columns(3)

    with first_column:
        st.metric(
            label="Status",
            value=str(summary.get("status", "Unknown")),
        )

    with second_column:
        st.metric(
            label="Portfolio Value",
            value=format_currency(
                summary.get("portfolio_value")
            ),
        )

    with third_column:
        st.metric(
            label="Trade Count",
            value=str(summary.get("trade_count", "Unknown")),
        )

    cost_column, tax_column = st.columns(2)

    with cost_column:
        st.metric(
            label="Transaction Cost",
            value=format_currency(
                summary.get("transaction_cost")
            ),
        )

    with tax_column:
        st.metric(
            label="Estimated Tax",
            value=format_currency(
                summary.get("estimated_tax_liability")
            ),
        )

    approval_column, pending_column = st.columns(2)

    with approval_column:
        st.metric(
            label="Approvals Required",
            value=str(
                summary.get(
                    "approval_required_count",
                    0,
                )
            ),
        )

    with pending_column:
        st.metric(
            label="Pending Approvals",
            value=str(
                summary.get(
                    "pending_approval_count",
                    0,
                )
            ),
        )

    st.markdown("#### Run Information")

    run_column, portfolio_column = st.columns(2)

    with run_column:
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

    with portfolio_column:
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


def _extract_approval(
    trade: JsonObject,
) -> dict[str, Any] | None:
    """Return nested approval metadata when available."""

    approval = trade.get("approval")

    if not isinstance(approval, dict):
        return None

    return approval


def _render_trade_explanations(
    trade_page: PaginatedResponse,
) -> None:
    """Render explanations and approval metadata per trade."""

    st.markdown("### Trade Details")

    for trade in trade_page.items:
        asset = trade.get("asset")
        action = trade.get("action")

        if not isinstance(asset, str) or not asset.strip():
            continue

        action_label = (
            action
            if isinstance(action, str)
            else "Unknown"
        )

        with st.expander(
            f"{asset} — {action_label}"
        ):
            first_column, second_column = st.columns(2)

            with first_column:
                st.write(
                    "**Threshold Breached:**",
                    trade.get(
                        "threshold_breached",
                        "Unknown",
                    ),
                )

                st.write(
                    "**Threshold Severity:**",
                    trade.get(
                        "threshold_severity",
                        "Unknown",
                    ),
                )

                st.write(
                    "**Breach Ratio:**",
                    trade.get(
                        "breach_ratio",
                        "Not available",
                    ),
                )

            with second_column:
                st.write(
                    "**Final Trigger:**",
                    trade.get(
                        "final_trigger_type",
                        "Not available",
                    ),
                )

                st.write(
                    "**Final Priority:**",
                    trade.get(
                        "final_priority",
                        "Not available",
                    ),
                )

                st.write(
                    "**Contributing Triggers:**",
                    trade.get(
                        "contributing_triggers",
                        "Not available",
                    ),
                )

            st.divider()

            st.markdown("#### Client Explanation")

            client_explanation = trade.get(
                "client_explanation"
            )

            st.write(
                client_explanation
                if isinstance(client_explanation, str)
                and client_explanation.strip()
                else "Not available"
            )

            st.markdown("#### Advisor Explanation")

            advisor_explanation = trade.get(
                "advisor_explanation"
            )

            st.write(
                advisor_explanation
                if isinstance(advisor_explanation, str)
                and advisor_explanation.strip()
                else "Not available"
            )

            st.markdown("#### Compliance Explanation")

            compliance_explanation = trade.get(
                "compliance_explanation"
            )

            st.write(
                compliance_explanation
                if isinstance(compliance_explanation, str)
                and compliance_explanation.strip()
                else "Not available"
            )

            approval = _extract_approval(trade)

            if approval is not None:
                st.divider()
                st.markdown("#### Approval")

                approval_left, approval_right = st.columns(2)

                with approval_left:
                    st.write(
                        "**Required:**",
                        approval.get(
                            "required",
                            "Unknown",
                        ),
                    )

                    st.write(
                        "**Status:**",
                        approval.get(
                            "status",
                            "Unknown",
                        ),
                    )

                    st.write(
                        "**Reason:**",
                        approval.get(
                            "reason",
                            "Not available",
                        ),
                    )

                with approval_right:
                    st.write(
                        "**Reviewed By:**",
                        approval.get(
                            "reviewed_by",
                            "Not recorded",
                        ),
                    )

                    st.write(
                        "**Reviewed At:**",
                        display_timestamp(
                            approval.get("reviewed_at")
                        ),
                    )


def _render_trades(
    trade_page: PaginatedResponse,
) -> None:
    """Render persisted trades for the current rebalance run."""

    st.subheader("Trades")

    if not trade_page.items:
        st.info(
            "No trades were returned for this rebalance run."
        )
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
        "threshold_breached",
        "threshold_severity",
        "breach_ratio",
        "final_trigger_type",
        "final_priority",
        "created_at",
    ]

    visible_columns = [
        column
        for column in preferred_columns
        if column in dataframe.columns
    ]

    if visible_columns:
        dataframe = dataframe[
            visible_columns
        ].copy()

    for column in (
        "current_weight",
        "trade_weight",
        "post_trade_weight",
    ):
        if column in dataframe.columns:
            dataframe[column] = (
                dataframe[column] * 100
            )

    st.dataframe(
        dataframe,
        width="stretch",
        hide_index=True,
        column_config={
            "asset": st.column_config.TextColumn(
                "Asset Class",
            ),
            "action": st.column_config.TextColumn(
                "Action",
            ),
            "current_weight": (
                st.column_config.NumberColumn(
                    "Current Weight",
                    format="%.2f%%",
                )
            ),
            "trade_weight": (
                st.column_config.NumberColumn(
                    "Trade Weight",
                    format="%.2f%%",
                )
            ),
            "post_trade_weight": (
                st.column_config.NumberColumn(
                    "Post-Trade Weight",
                    format="%.2f%%",
                )
            ),
            "trade_value": (
                st.column_config.NumberColumn(
                    "Trade Value",
                    format="$%.2f",
                )
            ),
            "estimated_transaction_cost": (
                st.column_config.NumberColumn(
                    "Transaction Cost",
                    format="$%.2f",
                )
            ),
            "estimated_tax": (
                st.column_config.NumberColumn(
                    "Estimated Tax",
                    format="$%.2f",
                )
            ),
            "breach_ratio": (
                st.column_config.NumberColumn(
                    "Breach Ratio",
                    format="%.2f",
                )
            ),
        },
    )

    st.caption(
        f"Showing {trade_page.count} trade(s), "
        f"limit={trade_page.limit}, "
        f"offset={trade_page.offset}."
    )

    _render_trade_explanations(
        trade_page
    )


def _render_audit(
    audit_page: PaginatedResponse,
) -> None:
    """Render persisted audit records."""

    st.subheader("Audit Trail")

    if not audit_page.items:
        st.info(
            "No audit records were returned for this "
            "rebalance run."
        )
        return

    dataframe = pd.DataFrame(
        audit_page.items
    )

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
        dataframe = dataframe[
            visible_columns
        ].copy()

    st.dataframe(
        dataframe,
        width="stretch",
        hide_index=True,
    )

    st.caption(
        f"Showing {audit_page.count} audit record(s), "
        f"limit={audit_page.limit}, "
        f"offset={audit_page.offset}."
    )


def _load_run_results(
    *,
    client: FastApiClient,
    run_id: str,
) -> None:
    """Load and render persisted details for one rebalance run."""

    try:
        summary = client.get_rebalance(
            run_id
        )
    except ApiClientError as exc:
        st.warning(
            "The rebalance completed, but its summary "
            f"could not be loaded: {exc}"
        )
        return

    _render_rebalance_summary(
        summary
    )

    try:
        trade_page = client.list_rebalance_trades(
            run_id=run_id,
            limit=20,
            offset=0,
        )
    except ApiClientError as exc:
        st.warning(
            f"Could not load rebalance trades: {exc}"
        )
    else:
        _render_trades(
            trade_page
        )

    try:
        audit_page = client.list_rebalance_audit(
            run_id=run_id,
            limit=20,
            offset=0,
        )
    except ApiClientError as exc:
        st.warning(
            f"Could not load the audit trail: {exc}"
        )
    else:
        _render_audit(
            audit_page
        )


def main() -> None:
    """Render the database-backed portfolio rebalancing page."""

    settings = get_settings()

    st.set_page_config(
        page_title=(
            f"Rebalancing | {settings.app_title}"
        ),
        page_icon="⚖️",
        layout="wide",
    )

    render_sidebar(
        settings
    )

    st.title(
        "Rebalancing"
    )

    st.caption(
        "Run the deterministic rebalance workflow for "
        "a portfolio stored in PostgreSQL."
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
        st.info(
            "No stored portfolios are available "
            "for rebalancing."
        )
        return

    st.subheader(
        "Rebalance Configuration"
    )

    selected_portfolio = st.selectbox(
        label="Select Portfolio",
        options=portfolio_ids,
    )

    transaction_cost_rate = st.number_input(
        label="Transaction Cost Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.002,
        step=0.001,
        format="%.4f",
        help=(
            "0.002 represents a transaction "
            "cost rate of 0.2%."
        ),
    )

    st.info(
        "Portfolio value and holdings are loaded "
        "from PostgreSQL. The frontend does not "
        "calculate portfolio trades."
    )

    run_button = st.button(
        label="Run Rebalance",
        type="primary",
    )

    if run_button:
        try:
            result = client.run_portfolio_rebalance(
                portfolio_id=selected_portfolio,
                transaction_cost_rate=float(
                    transaction_cost_rate
                ),
            )
        except (
            ApiClientError,
            ValueError,
        ) as exc:
            st.error(
                f"Rebalance failed: {exc}"
            )
            return

        run_id_value = result.get(
            "run_id"
        )

        if not isinstance(
            run_id_value,
            str,
        ) or not run_id_value.strip():
            st.error(
                "The backend completed the request but "
                "did not return a valid run ID."
            )
            return

        st.session_state[
            "rebalance_run_id"
        ] = run_id_value

        _render_command_result(
            result
        )

    stored_run_id = st.session_state.get(
        "rebalance_run_id"
    )

    if isinstance(
        stored_run_id,
        str,
    ) and stored_run_id.strip():
        st.divider()

        _load_run_results(
            client=client,
            run_id=stored_run_id,
        )


if __name__ == "__main__":
    main()