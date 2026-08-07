from __future__ import annotations

import streamlit as st

from streamlit_app.components.cards import (
    render_key_value,
    render_kpi_card,
)
from streamlit_app.components.metrics import (
    format_currency,
    format_percentage,
)
from streamlit_app.components.navigation import render_sidebar
from streamlit_app.components.status import status_label
from streamlit_app.components.tables import (
    render_rebalance_audit_table,
    render_rebalance_run_table,
    render_rebalance_trade_table,
)
from streamlit_app.config import get_settings
from streamlit_app.services.api_client import (
    ApiClientError,
    FastApiClient,
    JsonObject,
    JsonValue,
    PaginatedResponse,
)
from streamlit_app.services.display import display_timestamp
from streamlit_app.services.styles import load_global_styles


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


def _text_value(
    value: JsonValue,
    *,
    fallback: str = "Unavailable",
) -> str:
    """Return a safe compact text value from a JSON field."""

    if value is None:
        return fallback

    if isinstance(value, bool):
        return "Yes" if value else "No"

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, str) and value.strip():
        return value

    return fallback


def _render_run_history(
    rebalance_page: PaginatedResponse,
) -> None:
    """Render the portfolio's rebalance history."""

    st.subheader("Historical Runs")

    with st.container(border=True):
        render_rebalance_run_table(rebalance_page.items)
        st.caption(
            f"Showing {rebalance_page.count} run(s), "
            f"limit={rebalance_page.limit}, "
            f"offset={rebalance_page.offset}."
        )


def _render_run_summary(
    summary: JsonObject,
) -> None:
    """Render one historical rebalance summary."""

    st.subheader("Selected Run Summary")

    first, second, third, fourth = st.columns(4)

    with first:
        render_kpi_card(
            title="Status",
            value=status_label(_text_value(summary.get("status"))),
        )

    with second:
        render_kpi_card(
            title="Portfolio Value",
            value=format_currency(summary.get("portfolio_value")),
        )

    with third:
        render_kpi_card(
            title="Trade Count",
            value=_text_value(summary.get("trade_count")),
        )

    with fourth:
        render_kpi_card(
            title="Transaction Cost",
            value=format_currency(summary.get("transaction_cost")),
        )

    fifth, sixth, seventh = st.columns(3)

    with fifth:
        render_kpi_card(
            title="Estimated Tax",
            value=format_currency(
                summary.get("estimated_tax_liability")
            ),
        )

    with sixth:
        render_kpi_card(
            title="Approvals Required",
            value=_text_value(
                summary.get("approval_required_count")
            ),
        )

    with seventh:
        render_kpi_card(
            title="Pending Approvals",
            value=_text_value(
                summary.get("pending_approval_count")
            ),
        )

    with st.container(border=True):
        st.markdown("#### Run Metadata")

        left, right = st.columns(2)

        with left:
            render_key_value(
                label="Run ID",
                value=_text_value(summary.get("run_id")),
            )
            render_key_value(
                label="Portfolio ID",
                value=_text_value(summary.get("portfolio_id")),
            )
            render_key_value(
                label="Transaction Cost Rate",
                value=format_percentage(
                    summary.get("transaction_cost_rate")
                ),
            )

        with right:
            render_key_value(
                label="Created",
                value=display_timestamp(summary.get("created_at")),
            )
            render_key_value(
                label="Completed",
                value=display_timestamp(summary.get("completed_at")),
            )


def _render_trades(
    trade_page: PaginatedResponse,
) -> None:
    """Render historical rebalance trades."""

    st.subheader("Trades")
    render_rebalance_trade_table(trade_page.items)
    st.caption(
        f"Showing {trade_page.count} trade(s), "
        f"limit={trade_page.limit}, "
        f"offset={trade_page.offset}."
    )


def _render_audit(
    audit_page: PaginatedResponse,
) -> None:
    """Render audit records for one historical run."""

    st.subheader("Audit Trail")
    render_rebalance_audit_table(audit_page.items)
    st.caption(
        f"Showing {audit_page.count} audit record(s), "
        f"limit={audit_page.limit}, "
        f"offset={audit_page.offset}."
    )


def main() -> None:
    """Render persisted rebalance history."""

    settings = get_settings()

    st.set_page_config(
        page_title=f"History | {settings.app_title}",
        page_icon="🕘",
        layout="wide",
    )
    load_global_styles()

    render_sidebar(settings)

    st.title("Rebalance History")
    st.markdown(
        (
            "<div class='pm-page-caption'>"
            "Inspect previous database-backed rebalance runs."
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    client = _build_client()

    try:
        portfolio_page = client.list_portfolios(
            limit=20,
            offset=0,
        )
    except ApiClientError as exc:
        st.error(f"Could not load portfolios: {exc}")
        return

    portfolio_ids = _extract_portfolio_ids(portfolio_page)

    if not portfolio_ids:
        st.info("No stored portfolios are available.")
        return

    with st.container(border=True):
        st.caption(
            "Choose a stored portfolio to inspect persisted "
            "rebalance runs."
        )
        selected_portfolio = st.selectbox(
            "Portfolio",
            options=portfolio_ids,
        )

    try:
        rebalance_page = client.list_portfolio_rebalances(
            portfolio_id=selected_portfolio,
            limit=20,
            offset=0,
        )
    except ApiClientError as exc:
        st.error(f"Could not load rebalance history: {exc}")
        return

    _render_run_history(rebalance_page)

    run_ids = _extract_run_ids(rebalance_page)

    if not run_ids:
        return

    with st.container(border=True):
        selected_run = st.selectbox(
            "Rebalance Run",
            options=run_ids,
        )

    try:
        summary = client.get_rebalance(selected_run)
    except ApiClientError as exc:
        st.error(f"Could not load run details: {exc}")
        return

    _render_run_summary(summary)

    try:
        trade_page = client.list_rebalance_trades(
            run_id=selected_run,
            limit=20,
            offset=0,
        )
    except ApiClientError as exc:
        st.warning(f"Could not load trades: {exc}")
    else:
        _render_trades(trade_page)

    try:
        audit_page = client.list_rebalance_audit(
            run_id=selected_run,
            limit=20,
            offset=0,
        )
    except ApiClientError as exc:
        st.warning(f"Could not load audit records: {exc}")
    else:
        _render_audit(audit_page)


if __name__ == "__main__":
    main()
