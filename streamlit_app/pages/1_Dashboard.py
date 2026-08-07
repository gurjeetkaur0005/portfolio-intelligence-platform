from __future__ import annotations

from typing import Any

import streamlit as st

from streamlit_app.components.metrics import (
    format_currency,
    render_metric_card,
)
from streamlit_app.components.navigation import render_sidebar
from streamlit_app.components.tables import render_portfolio_table
from streamlit_app.config import get_settings
from streamlit_app.services.api_client import (
    ApiClientError,
    FastApiClient,
    PaginatedResponse,
)


def _build_client() -> FastApiClient:
    """Create the reusable FastAPI client from frontend settings."""

    settings = get_settings()

    return FastApiClient(
        base_url=settings.api_base_url,
        timeout_seconds=settings.api_timeout_seconds,
    )


def _extract_status(payload: dict[str, Any]) -> str:
    """Extract a readable status from a health response."""

    for key in ("status", "state", "message"):
        value = payload.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip().title()

    return "Available"


def _extract_displayed_portfolio_value(
    portfolio_page: PaginatedResponse,
) -> str:
    """Calculate the value of portfolios on the current page only."""

    total_value = 0.0
    found_value = False

    for portfolio in portfolio_page.items:
        raw_value = portfolio.get("portfolio_value")

        if raw_value is None:
            raw_value = portfolio.get("total_value")

        if not isinstance(
            raw_value,
            (int, float, str),
        ) or isinstance(raw_value, bool):
            continue

        try:
            total_value += float(raw_value)
        except ValueError:
            continue

        found_value = True

    if not found_value:
        return "Not available"

    return format_currency(total_value)


def _render_system_status(
    health_payload: dict[str, Any] | None,
    readiness_payload: dict[str, Any] | None,
) -> None:
    """Render FastAPI and PostgreSQL status cards."""

    st.subheader("System Status")

    health_column, readiness_column = st.columns(2)

    with health_column:
        if health_payload is None:
            st.error("FastAPI health check failed.")
        else:
            health_status = _extract_status(health_payload)
            st.success(f"FastAPI: {health_status}")

    with readiness_column:
        if readiness_payload is None:
            st.error("PostgreSQL readiness check failed.")
        else:
            readiness_status = _extract_status(
                readiness_payload
            )
            st.success(f"Readiness: {readiness_status}")


def main() -> None:
    """Render the database-backed Dashboard page."""

    settings = get_settings()

    st.set_page_config(
        page_title=f"Dashboard | {settings.app_title}",
        page_icon="📊",
        layout="wide",
    )

    render_sidebar(settings)

    st.title("Dashboard")
    st.caption(
        "Live information loaded from the existing FastAPI backend."
    )

    client = _build_client()

    health_payload: dict[str, Any] | None = None
    readiness_payload: dict[str, Any] | None = None
    portfolio_page: PaginatedResponse | None = None

    try:
        health_payload = client.get_health()
    except ApiClientError as exc:
        st.warning(f"Health endpoint unavailable: {exc}")

    try:
        readiness_payload = client.get_readiness()
    except ApiClientError as exc:
        st.warning(f"Readiness endpoint unavailable: {exc}")

    try:
        portfolio_page = client.list_portfolios(
            limit=20,
            offset=0,
        )
    except ApiClientError as exc:
        st.error(f"Could not load portfolios: {exc}")

    _render_system_status(
        health_payload=health_payload,
        readiness_payload=readiness_payload,
    )

    st.subheader("Portfolio Summary")

    portfolio_count_column, value_column, status_column = (
        st.columns(3)
    )

    with portfolio_count_column:
        render_metric_card(
            label="Portfolios Loaded",
            value=(
                str(portfolio_page.count)
                if portfolio_page is not None
                else "Unavailable"
            ),
            help_text=(
                "Number of portfolios returned in the current "
                "paginated response."
            ),
        )

    with value_column:
        render_metric_card(
            label="Displayed Portfolio Value",
            value=(
                _extract_displayed_portfolio_value(
                    portfolio_page
                )
                if portfolio_page is not None
                else "Unavailable"
            ),
            help_text=(
                "Sum of portfolio values visible on this page. "
                "This is not a database-wide total."
            ),
        )

    with status_column:
        backend_status = (
            "Operational"
            if health_payload is not None
            and readiness_payload is not None
            else "Degraded"
        )

        render_metric_card(
            label="Backend Status",
            value=backend_status,
        )

    st.subheader("Stored Portfolios")

    if portfolio_page is None:
        st.info(
            "Portfolio data will appear when the FastAPI backend "
            "is reachable."
        )
    else:
        render_portfolio_table(portfolio_page.items)

        st.caption(
            f"Showing {portfolio_page.count} portfolio(s), "
            f"limit={portfolio_page.limit}, "
            f"offset={portfolio_page.offset}."
        )


if __name__ == "__main__":
    main()
