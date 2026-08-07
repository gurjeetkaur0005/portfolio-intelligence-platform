from __future__ import annotations

from streamlit_app.components.cards import render_kpi_card
from streamlit_app.components.charts import (
    render_portfolio_value_distribution,
)
from streamlit_app.components.metrics import format_currency
from streamlit_app.components.navigation import render_sidebar
from streamlit_app.components.status import (
    render_status_badge,
    status_label,
)
from streamlit_app.components.tables import render_portfolio_table
from streamlit_app.config import get_settings
from streamlit_app.services.api_client import (
    ApiClientError,
    FastApiClient,
    JsonObject,
    PaginatedResponse,
    JsonValue,
)
from streamlit_app.services.styles import load_global_styles


def _build_client() -> FastApiClient:
    """Create the reusable FastAPI client from frontend settings."""

    settings = get_settings()

    return FastApiClient(
        base_url=settings.api_base_url,
        timeout_seconds=settings.api_timeout_seconds,
    )


def _status_value(
    payload: JsonObject | None,
) -> str:
    """Return a readable health status."""

    if payload is None:
        return "Unavailable"

    status = payload.get("status")

    if isinstance(status, str):
        return status_label(status)

    return "Available"


def _numeric_value(
    value: JsonValue,
) -> float | None:
    """Return a float for valid numeric JSON values."""

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None

    return None


def _displayed_portfolio_total(
    portfolio_page: PaginatedResponse,
) -> float | None:
    """Return the visible portfolio-value total."""

    total_value = 0.0
    found_value = False

    for portfolio in portfolio_page.items:
        value = _numeric_value(
            portfolio.get("portfolio_value")
        )

        if value is None:
            continue

        total_value += value
        found_value = True

    if not found_value:
        return None

    return total_value


def _render_system_summary(
    *,
    health_payload: JsonObject | None,
    readiness_payload: JsonObject | None,
) -> None:
    """Render compact system status details."""

    import streamlit as st

    with st.container(border=True):
        st.markdown("#### System Status")

        st.write("FastAPI")
        render_status_badge(
            _status_value(health_payload)
        )

        st.write("PostgreSQL")
        render_status_badge(
            _status_value(readiness_payload)
        )

        st.caption(
            "Status is loaded from the FastAPI liveness and "
            "readiness endpoints."
        )


def main() -> None:
    """Render the database-backed Dashboard page."""

    import streamlit as st

    settings = get_settings()

    st.set_page_config(
        page_title=f"Dashboard | {settings.app_title}",
        page_icon="📊",
        layout="wide",
    )
    load_global_styles()

    render_sidebar(settings)

    st.title("PortfolioMind")
    st.markdown(
        (
            "<div class='pm-page-caption'>"
            "Portfolio intelligence and rebalancing overview"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    client = _build_client()

    health_payload: JsonObject | None = None
    readiness_payload: JsonObject | None = None
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

    portfolio_total = (
        _displayed_portfolio_total(portfolio_page)
        if portfolio_page is not None
        else None
    )

    first, second, third, fourth = st.columns(4)

    with first:
        render_kpi_card(
            title="Portfolios",
            value=(
                str(portfolio_page.count)
                if portfolio_page is not None
                else "Unavailable"
            ),
            subtitle="Visible records in the current page.",
        )

    with second:
        render_kpi_card(
            title="Displayed Portfolio Value",
            value=format_currency(portfolio_total),
            subtitle="Sum of currently loaded portfolios.",
        )

    with third:
        render_kpi_card(
            title="FastAPI Status",
            value=_status_value(health_payload),
        )

    with fourth:
        render_kpi_card(
            title="PostgreSQL Status",
            value=_status_value(readiness_payload),
        )

    st.divider()

    chart_column, status_column = st.columns(
        [
            2,
            1,
        ]
    )

    with chart_column:
        with st.container(border=True):
            if portfolio_page is None:
                st.info(
                    "Portfolio values will appear when the "
                    "FastAPI backend is reachable."
                )
            else:
                render_portfolio_value_distribution(
                    portfolio_page.items
                )

    with status_column:
        _render_system_summary(
            health_payload=health_payload,
            readiness_payload=readiness_payload,
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
