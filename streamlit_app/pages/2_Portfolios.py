from __future__ import annotations

from typing import Any

from streamlit_app.components.cards import render_kpi_card
from streamlit_app.components.charts import (
    render_allocation_donut_chart,
    render_holding_value_bar_chart,
)
from streamlit_app.components.metrics import format_currency
from streamlit_app.components.navigation import (
    render_sidebar,
)
from streamlit_app.components.tables import (
    render_holdings_table,
)
from streamlit_app.config import get_settings
from streamlit_app.services.api_client import (
    ApiClientError,
    FastApiClient,
    JsonObject,
)
from streamlit_app.services.styles import load_global_styles


def _build_client() -> FastApiClient:
    """Create the reusable API client."""

    settings = get_settings()

    return FastApiClient(
        base_url=settings.api_base_url,
        timeout_seconds=settings.api_timeout_seconds,
    )


def _extract_holdings(
    portfolio: JsonObject,
) -> list[dict[str, Any]] | None:
    """Return portfolio holdings narrowed for UI renderers."""

    raw_holdings = portfolio.get("holdings")

    if not isinstance(raw_holdings, list):
        return None

    holdings: list[dict[str, Any]] = []

    for holding in raw_holdings:
        if isinstance(holding, dict):
            holdings.append(dict(holding))

    return holdings


def _portfolio_ids(
    portfolios: list[JsonObject],
) -> list[str]:
    """Return valid portfolio identifiers."""

    portfolio_ids: list[str] = []

    for portfolio in portfolios:
        portfolio_id = portfolio.get("portfolio_id")

        if isinstance(portfolio_id, str) and portfolio_id.strip():
            portfolio_ids.append(portfolio_id)

    return portfolio_ids


def main() -> None:
    """Render the Portfolio Details page."""

    import streamlit as st

    settings = get_settings()

    st.set_page_config(
        page_title=f"Portfolio Details | {settings.app_title}",
        page_icon="📊",
        layout="wide",
    )
    load_global_styles()

    render_sidebar(settings)

    st.title("Portfolio Details")
    st.markdown(
        (
            "<div class='pm-page-caption'>"
            "Inspect portfolio composition loaded from FastAPI."
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    client = _build_client()

    try:
        portfolio_page = client.list_portfolios()
    except ApiClientError as exc:
        st.error(str(exc))
        return

    portfolio_ids = _portfolio_ids(portfolio_page.items)

    if not portfolio_ids:
        st.info("No portfolios found.")
        return

    selected_portfolio = st.selectbox(
        label="Select Portfolio",
        options=portfolio_ids,
    )

    if not isinstance(selected_portfolio, str):
        st.info("Select a portfolio to continue.")
        return

    try:
        portfolio = client.get_portfolio(
            selected_portfolio,
        )
    except ApiClientError as exc:
        st.error(
            f"Could not load portfolio details: {exc}"
        )
        return

    holdings = _extract_holdings(portfolio)

    st.subheader("Summary")

    first, second, third, fourth = st.columns(4)

    with first:
        render_kpi_card(
            title="Portfolio Value",
            value=format_currency(
                portfolio.get("portfolio_value")
            ),
        )

    with second:
        render_kpi_card(
            title="Client ID",
            value=str(portfolio.get("client_id", "Unavailable")),
        )

    with third:
        render_kpi_card(
            title="Holdings",
            value=str(portfolio.get("holding_count", "Unavailable")),
        )

    with fourth:
        render_kpi_card(
            title="Currency",
            value=str(portfolio.get("currency", "Unavailable")),
        )

    st.divider()

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            if holdings is None:
                st.warning("Portfolio holdings are unavailable.")
            else:
                render_allocation_donut_chart(holdings)

    with right:
        with st.container(border=True):
            if holdings is None:
                st.warning("Portfolio holdings are unavailable.")
            else:
                render_holding_value_bar_chart(holdings)

    st.subheader("Current Holdings")

    if holdings is None:
        st.warning("Portfolio holdings are unavailable.")
    else:
        render_holdings_table(holdings)


if __name__ == "__main__":
    main()
