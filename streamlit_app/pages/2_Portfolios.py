from __future__ import annotations

from typing import Any

import streamlit as st

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


def main() -> None:
    """Render the Portfolio Details page."""

    settings = get_settings()

    st.set_page_config(
        page_title=f"Portfolio Details | {settings.app_title}",
        page_icon="📊",
        layout="wide",
    )

    render_sidebar(settings)

    st.title("Portfolio Details")

    client = _build_client()

    try:
        portfolio_page = client.list_portfolios()
    except ApiClientError as exc:
        st.error(str(exc))
        return

    portfolio_ids = []

    for portfolio in portfolio_page.items:
        portfolio_id = portfolio.get("portfolio_id")

        if isinstance(portfolio_id, str):
            portfolio_ids.append(portfolio_id)

    if not portfolio_ids:
        st.info("No portfolios found.")
        return

    selected_portfolio = st.selectbox(
        label="Select Portfolio",
        options=portfolio_ids,
    )
    st.success(
        f"Selected Portfolio: {selected_portfolio}"
    )
    try:
        portfolio = client.get_portfolio(
            selected_portfolio,
        )
    except ApiClientError as exc:
        st.error(
            f"Could not load portfolio details: {exc}"
        )
        return

    st.subheader("Portfolio Summary")

    portfolio_id_column, client_id_column = st.columns(2)
    value_column, currency_column = st.columns(2)
    holding_column, _ = st.columns(2)

    with portfolio_id_column:
        st.metric(
            label="Portfolio ID",
            value=str(portfolio["portfolio_id"]),
        )

    with client_id_column:
        st.metric(
            label="Client ID",
            value=str(portfolio["client_id"]),
        )

    with value_column:
        st.metric(
            label="Portfolio Value",
            value=format_currency(
                portfolio["portfolio_value"],
            ),
        )

    with currency_column:
        st.metric(
            label="Currency",
            value=str(portfolio["currency"]),
        )

    with holding_column:
        st.metric(
            label="Holdings",
            value=str(portfolio["holding_count"]),
        )

    st.subheader("Current Holdings")

    holdings = _extract_holdings(portfolio)

    if holdings is not None:
        render_holdings_table(holdings)
    else:
        st.warning("Portfolio holdings are unavailable.")

    st.subheader("Current Allocation")

    if holdings is not None:
        render_allocation_donut_chart(holdings)

    st.subheader("Holding Values")

    if holdings is not None:
        render_holding_value_bar_chart(holdings)


if __name__ == "__main__":
    main()
