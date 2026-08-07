from __future__ import annotations

import streamlit as st

from streamlit_app.components.navigation import (
    render_sidebar,
)
from streamlit_app.config import get_settings
from streamlit_app.services.api_client import (
    ApiClientError,
    FastApiClient,
)
from streamlit_app.components.metrics import format_currency

def _build_client() -> FastApiClient:
    """Create the reusable API client."""

    settings = get_settings()

    return FastApiClient(
        base_url=settings.api_base_url,
        timeout_seconds=settings.api_timeout_seconds,
    )


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
                portfolio["portfolio_value"],)
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


if __name__ == "__main__":
    main()