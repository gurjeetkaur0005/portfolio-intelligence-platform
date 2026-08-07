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


if __name__ == "__main__":
    main()