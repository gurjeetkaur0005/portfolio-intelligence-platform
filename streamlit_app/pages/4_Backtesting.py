from __future__ import annotations

import streamlit as st

from streamlit_app.components.navigation import render_sidebar
from streamlit_app.config import get_settings
from streamlit_app.services.api_client import (
    ApiClientError,
    FastApiClient,
    JsonObject,
)


def _build_client() -> FastApiClient:
    """Create the reusable FastAPI client."""

    settings = get_settings()

    return FastApiClient(
        base_url=settings.api_base_url,
        timeout_seconds=settings.api_timeout_seconds,
    )


def _extract_portfolio_ids(
    items: list[JsonObject],
) -> list[str]:
    """Extract valid portfolio identifiers."""

    portfolio_ids: list[str] = []

    for portfolio in items:
        portfolio_id = portfolio.get("portfolio_id")

        if isinstance(portfolio_id, str) and portfolio_id.strip():
            portfolio_ids.append(portfolio_id)

    return portfolio_ids


def main() -> None:
    """Render the database-backed rebalancing page."""

    settings = get_settings()

    st.set_page_config(
        page_title=f"Rebalancing | {settings.app_title}",
        page_icon="⚖️",
        layout="wide",
    )

    render_sidebar(settings)

    st.title("Rebalancing")
    st.caption(
        "Run the deterministic rebalance workflow for a stored portfolio."
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

    portfolio_ids = _extract_portfolio_ids(
        portfolio_page.items
    )

    if not portfolio_ids:
        st.info("No stored portfolios are available.")
        return

    selected_portfolio = st.selectbox(
        "Select Portfolio",
        options=portfolio_ids,
    )

    transaction_cost_rate = st.number_input(
        "Transaction Cost Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.002,
        step=0.001,
        format="%.4f",
        help=(
            "Example: 0.002 represents a 0.2% "
            "transaction cost rate."
        ),
    )

    run_rebalance = st.button(
        "Run Rebalance",
        type="primary",
    )

    if not run_rebalance:
        return

    try:
        result = client.run_portfolio_rebalance(
            portfolio_id=selected_portfolio,
            transaction_cost_rate=float(
                transaction_cost_rate
            ),
        )
    except (ApiClientError, ValueError) as exc:
        st.error(f"Rebalance failed: {exc}")
        return

    st.success(
        str(
            result.get(
                "message",
                "Rebalance completed successfully.",
            )
        )
    )

    st.subheader("Rebalance Result")

    status_column, trade_column = st.columns(2)
    portfolio_column, run_column = st.columns(2)

    with status_column:
        st.metric(
            "Status",
            str(result.get("status", "Unknown")),
        )

    with trade_column:
        st.metric(
            "Trade Count",
            str(result.get("trade_count", "Unknown")),
        )

    with portfolio_column:
        st.metric(
            "Portfolio ID",
            str(result.get("portfolio_id", "Unknown")),
        )

    with run_column:
        st.metric(
            "Run ID",
            str(result.get("run_id", "Unknown")),
        )


if __name__ == "__main__":
    main()
