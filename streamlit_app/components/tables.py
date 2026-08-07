from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def render_portfolio_table(
    portfolios: list[dict[str, Any]],
) -> None:
    """Render a table containing stored portfolios."""

    if not portfolios:
        st.info("No portfolios were returned by the backend.")
        return

    dataframe = pd.DataFrame(portfolios)

    preferred_columns = [
        "portfolio_id",
        "client_id",
        "risk_category",
        "portfolio_value",
        "created_at",
        "updated_at",
    ]

    visible_columns = [
        column
        for column in preferred_columns
        if column in dataframe.columns
    ]

    if visible_columns:
        dataframe = dataframe[visible_columns]

    st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True,
    )

def render_holdings_table(
    holdings: list[dict[str, Any]],
) -> None:
    """Render portfolio holdings in a user-friendly table."""

    if not holdings:
        st.info("No holdings were returned by the backend.")
        return

    dataframe = pd.DataFrame(holdings)

    preferred_columns = [
        "asset",
        "current_weight",
        "current_value",
        "cost_basis",
    ]

    visible_columns = [
        column
        for column in preferred_columns
        if column in dataframe.columns
    ]

    if visible_columns:
        dataframe = dataframe[visible_columns].copy()

    if "current_weight" in dataframe.columns:
        dataframe["current_weight"] = (
            dataframe["current_weight"] * 100
        )

    st.dataframe(
        dataframe,
        width="stretch",
        hide_index=True,
        column_config={
            "asset": st.column_config.TextColumn(
                "Asset Class",
            ),
            "current_weight": st.column_config.NumberColumn(
                "Current Weight",
                format="%.2f%%",
            ),
            "current_value": st.column_config.NumberColumn(
                "Current Value",
                format="$%.2f",
            ),
            "cost_basis": st.column_config.NumberColumn(
                "Cost Basis",
                format="$%.2f",
            ),
        },
    )