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