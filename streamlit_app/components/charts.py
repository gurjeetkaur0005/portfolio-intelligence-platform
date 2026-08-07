from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


def render_allocation_donut_chart(
    holdings: list[dict[str, Any]],
) -> None:
    """Render current portfolio allocation as a donut chart."""

    if not holdings:
        st.info("No holdings are available for allocation analysis.")
        return

    dataframe = pd.DataFrame(holdings)

    required_columns = {
        "asset",
        "current_weight",
    }

    if not required_columns.issubset(dataframe.columns):
        st.warning(
            "The holdings response does not contain allocation data."
        )
        return

    dataframe = dataframe.copy()
    dataframe["allocation_percent"] = (
        dataframe["current_weight"] * 100
    )

    figure = px.pie(
        dataframe,
        names="asset",
        values="allocation_percent",
        hole=0.55,
        color_discrete_sequence=px.colors.qualitative.Set3,
    )

    figure.update_traces(
        textposition="inside",
        textinfo="percent+label",
    )

    figure.update_layout(
        legend_title_text="Asset Class",
    )

    st.plotly_chart(
        figure,
        width="stretch",
    )
def render_holding_value_bar_chart(
    holdings: list[dict[str, Any]],
) -> None:
    """Render current holding values by asset class."""

    if not holdings:
        st.info("No holdings are available for value analysis.")
        return

    dataframe = pd.DataFrame(holdings)

    required_columns = {
        "asset",
        "current_value",
    }

    if not required_columns.issubset(dataframe.columns):
        st.warning(
            "The holdings response does not contain holding values."
        )
        return

    figure = px.bar(
        dataframe,
        x="asset",
        y="current_value",
        title="Current Holding Value by Asset Class",
        labels={
            "asset": "Asset Class",
            "current_value": "Current Value",
        },
    )

    figure.update_layout(
        xaxis_tickangle=-30,
    )

    st.plotly_chart(
        figure,
        width="stretch",
    )