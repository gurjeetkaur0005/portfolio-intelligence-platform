from __future__ import annotations

from typing import Any, Protocol

import pandas as pd


CHART_FONT = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
GRID_COLOR = "rgba(31, 41, 55, 0.10)"
TEXT_COLOR = "#1f2937"
MUTED_COLOR = "#667085"
ACCENT_COLOR = "#1f7a8c"
SECONDARY_COLOR = "#64748b"


class ChartFigure(Protocol):
    """Protocol for the Plotly figure methods used by this module."""

    def update_layout(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Update figure layout."""

    def update_xaxes(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Update figure x-axis configuration."""

    def update_yaxes(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Update figure y-axis configuration."""


def _format_asset_label(
    asset: str,
) -> str:
    """Return a human-readable asset label."""

    return asset.replace("_", " ").strip().title()


def _apply_chart_layout(
    figure: ChartFigure,
    *,
    title: str,
    x_title: str | None = None,
    y_title: str | None = None,
) -> ChartFigure:
    """Apply the shared Plotly dashboard layout."""

    figure.update_layout(
        title={
            "text": title,
            "font": {
                "size": 16,
                "color": TEXT_COLOR,
            },
        },
        font={
            "family": CHART_FONT,
            "size": 12,
            "color": TEXT_COLOR,
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={
            "l": 8,
            "r": 8,
            "t": 56,
            "b": 16,
        },
        legend={
            "title": None,
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.22,
            "xanchor": "left",
            "x": 0,
        },
        xaxis_title=x_title,
        yaxis_title=y_title,
    )
    figure.update_xaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
        title_font={
            "color": MUTED_COLOR,
        },
    )
    figure.update_yaxes(
        showgrid=False,
        zeroline=False,
        title_font={
            "color": MUTED_COLOR,
        },
    )

    return figure


def prepare_holding_value_data(
    holdings: list[dict[str, Any]],
) -> pd.DataFrame:
    """Return cleaned holding values for chart rendering."""

    dataframe = pd.DataFrame(holdings)

    if not {
        "asset",
        "current_value",
    }.issubset(dataframe.columns):
        return pd.DataFrame()

    result = dataframe[
        [
            "asset",
            "current_value",
        ]
    ].copy()
    result["asset_label"] = result["asset"].astype(str).map(
        _format_asset_label
    )
    result["current_value"] = pd.to_numeric(
        result["current_value"],
        errors="coerce",
    )
    result = result.dropna(
        subset=[
            "current_value",
        ]
    )

    return result.sort_values(
        "current_value",
        ascending=False,
    )


def prepare_allocation_data(
    holdings: list[dict[str, Any]],
) -> pd.DataFrame:
    """Return cleaned allocation rows for chart rendering."""

    dataframe = pd.DataFrame(holdings)

    if not {
        "asset",
        "current_weight",
    }.issubset(dataframe.columns):
        return pd.DataFrame()

    result = dataframe[
        [
            "asset",
            "current_weight",
        ]
    ].copy()
    result["asset_label"] = result["asset"].astype(str).map(
        _format_asset_label
    )
    result["current_weight"] = pd.to_numeric(
        result["current_weight"],
        errors="coerce",
    )
    result = result.dropna(
        subset=[
            "current_weight",
        ]
    )
    result["allocation_percent"] = result["current_weight"] * 100

    return result


def prepare_portfolio_value_data(
    portfolios: list[dict[str, Any]],
) -> pd.DataFrame:
    """Return portfolio values for dashboard distribution charts."""

    dataframe = pd.DataFrame(portfolios)

    if not {
        "portfolio_id",
        "portfolio_value",
    }.issubset(dataframe.columns):
        return pd.DataFrame()

    result = dataframe[
        [
            "portfolio_id",
            "portfolio_value",
        ]
    ].copy()
    result["portfolio_value"] = pd.to_numeric(
        result["portfolio_value"],
        errors="coerce",
    )
    result = result.dropna(
        subset=[
            "portfolio_value",
        ]
    )

    return result.sort_values(
        "portfolio_value",
        ascending=True,
    )


def prepare_rebalance_allocation_comparison_data(
    trades: list[dict[str, Any]],
) -> pd.DataFrame:
    """Return current and post-trade allocation rows for charting."""

    dataframe = pd.DataFrame(trades)

    if not {
        "asset",
        "current_weight",
        "post_trade_weight",
    }.issubset(dataframe.columns):
        return pd.DataFrame()

    result = dataframe[
        [
            "asset",
            "current_weight",
            "post_trade_weight",
        ]
    ].copy()
    result["asset_label"] = result["asset"].astype(str).map(
        _format_asset_label
    )

    for column in (
        "current_weight",
        "post_trade_weight",
    ):
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    result = result.dropna(
        subset=[
            "current_weight",
            "post_trade_weight",
        ]
    )

    if result.empty:
        return pd.DataFrame()

    comparison = result.melt(
        id_vars=[
            "asset_label",
        ],
        value_vars=[
            "current_weight",
            "post_trade_weight",
        ],
        var_name="allocation_type",
        value_name="weight_percent",
    )
    comparison["weight_percent"] = comparison["weight_percent"] * 100
    comparison["allocation_type"] = comparison["allocation_type"].map(
        {
            "current_weight": "Current Weight",
            "post_trade_weight": "Post-Trade Weight",
        }
    )

    return comparison.sort_values(
        [
            "asset_label",
            "allocation_type",
        ],
        ascending=True,
    )


def render_portfolio_value_distribution(
    portfolios: list[dict[str, Any]],
) -> None:
    """Render portfolio value distribution for visible portfolios."""

    import streamlit as st
    import plotly.express as px

    dataframe = prepare_portfolio_value_data(portfolios)

    if dataframe.empty:
        st.info("Portfolio values are unavailable for this page.")
        return

    figure = px.bar(
        dataframe,
        x="portfolio_value",
        y="portfolio_id",
        orientation="h",
        color_discrete_sequence=[
            ACCENT_COLOR,
        ],
    )
    figure.update_traces(
        hovertemplate=(
            "Portfolio %{y}<br>"
            "Value $%{x:,.2f}<extra></extra>"
        )
    )
    _apply_chart_layout(
        figure,
        title="Portfolio Value Distribution",
        x_title="Portfolio Value",
        y_title="Portfolio",
    )
    figure.update_xaxes(
        tickprefix="$",
        separatethousands=True,
    )

    st.plotly_chart(
        figure,
        width="stretch",
    )


def render_rebalance_allocation_comparison(
    trades: list[dict[str, Any]],
) -> None:
    """Render current vs post-trade allocation for a rebalance run."""

    import streamlit as st
    import plotly.express as px

    dataframe = prepare_rebalance_allocation_comparison_data(trades)

    if dataframe.empty:
        st.info("Allocation comparison data is unavailable.")
        return

    figure = px.bar(
        dataframe,
        x="weight_percent",
        y="asset_label",
        color="allocation_type",
        barmode="group",
        orientation="h",
        color_discrete_map={
            "Current Weight": SECONDARY_COLOR,
            "Post-Trade Weight": ACCENT_COLOR,
        },
    )
    figure.update_traces(
        hovertemplate=(
            "%{y}<br>"
            "%{legendgroup}: %{x:.2f}%<extra></extra>"
        )
    )
    _apply_chart_layout(
        figure,
        title="Allocation Before vs After Rebalance",
        x_title="Weight",
        y_title=None,
    )
    figure.update_xaxes(
        ticksuffix="%",
    )

    st.plotly_chart(
        figure,
        width="stretch",
    )


def render_allocation_donut_chart(
    holdings: list[dict[str, Any]],
) -> None:
    """Render current portfolio allocation as a donut chart."""

    import streamlit as st
    import plotly.express as px

    dataframe = prepare_allocation_data(holdings)

    if dataframe.empty:
        st.info("Allocation data is unavailable.")
        return

    figure = px.pie(
        dataframe,
        names="asset_label",
        values="allocation_percent",
        hole=0.62,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    figure.update_traces(
        textposition="outside",
        textinfo="none",
        hovertemplate=(
            "%{label}<br>"
            "Allocation %{percent:.2%}<extra></extra>"
        ),
    )
    _apply_chart_layout(
        figure,
        title="Current Allocation",
    )
    figure.update_layout(
        showlegend=True,
        margin={
            "l": 8,
            "r": 8,
            "t": 56,
            "b": 48,
        },
    )

    st.plotly_chart(
        figure,
        width="stretch",
    )


def render_holding_value_bar_chart(
    holdings: list[dict[str, Any]],
) -> None:
    """Render current holding values by asset class."""

    import streamlit as st
    import plotly.express as px

    dataframe = prepare_holding_value_data(holdings)

    if dataframe.empty:
        st.info("Holding values are unavailable.")
        return

    figure = px.bar(
        dataframe,
        x="current_value",
        y="asset_label",
        orientation="h",
        color_discrete_sequence=[
            ACCENT_COLOR,
        ],
    )
    figure.update_traces(
        hovertemplate=(
            "%{y}<br>"
            "Value $%{x:,.2f}<extra></extra>"
        )
    )
    _apply_chart_layout(
        figure,
        title="Holding Values",
        x_title="Current Value",
        y_title=None,
    )
    figure.update_xaxes(
        tickprefix="$",
        separatethousands=True,
    )
    figure.update_yaxes(
        autorange="reversed",
    )

    st.plotly_chart(
        figure,
        width="stretch",
    )
