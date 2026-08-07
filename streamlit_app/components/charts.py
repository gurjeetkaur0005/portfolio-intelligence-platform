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


def prepare_backtest_portfolio_history_data(
    history: list[dict[str, Any]],
) -> pd.DataFrame:
    """Return display-ready portfolio history rows."""

    dataframe = pd.DataFrame(history)

    if "portfolio_value" not in dataframe.columns:
        return pd.DataFrame()

    result = dataframe[
        [
            "portfolio_value",
        ]
    ].copy()
    result["portfolio_value"] = pd.to_numeric(
        result["portfolio_value"],
        errors="coerce",
    )

    if "date" in dataframe.columns:
        result["period_label"] = dataframe["date"].astype(str)
    elif "period" in dataframe.columns:
        result["period_label"] = dataframe["period"].astype(str)
    else:
        result["period_label"] = [
            str(index)
            for index in range(len(result))
        ]

    result = result.dropna(
        subset=[
            "portfolio_value",
        ]
    )

    return result


def prepare_backtest_drawdown_data(
    history: list[dict[str, Any]],
) -> pd.DataFrame:
    """Return drawdown history only when the backend provides it."""

    dataframe = pd.DataFrame(history)
    drawdown_column = ""

    for candidate in (
        "drawdown",
        "portfolio_drawdown",
    ):
        if candidate in dataframe.columns:
            drawdown_column = candidate
            break

    if not drawdown_column:
        return pd.DataFrame()

    result = dataframe[
        [
            drawdown_column,
        ]
    ].copy()
    result["drawdown_percent"] = pd.to_numeric(
        result[drawdown_column],
        errors="coerce",
    ) * 100

    if "date" in dataframe.columns:
        result["period_label"] = dataframe["date"].astype(str)
    elif "period" in dataframe.columns:
        result["period_label"] = dataframe["period"].astype(str)
    else:
        result["period_label"] = [
            str(index)
            for index in range(len(result))
        ]

    return result.dropna(
        subset=[
            "drawdown_percent",
        ]
    )


def prepare_strategy_comparison_chart_data(
    comparison: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Return grouped strategy comparison rows by compatible scale."""

    buy_and_hold = comparison.get("buy_and_hold")
    threshold = comparison.get("threshold_rebalancing")

    if not isinstance(buy_and_hold, dict) or not isinstance(
        threshold,
        dict,
    ):
        return {}

    percentage_rows = _strategy_comparison_rows(
        buy_and_hold=buy_and_hold,
        threshold=threshold,
        metric_names=[
            "total_return",
            "annualized_return",
            "volatility",
            "maximum_drawdown",
        ],
        multiplier=100.0,
    )
    ratio_rows = _strategy_comparison_rows(
        buy_and_hold=buy_and_hold,
        threshold=threshold,
        metric_names=[
            "sharpe_ratio",
        ],
        multiplier=1.0,
    )
    cost_rows = _strategy_comparison_rows(
        buy_and_hold=buy_and_hold,
        threshold=threshold,
        metric_names=[
            "transaction_costs",
            "taxes_paid",
            "total_implementation_cost",
        ],
        multiplier=1.0,
    )
    count_rows = _strategy_comparison_rows(
        buy_and_hold=buy_and_hold,
        threshold=threshold,
        metric_names=[
            "number_of_rebalances",
        ],
        multiplier=1.0,
    )

    dataframes: dict[str, pd.DataFrame] = {}

    for key, rows in {
        "percentage": percentage_rows,
        "ratio": ratio_rows,
        "cost": cost_rows,
        "count": count_rows,
    }.items():
        if rows:
            dataframes[key] = pd.DataFrame(rows)

    return dataframes


def _strategy_comparison_rows(
    *,
    buy_and_hold: dict[str, Any],
    threshold: dict[str, Any],
    metric_names: list[str],
    multiplier: float,
) -> list[dict[str, Any]]:
    """Return chart rows for compatible strategy metrics."""

    from streamlit_app.services.display import display_label

    rows: list[dict[str, Any]] = []

    for metric_name in metric_names:
        for strategy_name, source in (
            ("Buy & Hold", buy_and_hold),
            ("Threshold Rebalancing", threshold),
        ):
            value = source.get(metric_name)

            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                continue

            rows.append(
                {
                    "metric": display_label(metric_name),
                    "strategy": strategy_name,
                    "value": float(value) * multiplier,
                }
            )

    return rows


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


def render_backtest_portfolio_history(
    history: list[dict[str, Any]],
    *,
    title: str = "Portfolio Value History",
) -> None:
    """Render returned backtest portfolio value history."""

    import streamlit as st
    import plotly.express as px

    dataframe = prepare_backtest_portfolio_history_data(history)

    if dataframe.empty:
        st.info("No portfolio value history was returned.")
        return

    figure = px.line(
        dataframe,
        x="period_label",
        y="portfolio_value",
        markers=True,
        color_discrete_sequence=[
            ACCENT_COLOR,
        ],
    )
    figure.update_traces(
        hovertemplate=(
            "Period %{x}<br>"
            "Value $%{y:,.2f}<extra></extra>"
        )
    )
    _apply_chart_layout(
        figure,
        title=title,
        x_title="Period",
        y_title="Portfolio Value",
    )
    figure.update_yaxes(
        tickprefix="$",
        separatethousands=True,
    )

    st.plotly_chart(
        figure,
        width="stretch",
        config={
            "displayModeBar": False,
        },
    )


def render_backtest_drawdown_history(
    history: list[dict[str, Any]],
) -> None:
    """Render returned drawdown history when available."""

    import streamlit as st
    import plotly.express as px

    dataframe = prepare_backtest_drawdown_data(history)

    if dataframe.empty:
        return

    figure = px.area(
        dataframe,
        x="period_label",
        y="drawdown_percent",
        color_discrete_sequence=[
            "#b42318",
        ],
    )
    figure.update_traces(
        hovertemplate=(
            "Period %{x}<br>"
            "Drawdown %{y:.2f}%<extra></extra>"
        )
    )
    _apply_chart_layout(
        figure,
        title="Drawdown History",
        x_title="Period",
        y_title="Drawdown",
    )
    figure.update_yaxes(
        ticksuffix="%",
    )

    st.plotly_chart(
        figure,
        width="stretch",
        config={
            "displayModeBar": False,
        },
    )


def render_strategy_comparison(
    comparison: dict[str, Any],
) -> None:
    """Render strategy comparison charts by compatible metric group."""

    import streamlit as st
    import plotly.express as px

    dataframes = prepare_strategy_comparison_chart_data(comparison)

    if not dataframes:
        st.info("Strategy comparison chart data is unavailable.")
        return

    chart_specs = [
        (
            "percentage",
            "Returns and Risk",
            "Percent",
            "%",
            None,
        ),
        (
            "ratio",
            "Sharpe Ratio",
            "Ratio",
            None,
            None,
        ),
        (
            "cost",
            "Implementation Costs",
            "Amount",
            None,
            "$",
        ),
        (
            "count",
            "Rebalance Count",
            "Count",
            None,
            None,
        ),
    ]

    for key, title, y_title, suffix, prefix in chart_specs:
        dataframe = dataframes.get(key)

        if dataframe is None or dataframe.empty:
            continue

        figure = px.bar(
            dataframe,
            x="metric",
            y="value",
            color="strategy",
            barmode="group",
            color_discrete_map={
                "Buy & Hold": SECONDARY_COLOR,
                "Threshold Rebalancing": ACCENT_COLOR,
            },
        )
        figure.update_traces(
            hovertemplate=(
                "%{x}<br>"
                "%{legendgroup}: %{y:,.2f}<extra></extra>"
            )
        )
        _apply_chart_layout(
            figure,
            title=title,
            x_title=None,
            y_title=y_title,
        )

        if suffix is not None:
            figure.update_yaxes(ticksuffix=suffix)

        if prefix is not None:
            figure.update_yaxes(
                tickprefix=prefix,
                separatethousands=True,
            )

        st.plotly_chart(
            figure,
            width="stretch",
            config={
                "displayModeBar": False,
            },
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
