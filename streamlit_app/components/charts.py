from __future__ import annotations

from typing import Any, Protocol

import pandas as pd


CHART_FONT = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
GRID_COLOR = "rgba(148, 163, 184, 0.18)"
TEXT_COLOR = "#f8fafc"
MUTED_COLOR = "#cbd5e1"
ACCENT_COLOR = "#3b82f6"
SECONDARY_COLOR = "#94a3b8"
POSITIVE_COLOR = "#22c55e"
NEGATIVE_COLOR = "#ef4444"
WARNING_COLOR = "#f59e0b"
CHART_PALETTE = [
    "#3b82f6",
    "#22c55e",
    "#f59e0b",
    "#a78bfa",
    "#14b8a6",
    "#f97316",
]

__all__ = [
    "apply_portfoliomind_chart_layout",
    "prepare_allocation_data",
    "prepare_backtest_drawdown_data",
    "prepare_drawdown_chart_data",
    "prepare_backtest_portfolio_history_data",
    "prepare_backtest_strategy_drawdown_data",
    "prepare_backtest_strategy_history_data",
    "prepare_current_vs_target_allocation_data",
    "prepare_holding_value_data",
    "prepare_portfolio_value_data",
    "prepare_drift_chart_data",
    "prepare_trade_value_chart_data",
    "prepare_cost_tax_impact_data",
    "prepare_rebalance_allocation_comparison_data",
    "prepare_strategy_comparison_chart_data",
    "prepare_target_allocation_data",
    "render_allocation_donut_chart",
    "render_backtest_drawdown_history",
    "render_backtest_strategy_drawdown_history",
    "render_backtest_portfolio_history",
    "render_backtest_strategy_history",
    "render_current_vs_target_allocation",
    "render_holding_value_bar_chart",
    "render_drift_chart",
    "render_portfolio_value_distribution",
    "render_rebalance_allocation_comparison",
    "render_trade_value_chart",
    "render_cost_tax_impact_chart",
    "render_strategy_comparison",
    "render_target_allocation_donut_chart",
]


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

    from streamlit_app.services.display import display_label

    return display_label(asset)


def apply_portfoliomind_chart_layout(
    figure: ChartFigure,
    *,
    title: str,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    legend_title: str | None = None,
) -> ChartFigure:
    """Apply PortfolioMind's shared Plotly dashboard layout."""

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
        hoverlabel={
            "bgcolor": "#1b2638",
            "bordercolor": "#2a374a",
            "font": {
                "color": TEXT_COLOR,
                "family": CHART_FONT,
            },
        },
        margin={
            "l": 8,
            "r": 8,
            "t": 56,
            "b": 16,
        },
        legend={
            "title": {
                "text": legend_title,
                "font": {
                    "color": MUTED_COLOR,
                },
            },
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.22,
            "xanchor": "left",
            "x": 0,
            "font": {
                "color": MUTED_COLOR,
            },
        },
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
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


def _apply_chart_layout(
    figure: ChartFigure,
    *,
    title: str,
    x_title: str | None = None,
    y_title: str | None = None,
) -> ChartFigure:
    """Apply the shared Plotly dashboard layout."""

    return apply_portfoliomind_chart_layout(
        figure,
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
    )


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

    columns = [
        "asset",
        "current_weight",
    ]

    if "current_value" in dataframe.columns:
        columns.append("current_value")

    result = dataframe[columns].copy()
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

    if "current_value" in result.columns:
        result["current_value"] = pd.to_numeric(
            result["current_value"],
            errors="coerce",
        )

    return result


def prepare_target_allocation_data(
    holdings: list[dict[str, object]],
) -> pd.DataFrame:
    """Return cleaned target allocation rows for chart rendering."""

    dataframe = pd.DataFrame(holdings)

    if not {
        "asset",
        "target_weight",
    }.issubset(dataframe.columns):
        return pd.DataFrame()

    result = dataframe[
        [
            "asset",
            "target_weight",
        ]
    ].copy()
    result["asset_label"] = result["asset"].astype(str).map(
        _format_asset_label
    )
    result["target_weight"] = pd.to_numeric(
        result["target_weight"],
        errors="coerce",
    )
    result = result.dropna(
        subset=[
            "target_weight",
        ]
    )
    result["allocation_percent"] = result["target_weight"] * 100

    return result


def prepare_current_vs_target_allocation_data(
    holdings: list[dict[str, object]],
) -> pd.DataFrame:
    """Return current and target allocation rows for charting."""

    dataframe = pd.DataFrame(holdings)

    if not {
        "asset",
        "current_weight",
        "target_weight",
    }.issubset(dataframe.columns):
        return pd.DataFrame()

    result = dataframe[
        [
            "asset",
            "current_weight",
            "target_weight",
        ]
    ].copy()
    result["asset_label"] = result["asset"].astype(str).map(
        _format_asset_label
    )

    for column in (
        "current_weight",
        "target_weight",
    ):
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    result = result.dropna(
        subset=[
            "current_weight",
            "target_weight",
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
            "target_weight",
        ],
        var_name="allocation_type",
        value_name="weight_percent",
    )
    comparison["weight_percent"] = comparison["weight_percent"] * 100
    comparison["allocation_type"] = comparison["allocation_type"].map(
        {
            "current_weight": "Current Allocation",
            "target_weight": "Target Allocation",
        }
    )

    return comparison.sort_values(
        [
            "asset_label",
            "allocation_type",
        ],
        ascending=True,
    )


def prepare_drift_chart_data(
    holdings: list[dict[str, object]],
) -> pd.DataFrame:
    """Return signed allocation drift rows for charting."""

    dataframe = pd.DataFrame(holdings)

    if not {
        "asset",
        "drift",
    }.issubset(dataframe.columns):
        return pd.DataFrame()

    result = dataframe[
        [
            "asset",
            "drift",
        ]
    ].copy()
    result["asset_label"] = result["asset"].astype(str).map(
        _format_asset_label
    )
    result["drift_percent"] = pd.to_numeric(
        result["drift"],
        errors="coerce",
    ) * 100
    result = result.dropna(
        subset=[
            "drift_percent",
        ]
    )

    if result.empty:
        return pd.DataFrame()

    result["drift_direction"] = result["drift_percent"].map(
        _drift_direction
    )

    return result.sort_values(
        "drift_percent",
        key=lambda series: series.abs(),
        ascending=True,
    )


def prepare_trade_value_chart_data(
    trades: list[dict[str, Any]],
) -> pd.DataFrame:
    """Return display-only trade amounts by asset and action."""

    dataframe = pd.DataFrame(trades)

    if not {
        "asset",
        "action",
        "trade_value",
    }.issubset(dataframe.columns):
        return pd.DataFrame()

    result = dataframe[
        [
            "asset",
            "action",
            "trade_value",
        ]
    ].copy()
    result["asset_label"] = result["asset"].astype(str).map(
        _format_asset_label
    )
    result["trade_value"] = pd.to_numeric(
        result["trade_value"],
        errors="coerce",
    )
    result = result.dropna(
        subset=[
            "trade_value",
        ]
    )

    if result.empty:
        return pd.DataFrame()

    result["action"] = result["action"].astype(str).str.upper()
    result["display_trade_value"] = result["trade_value"].where(
        result["action"] != "SELL",
        -result["trade_value"].abs(),
    )
    result["display_trade_value"] = result["display_trade_value"].where(
        result["action"] != "BUY",
        result["trade_value"].abs(),
    )
    result["hover_trade_value"] = result["trade_value"].abs()

    return result.sort_values(
        "display_trade_value",
        ascending=True,
    )


def prepare_cost_tax_impact_data(
    trades: list[dict[str, Any]],
) -> pd.DataFrame:
    """Return transaction-cost and tax rows by asset."""

    dataframe = pd.DataFrame(trades)

    if not {
        "asset",
        "estimated_transaction_cost",
        "estimated_tax",
    }.issubset(dataframe.columns):
        return pd.DataFrame()

    result = dataframe[
        [
            "asset",
            "estimated_transaction_cost",
            "estimated_tax",
        ]
    ].copy()
    result["asset_label"] = result["asset"].astype(str).map(
        _format_asset_label
    )

    for column in (
        "estimated_transaction_cost",
        "estimated_tax",
    ):
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    result = result.dropna(
        subset=[
            "estimated_transaction_cost",
            "estimated_tax",
        ]
    )

    if result.empty:
        return pd.DataFrame()

    impact = result.melt(
        id_vars=[
            "asset_label",
        ],
        value_vars=[
            "estimated_transaction_cost",
            "estimated_tax",
        ],
        var_name="impact_type",
        value_name="amount",
    )
    impact["impact_type"] = impact["impact_type"].map(
        {
            "estimated_transaction_cost": "Transaction Cost",
            "estimated_tax": "Estimated Tax",
        }
    )
    impact = impact[impact["amount"] > 0]

    return impact.sort_values(
        "amount",
        ascending=True,
    )


def _drift_direction(
    value: float,
) -> str:
    """Return a sign label for drift chart coloring."""

    if value > 0:
        return "Overweight"

    if value < 0:
        return "Underweight"

    return "Near Target"


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
            "current_weight": "Current Allocation",
            "post_trade_weight": "Post-Trade Allocation",
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


def prepare_drawdown_chart_data(
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


def prepare_backtest_drawdown_data(
    history: list[dict[str, Any]],
) -> pd.DataFrame:
    """Return drawdown chart rows for backwards-compatible callers."""

    return prepare_drawdown_chart_data(history)


def prepare_backtest_strategy_history_data(
    *,
    buy_and_hold_history: list[dict[str, Any]],
    threshold_history: list[dict[str, Any]],
) -> pd.DataFrame:
    """Return combined portfolio-value history for strategy comparison."""

    rows: list[pd.DataFrame] = []

    for strategy, history in (
        (
            "Buy & Hold",
            buy_and_hold_history,
        ),
        (
            "Threshold Rebalancing",
            threshold_history,
        ),
    ):
        dataframe = prepare_backtest_portfolio_history_data(history)

        if dataframe.empty:
            continue

        dataframe = dataframe.copy()
        dataframe["strategy"] = strategy
        rows.append(dataframe)

    if not rows:
        return pd.DataFrame()

    return pd.concat(
        rows,
        ignore_index=True,
    )


def prepare_backtest_strategy_drawdown_data(
    *,
    buy_and_hold_drawdown: list[dict[str, Any]],
    threshold_drawdown: list[dict[str, Any]],
) -> pd.DataFrame:
    """Return comparable drawdown history for both strategies."""

    rows: list[pd.DataFrame] = []

    for strategy, history in (
        (
            "Buy & Hold",
            buy_and_hold_drawdown,
        ),
        (
            "Threshold Rebalancing",
            threshold_drawdown,
        ),
    ):
        dataframe = prepare_drawdown_chart_data(history)

        if dataframe.empty:
            continue

        dataframe = dataframe.copy()
        dataframe["strategy"] = strategy
        rows.append(dataframe)

    if not rows:
        return pd.DataFrame()

    return pd.concat(
        rows,
        ignore_index=True,
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

    return_rows = _strategy_comparison_rows(
        buy_and_hold=buy_and_hold,
        threshold=threshold,
        metric_names=[
            "total_return",
            "annualized_return",
        ],
        multiplier=100.0,
    )
    risk_rows = _strategy_comparison_rows(
        buy_and_hold=buy_and_hold,
        threshold=threshold,
        metric_names=[
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
        "return": return_rows,
        "risk": risk_rows,
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
            "Current Allocation": SECONDARY_COLOR,
            "Post-Trade Allocation": ACCENT_COLOR,
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


def render_drift_chart(
    holdings: list[dict[str, object]],
) -> None:
    """Render signed current-minus-target allocation drift."""

    import streamlit as st
    import plotly.express as px

    dataframe = prepare_drift_chart_data(holdings)

    if dataframe.empty:
        st.info("Drift data is unavailable.")
        return

    figure = px.bar(
        dataframe,
        x="drift_percent",
        y="asset_label",
        color="drift_direction",
        orientation="h",
        custom_data=[
            "drift_direction",
        ],
        color_discrete_map={
            "Overweight": POSITIVE_COLOR,
            "Underweight": NEGATIVE_COLOR,
            "Near Target": SECONDARY_COLOR,
        },
    )
    figure.update_traces(
        hovertemplate=(
            "%{y}<br>"
            "Drift: %{x:+.2f} percentage points<br>"
            "Status: %{customdata[0]}<extra></extra>"
        )
    )
    _apply_chart_layout(
        figure,
        title="Allocation Drift by Asset",
        x_title="Current minus Target",
        y_title=None,
    )
    figure.update_xaxes(
        ticksuffix="%",
        zeroline=True,
        zerolinecolor=GRID_COLOR,
    )

    st.plotly_chart(
        figure,
        width="stretch",
    )


def render_trade_value_chart(
    trades: list[dict[str, Any]],
) -> None:
    """Render display-only trade amounts by asset."""

    import streamlit as st
    import plotly.express as px

    dataframe = prepare_trade_value_chart_data(trades)

    if dataframe.empty:
        st.info("Trade value data is unavailable.")
        return

    figure = px.bar(
        dataframe,
        x="display_trade_value",
        y="asset_label",
        color="action",
        orientation="h",
        custom_data=[
            "hover_trade_value",
        ],
        color_discrete_map={
            "BUY": POSITIVE_COLOR,
            "SELL": NEGATIVE_COLOR,
            "HOLD": SECONDARY_COLOR,
        },
    )
    figure.update_traces(
        hovertemplate=(
            "%{y}<br>"
            "Action: %{legendgroup}<br>"
            "Trade Amount: $%{customdata[0]:,.2f}<extra></extra>"
        )
    )
    _apply_chart_layout(
        figure,
        title="Proposed Trade Value by Asset",
        x_title="Trade Amount",
        y_title=None,
    )
    figure.update_xaxes(
        tickprefix="$",
        separatethousands=True,
        zeroline=True,
        zerolinecolor=GRID_COLOR,
    )

    st.plotly_chart(
        figure,
        width="stretch",
    )


def render_cost_tax_impact_chart(
    trades: list[dict[str, Any]],
) -> None:
    """Render non-zero transaction cost and tax impact by asset."""

    import streamlit as st
    import plotly.express as px

    dataframe = prepare_cost_tax_impact_data(trades)

    if dataframe.empty:
        st.caption(
            "Cost and tax impact is not material for the returned trades."
        )
        return

    figure = px.bar(
        dataframe,
        x="amount",
        y="asset_label",
        color="impact_type",
        barmode="group",
        orientation="h",
        color_discrete_map={
            "Transaction Cost": WARNING_COLOR,
            "Estimated Tax": NEGATIVE_COLOR,
        },
    )
    figure.update_traces(
        hovertemplate=(
            "%{y}<br>"
            "%{legendgroup}: $%{x:,.2f}<extra></extra>"
        )
    )
    _apply_chart_layout(
        figure,
        title="Estimated Cost and Tax Impact",
        x_title="Amount",
        y_title=None,
    )
    figure.update_xaxes(
        tickprefix="$",
        separatethousands=True,
    )

    st.plotly_chart(
        figure,
        width="stretch",
    )


def render_backtest_portfolio_history(
    history: list[dict[str, Any]],
    *,
    title: str = "Portfolio Value Over Time",
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
        st.info("Drawdown history is not available for this backtest.")
        return

    figure = px.area(
        dataframe,
        x="period_label",
        y="drawdown_percent",
        color_discrete_sequence=[
            NEGATIVE_COLOR,
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
        title="Drawdown Over Time",
        x_title="Period",
        y_title="Drawdown",
    )
    figure.update_yaxes(
        ticksuffix="%",
        zeroline=True,
        zerolinecolor=GRID_COLOR,
    )

    st.plotly_chart(
        figure,
        width="stretch",
        config={
            "displayModeBar": False,
        },
    )


def render_backtest_strategy_history(
    *,
    buy_and_hold_history: list[dict[str, Any]],
    threshold_history: list[dict[str, Any]],
) -> None:
    """Render comparable portfolio-value history for both strategies."""

    import streamlit as st
    import plotly.express as px

    dataframe = prepare_backtest_strategy_history_data(
        buy_and_hold_history=buy_and_hold_history,
        threshold_history=threshold_history,
    )

    if dataframe.empty:
        st.info("No comparable portfolio value history was returned.")
        return

    figure = px.line(
        dataframe,
        x="period_label",
        y="portfolio_value",
        color="strategy",
        markers=True,
        color_discrete_map={
            "Buy & Hold": SECONDARY_COLOR,
            "Threshold Rebalancing": ACCENT_COLOR,
        },
    )
    figure.update_traces(
        hovertemplate=(
            "%{legendgroup}<br>"
            "Period %{x}<br>"
            "Value $%{y:,.2f}<extra></extra>"
        )
    )
    _apply_chart_layout(
        figure,
        title="Portfolio Value Over Time",
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


def render_backtest_strategy_drawdown_history(
    *,
    buy_and_hold_drawdown: list[dict[str, Any]],
    threshold_drawdown: list[dict[str, Any]],
) -> None:
    """Render comparable drawdown history for both strategies."""

    import streamlit as st
    import plotly.express as px

    dataframe = prepare_backtest_strategy_drawdown_data(
        buy_and_hold_drawdown=buy_and_hold_drawdown,
        threshold_drawdown=threshold_drawdown,
    )

    if dataframe.empty:
        return

    figure = px.area(
        dataframe,
        x="period_label",
        y="drawdown_percent",
        color="strategy",
        color_discrete_map={
            "Buy & Hold": SECONDARY_COLOR,
            "Threshold Rebalancing": NEGATIVE_COLOR,
        },
    )
    figure.update_traces(
        hovertemplate=(
            "%{legendgroup}<br>"
            "Period %{x}<br>"
            "Drawdown: %{y:.2f}%<extra></extra>"
        )
    )
    _apply_chart_layout(
        figure,
        title="Drawdown Comparison",
        x_title="Period",
        y_title="Drawdown",
    )
    figure.update_yaxes(
        ticksuffix="%",
        zeroline=True,
        zerolinecolor=GRID_COLOR,
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
            "return",
            "Return Comparison",
            "Percent",
            "%",
            None,
        ),
        (
            "risk",
            "Risk Comparison",
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
        custom_data=[
            "current_value",
        ] if "current_value" in dataframe.columns else None,
        color_discrete_sequence=CHART_PALETTE,
    )
    if "current_value" in dataframe.columns:
        figure.update_traces(
            textposition="outside",
            textinfo="none",
            hovertemplate=(
                "%{label}<br>"
                "Current Allocation: %{percent:.2%}<br>"
                "Current Value: $%{customdata[0]:,.2f}<extra></extra>"
            ),
        )
    else:
        figure.update_traces(
            textposition="outside",
            textinfo="none",
            hovertemplate=(
                "%{label}<br>"
                "Current Allocation: %{percent:.2%}<extra></extra>"
            ),
        )
    _apply_chart_layout(
        figure,
        title="Current Portfolio Composition",
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


def render_target_allocation_donut_chart(
    holdings: list[dict[str, object]],
) -> None:
    """Render target portfolio allocation as a donut chart."""

    import streamlit as st
    import plotly.express as px

    dataframe = prepare_target_allocation_data(holdings)

    if dataframe.empty:
        st.info("Target allocation data is unavailable.")
        return

    figure = px.pie(
        dataframe,
        names="asset_label",
        values="allocation_percent",
        hole=0.62,
        color_discrete_sequence=CHART_PALETTE,
    )
    figure.update_traces(
        textposition="outside",
        textinfo="none",
        hovertemplate=(
            "%{label}<br>"
            "Target %{percent:.2%}<extra></extra>"
        ),
    )
    _apply_chart_layout(
        figure,
        title="Target Allocation",
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


def render_current_vs_target_allocation(
    holdings: list[dict[str, object]],
) -> None:
    """Render current vs target allocation from API holdings."""

    import streamlit as st
    import plotly.express as px

    dataframe = prepare_current_vs_target_allocation_data(holdings)

    if dataframe.empty:
        st.info("Current and target allocation data is unavailable.")
        return

    figure = px.bar(
        dataframe,
        x="weight_percent",
        y="asset_label",
        color="allocation_type",
        barmode="group",
        orientation="h",
        color_discrete_map={
            "Current Allocation": SECONDARY_COLOR,
            "Target Allocation": ACCENT_COLOR,
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
        title="Current vs Target Allocation",
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
