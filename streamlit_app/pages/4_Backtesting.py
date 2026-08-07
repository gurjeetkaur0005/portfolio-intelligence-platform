from __future__ import annotations

import streamlit as st

from streamlit_app.components.cards import render_kpi_card
from streamlit_app.components.charts import (
    render_backtest_drawdown_history,
    render_backtest_portfolio_history,
    render_strategy_comparison,
)
from streamlit_app.components.metrics import (
    format_currency,
    format_number,
    format_percentage,
)
from streamlit_app.components.navigation import render_sidebar
from streamlit_app.components.tables import render_strategy_comparison_table
from streamlit_app.config import get_settings
from streamlit_app.services.api_client import (
    ApiClientError,
    FastApiClient,
    JsonObject,
)
from streamlit_app.services.styles import load_global_styles


ASSET_NAMES = [
    "domestic_equity",
    "international_equity",
    "fixed_income",
    "real_estate",
    "commodities",
    "cash",
]

DEFAULT_MARKET_RETURNS = [
    [0.010, 0.008, 0.002, 0.004, 0.003, 0.001],
    [-0.006, -0.004, 0.003, -0.002, 0.005, 0.001],
    [0.012, 0.009, 0.001, 0.006, -0.003, 0.001],
    [0.004, 0.003, 0.002, 0.001, 0.002, 0.001],
    [-0.009, -0.007, 0.004, -0.004, 0.006, 0.001],
    [0.015, 0.011, 0.001, 0.007, 0.004, 0.001],
    [0.003, 0.002, 0.002, 0.003, -0.002, 0.001],
    [0.009, 0.006, 0.001, 0.004, 0.003, 0.001],
    [-0.004, -0.006, 0.003, -0.001, 0.005, 0.001],
    [0.011, 0.008, 0.002, 0.005, -0.001, 0.001],
    [0.006, 0.005, 0.001, 0.002, 0.002, 0.001],
    [0.013, 0.010, 0.000, 0.006, 0.004, 0.001],
]

INITIAL_WEIGHTS = [
    0.45,
    0.15,
    0.25,
    0.05,
    0.03,
    0.07,
]

TARGET_WEIGHTS = [
    0.40,
    0.18,
    0.27,
    0.06,
    0.04,
    0.05,
]


def _build_client() -> FastApiClient:
    """Create the reusable FastAPI client."""

    settings = get_settings()

    return FastApiClient(
        base_url=settings.api_base_url,
        timeout_seconds=settings.api_timeout_seconds,
    )


def _metrics(
    result: JsonObject,
) -> JsonObject | None:
    """Return metrics from a backtest response when available."""

    metrics = result.get("metrics")

    if isinstance(metrics, dict):
        return metrics

    return None


def _history(
    result: JsonObject,
) -> list[JsonObject]:
    """Return history records from a backtest response."""

    raw_history = result.get("portfolio_history")

    if not isinstance(raw_history, list):
        return []

    history: list[JsonObject] = []

    for row in raw_history:
        if isinstance(row, dict):
            history.append(row)

    return history


def _sum_history_number(
    history: list[JsonObject],
    field_name: str,
) -> float:
    """Sum one numeric history field returned by the backend."""

    total = 0.0

    for row in history:
        value = row.get(field_name)

        if isinstance(value, (int, float)) and not isinstance(
            value,
            bool,
        ):
            total += float(value)

    return total


def _sum_history_int(
    history: list[JsonObject],
    field_name: str,
) -> int:
    """Sum integer or boolean history fields returned by the backend."""

    total = 0

    for row in history:
        value = row.get(field_name)

        if isinstance(value, bool):
            total += int(value)
        elif isinstance(value, int):
            total += value

    return total


def _max_history_number(
    history: list[JsonObject],
    field_names: tuple[str, ...],
) -> float | None:
    """Return the maximum returned numeric history field value."""

    values: list[float] = []

    for row in history:
        for field_name in field_names:
            value = row.get(field_name)

            if isinstance(value, (int, float)) and not isinstance(
                value,
                bool,
            ):
                values.append(float(value))
                break

    if not values:
        return None

    return max(values)


def _integer(
    value: int,
) -> str:
    """Format an integer count."""

    return f"{value:,}"


def _render_performance_metrics(
    result: JsonObject,
) -> None:
    """Render standard backtest performance metrics."""

    metrics = _metrics(result)

    if metrics is None:
        st.warning("Backtest metrics are unavailable.")
        return

    first, second, third, fourth, fifth = st.columns(5)

    with first:
        render_kpi_card(
            title="Total Return",
            value=format_percentage(metrics.get("total_return")),
        )

    with second:
        render_kpi_card(
            title="Annualized Return",
            value=format_percentage(metrics.get("annualized_return")),
        )

    with third:
        render_kpi_card(
            title="Volatility",
            value=format_percentage(metrics.get("volatility")),
        )

    with fourth:
        render_kpi_card(
            title="Sharpe Ratio",
            value=format_number(metrics.get("sharpe_ratio")),
        )

    with fifth:
        render_kpi_card(
            title="Maximum Drawdown",
            value=format_percentage(metrics.get("maximum_drawdown")),
        )


def _render_threshold_metrics(
    result: JsonObject,
) -> None:
    """Render threshold-specific fields returned in history rows."""

    history = _history(result)

    st.subheader("Strategy-Specific Metrics")
    st.caption(
        "Values in this section summarize fields returned in the "
        "threshold backtest history."
    )

    first, second, third = st.columns(3)
    fourth, fifth, sixth = st.columns(3)

    with first:
        render_kpi_card(
            title="Number of Trades",
            value=_integer(_sum_history_int(history, "trade_count")),
        )

    with second:
        render_kpi_card(
            title="Rebalances",
            value=_integer(_sum_history_int(history, "rebalanced")),
        )

    with third:
        render_kpi_card(
            title="Turnover",
            value=format_percentage(
                _sum_history_number(history, "turnover")
            ),
        )

    with fourth:
        render_kpi_card(
            title="Transaction Costs",
            value=format_currency(
                _sum_history_number(history, "transaction_cost")
            ),
        )

    with fifth:
        render_kpi_card(
            title="Estimated Taxes",
            value=format_currency(
                _sum_history_number(
                    history,
                    "estimated_tax_liability",
                )
            ),
        )

    with sixth:
        render_kpi_card(
            title="Maximum Drift",
            value=format_percentage(
                _max_history_number(
                    history,
                    (
                        "max_absolute_drift",
                        "maximum_drift",
                        "breach_ratio",
                    ),
                )
            ),
        )


def _render_history_charts(
    result: JsonObject,
    *,
    title: str,
) -> None:
    """Render available backtest history charts."""

    history = _history(result)

    st.subheader("Portfolio Value Chart")

    with st.container(border=True):
        render_backtest_portfolio_history(
            history,
            title=title,
        )
        render_backtest_drawdown_history(history)


def _render_comparison(
    comparison: JsonObject,
) -> None:
    """Render strategy-comparison output."""

    st.subheader("Strategy Comparison")

    with st.container(border=True):
        render_strategy_comparison_table(comparison)

    with st.container(border=True):
        render_strategy_comparison(comparison)

    summary = comparison.get("performance_summary")

    if isinstance(summary, str) and summary.strip():
        st.info(summary)


def _run_buy_and_hold(
    *,
    client: FastApiClient,
    initial_portfolio_value: float,
    risk_free_rate: float,
    periods_per_year: int,
) -> JsonObject:
    """Call the Buy & Hold backtest endpoint."""

    return client.run_buy_and_hold_backtest(
        asset_names=ASSET_NAMES,
        market_returns=DEFAULT_MARKET_RETURNS,
        initial_weights=INITIAL_WEIGHTS,
        initial_portfolio_value=initial_portfolio_value,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )


def _run_threshold(
    *,
    client: FastApiClient,
    initial_portfolio_value: float,
    risk_free_rate: float,
    periods_per_year: int,
    drift_band: float,
    transaction_cost_rate: float,
    tax_rate: float,
    turnover_budget: float,
) -> JsonObject:
    """Call the Threshold Rebalancing backtest endpoint."""

    return client.run_threshold_backtest(
        asset_names=ASSET_NAMES,
        market_returns=DEFAULT_MARKET_RETURNS,
        initial_weights=INITIAL_WEIGHTS,
        target_weights=TARGET_WEIGHTS,
        initial_portfolio_value=initial_portfolio_value,
        drift_band=drift_band,
        transaction_cost_rate=transaction_cost_rate,
        tax_rate=tax_rate,
        turnover_budget=turnover_budget,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )


def _render_configuration(
    strategy: str,
) -> tuple[float, float, int, float, float, float, float]:
    """Render backtest configuration controls."""

    with st.container(border=True):
        st.caption(
            "Backtests simulate deterministic strategy behavior from "
            "submitted market-return inputs. They are not live trading "
            "recommendations."
        )

        initial_portfolio_value = st.number_input(
            "Initial Portfolio Value",
            min_value=1_000.0,
            value=100_000.0,
            step=10_000.0,
            format="%.2f",
        )
        risk_free_rate = st.number_input(
            "Risk-Free Rate",
            value=0.0,
            step=0.001,
            format="%.4f",
        )
        periods_per_year = st.number_input(
            "Periods Per Year",
            min_value=1,
            value=252,
            step=1,
        )

    drift_band = 0.05
    transaction_cost_rate = 0.002
    tax_rate = 0.20
    turnover_budget = 0.10

    if strategy in {
        "Threshold Rebalancing",
        "Strategy Comparison",
    }:
        with st.container(border=True):
            st.caption("Threshold strategy settings")
            drift_band = st.number_input(
                "Drift Band",
                min_value=0.001,
                value=0.05,
                step=0.005,
                format="%.4f",
            )
            transaction_cost_rate = st.number_input(
                "Transaction Cost Rate",
                min_value=0.0,
                value=0.002,
                step=0.001,
                format="%.4f",
            )
            tax_rate = st.number_input(
                "Tax Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.20,
                step=0.01,
                format="%.4f",
            )
            turnover_budget = st.number_input(
                "Turnover Budget",
                min_value=0.001,
                value=0.10,
                step=0.01,
                format="%.4f",
            )

    return (
        float(initial_portfolio_value),
        float(risk_free_rate),
        int(periods_per_year),
        float(drift_band),
        float(transaction_cost_rate),
        float(tax_rate),
        float(turnover_budget),
    )


def main() -> None:
    """Render the FastAPI-backed backtesting page."""

    settings = get_settings()

    st.set_page_config(
        page_title=f"Backtesting | {settings.app_title}",
        page_icon="📈",
        layout="wide",
    )
    load_global_styles()

    render_sidebar(settings)

    st.title("Backtesting")
    st.markdown(
        (
            "<div class='pm-page-caption'>"
            "Simulate deterministic strategy performance through FastAPI."
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    client = _build_client()

    strategy = st.selectbox(
        label="Strategy",
        options=[
            "Buy & Hold",
            "Threshold Rebalancing",
            "Strategy Comparison",
        ],
    )

    (
        initial_portfolio_value,
        risk_free_rate,
        periods_per_year,
        drift_band,
        transaction_cost_rate,
        tax_rate,
        turnover_budget,
    ) = _render_configuration(strategy)

    if not st.button(
        "Run Backtest",
        type="primary",
    ):
        return

    try:
        with st.spinner("Running deterministic backtest..."):
            if strategy == "Buy & Hold":
                result = _run_buy_and_hold(
                    client=client,
                    initial_portfolio_value=initial_portfolio_value,
                    risk_free_rate=risk_free_rate,
                    periods_per_year=periods_per_year,
                )
                st.subheader("Performance KPIs")
                _render_performance_metrics(result)
                _render_history_charts(
                    result,
                    title="Buy & Hold Portfolio Value",
                )
                return

            threshold_result = _run_threshold(
                client=client,
                initial_portfolio_value=initial_portfolio_value,
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
                drift_band=drift_band,
                transaction_cost_rate=transaction_cost_rate,
                tax_rate=tax_rate,
                turnover_budget=turnover_budget,
            )

            if strategy == "Threshold Rebalancing":
                st.subheader("Performance KPIs")
                _render_performance_metrics(threshold_result)
                _render_history_charts(
                    threshold_result,
                    title="Threshold Rebalancing Portfolio Value",
                )
                _render_threshold_metrics(threshold_result)
                return

            buy_and_hold_result = _run_buy_and_hold(
                client=client,
                initial_portfolio_value=initial_portfolio_value,
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
            )
            comparison = client.compare_strategies(
                buy_and_hold=buy_and_hold_result,
                threshold_rebalancing=threshold_result,
            )
    except (ApiClientError, ValueError) as exc:
        st.error(f"Backtest failed: {exc}")
        return

    _render_comparison(comparison)

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            render_backtest_portfolio_history(
                _history(buy_and_hold_result),
                title="Buy & Hold Portfolio Value",
            )

    with right:
        with st.container(border=True):
            render_backtest_portfolio_history(
                _history(threshold_result),
                title="Threshold Rebalancing Portfolio Value",
            )


if __name__ == "__main__":
    main()
