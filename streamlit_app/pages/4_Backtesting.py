from __future__ import annotations

import pandas as pd
import streamlit as st

from streamlit_app.components.metrics import (
    format_currency,
    render_metric_card,
)
from streamlit_app.components.navigation import render_sidebar
from streamlit_app.config import get_settings
from streamlit_app.services.api_client import (
    ApiClientError,
    FastApiClient,
    JsonObject,
    JsonValue,
)


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


def _percent(
    value: JsonValue,
) -> str:
    """Format a decimal performance value as a percentage."""

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "Not available"

    return f"{float(value) * 100:.2f}%"


def _number(
    value: JsonValue,
) -> str:
    """Format a numeric backend value."""

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "Not available"

    return f"{float(value):,.2f}"


def _integer(
    value: int,
) -> str:
    """Format an integer count."""

    return f"{value:,}"


def _metrics(
    result: JsonObject,
) -> JsonObject | None:
    """Return validated-looking metrics for rendering."""

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
    """Sum one numeric history field."""

    total = 0.0

    for row in history:
        value = row.get(field_name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total += float(value)

    return total


def _sum_history_int(
    history: list[JsonObject],
    field_name: str,
) -> int:
    """Sum integer or boolean history fields."""

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
) -> float:
    """Return the maximum available numeric field value."""

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
        return 0.0

    return max(values)


def _render_performance_metrics(
    result: JsonObject,
) -> None:
    """Render standard backtest performance metrics."""

    metrics = _metrics(result)

    if metrics is None:
        st.warning("Backtest metrics are unavailable.")
        return

    first, second, third = st.columns(3)
    fourth, fifth = st.columns(2)

    with first:
        render_metric_card(
            label="Total Return",
            value=_percent(metrics.get("total_return")),
        )

    with second:
        render_metric_card(
            label="Annualized Return",
            value=_percent(metrics.get("annualized_return")),
        )

    with third:
        render_metric_card(
            label="Volatility",
            value=_percent(metrics.get("volatility")),
        )

    with fourth:
        render_metric_card(
            label="Sharpe Ratio",
            value=_number(metrics.get("sharpe_ratio")),
        )

    with fifth:
        render_metric_card(
            label="Maximum Drawdown",
            value=_percent(metrics.get("maximum_drawdown")),
        )


def _render_threshold_metrics(
    result: JsonObject,
) -> None:
    """Render threshold-specific fields returned in history rows."""

    history = _history(result)

    first, second, third = st.columns(3)
    fourth, fifth = st.columns(2)

    with first:
        render_metric_card(
            label="Number of Trades",
            value=_integer(
                _sum_history_int(history, "trade_count")
            ),
        )

    with second:
        render_metric_card(
            label="Turnover",
            value=_percent(
                _sum_history_number(history, "turnover")
            ),
        )

    with third:
        render_metric_card(
            label="Transaction Costs",
            value=format_currency(
                _sum_history_number(
                    history,
                    "transaction_cost",
                )
            ),
        )

    with fourth:
        render_metric_card(
            label="Estimated Taxes",
            value=format_currency(
                _sum_history_number(
                    history,
                    "estimated_tax_liability",
                )
            ),
        )

    with fifth:
        render_metric_card(
            label="Maximum Drift",
            value=_percent(
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


def _render_history_chart(
    result: JsonObject,
) -> None:
    """Plot portfolio value history returned by the backend."""

    history = _history(result)

    if not history:
        st.info("No portfolio history was returned.")
        return

    dataframe = pd.DataFrame(history)

    if "portfolio_value" not in dataframe.columns:
        st.warning("Portfolio history does not include portfolio value.")
        return

    chart_data = dataframe[["portfolio_value"]].copy()

    if "date" in dataframe.columns:
        chart_data.index = dataframe["date"].astype(str)

    st.line_chart(
        chart_data,
        use_container_width=True,
    )


def _comparison_dataframe(
    comparison: JsonObject,
) -> pd.DataFrame:
    """Build a side-by-side comparison table."""

    buy_and_hold = comparison.get("buy_and_hold")
    threshold = comparison.get("threshold_rebalancing")

    if not isinstance(buy_and_hold, dict) or not isinstance(
        threshold,
        dict,
    ):
        return pd.DataFrame()

    rows = [
        ("Total Return", _percent(buy_and_hold.get("total_return")),
         _percent(threshold.get("total_return"))),
        (
            "Annualized Return",
            _percent(buy_and_hold.get("annualized_return")),
            _percent(threshold.get("annualized_return")),
        ),
        (
            "Volatility",
            _percent(buy_and_hold.get("volatility")),
            _percent(threshold.get("volatility")),
        ),
        (
            "Sharpe Ratio",
            _number(buy_and_hold.get("sharpe_ratio")),
            _number(threshold.get("sharpe_ratio")),
        ),
        (
            "Maximum Drawdown",
            _percent(buy_and_hold.get("maximum_drawdown")),
            _percent(threshold.get("maximum_drawdown")),
        ),
        (
            "Implementation Cost",
            format_currency(
                buy_and_hold.get("total_implementation_cost")
            ),
            format_currency(
                threshold.get("total_implementation_cost")
            ),
        ),
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "Metric",
            "Buy & Hold",
            "Threshold Rebalancing",
        ],
    )


def _render_comparison(
    comparison: JsonObject,
) -> None:
    """Render strategy-comparison output."""

    dataframe = _comparison_dataframe(comparison)

    if dataframe.empty:
        st.warning("Strategy comparison data is unavailable.")
        return

    st.dataframe(
        dataframe,
        width="stretch",
        hide_index=True,
    )

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


def main() -> None:
    """Render the FastAPI-backed backtesting page."""

    settings = get_settings()

    st.set_page_config(
        page_title=f"Backtesting | {settings.app_title}",
        page_icon="📈",
        layout="wide",
    )

    render_sidebar(settings)

    st.title("Backtesting")

    client = _build_client()

    strategy = st.selectbox(
        label="Strategy",
        options=[
            "Buy & Hold",
            "Threshold Rebalancing",
            "Strategy Comparison",
        ],
    )

    with st.expander("Backtest Inputs", expanded=True):
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
        with st.expander(
            "Threshold Settings",
            expanded=True,
        ):
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

    if not st.button(
        "Run Backtest",
        type="primary",
    ):
        return

    try:
        if strategy == "Buy & Hold":
            result = _run_buy_and_hold(
                client=client,
                initial_portfolio_value=float(
                    initial_portfolio_value
                ),
                risk_free_rate=float(risk_free_rate),
                periods_per_year=int(periods_per_year),
            )
            st.subheader("Buy & Hold Results")
            _render_performance_metrics(result)
            _render_history_chart(result)
            return

        threshold_result = _run_threshold(
            client=client,
            initial_portfolio_value=float(
                initial_portfolio_value
            ),
            risk_free_rate=float(risk_free_rate),
            periods_per_year=int(periods_per_year),
            drift_band=float(drift_band),
            transaction_cost_rate=float(
                transaction_cost_rate
            ),
            tax_rate=float(tax_rate),
            turnover_budget=float(turnover_budget),
        )

        if strategy == "Threshold Rebalancing":
            st.subheader("Threshold Rebalancing Results")
            _render_performance_metrics(threshold_result)
            _render_threshold_metrics(threshold_result)
            _render_history_chart(threshold_result)
            return

        buy_and_hold_result = _run_buy_and_hold(
            client=client,
            initial_portfolio_value=float(
                initial_portfolio_value
            ),
            risk_free_rate=float(risk_free_rate),
            periods_per_year=int(periods_per_year),
        )
        comparison = client.compare_strategies(
            buy_and_hold=buy_and_hold_result,
            threshold_rebalancing=threshold_result,
        )
    except (ApiClientError, ValueError) as exc:
        st.error(f"Backtest failed: {exc}")
        return

    st.subheader("Strategy Comparison")
    _render_comparison(comparison)

    left, right = st.columns(2)

    with left:
        st.markdown("#### Buy & Hold")
        _render_history_chart(buy_and_hold_result)

    with right:
        st.markdown("#### Threshold Rebalancing")
        _render_history_chart(threshold_result)


if __name__ == "__main__":
    main()
