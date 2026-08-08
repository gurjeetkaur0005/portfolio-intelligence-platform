from __future__ import annotations

import streamlit as st

from streamlit_app.components.cards import (
    render_kpi_card,
    render_page_header,
)
from streamlit_app.components.charts import (
    render_backtest_drawdown_history,
    render_backtest_portfolio_history,
    render_backtest_strategy_drawdown_history,
    render_backtest_strategy_history,
    render_strategy_comparison,
)
from streamlit_app.components.metrics import (
    backtest_metric_help,
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
from streamlit_app.services.help_text import chart_help, input_help
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

STRATEGIES = [
    "Buy & Hold",
    "Threshold Rebalancing",
    "Strategy Comparison",
]

STRATEGY_DESCRIPTIONS = {
    "Buy & Hold": (
        "Invest once and allow the portfolio allocation to move naturally "
        "with market performance."
    ),
    "Threshold Rebalancing": (
        "Rebalance the portfolio when allocation drift exceeds the "
        "configured threshold."
    ),
    "Strategy Comparison": (
        "Compare Buy & Hold and Threshold Rebalancing across return, "
        "risk, and trading activity."
    ),
}

SELECTED_STRATEGY_STATE_KEY = "backtest_selected_strategy"
LAST_RUN_STRATEGY_STATE_KEY = "backtest_result_strategy"
BACKTEST_RESULT_STATE_KEY = "backtest_result"
LEGACY_BACKTEST_RESULT_STATE_KEYS = (
    "buy_and_hold_result",
    "threshold_result",
    "comparison_result",
)


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


def _json_object(
    value: object,
) -> JsonObject | None:
    """Return a JSON object when a session value has the expected shape."""

    if isinstance(value, dict):
        return value

    return None


def backtest_strategy_description(
    strategy: str,
) -> str:
    """Return the concise description for a backtest strategy."""

    return STRATEGY_DESCRIPTIONS.get(strategy, "")


def should_render_backtest_result(
    *,
    selected_strategy: object,
    result_strategy: object,
    result: object,
) -> bool:
    """Return whether a stored result belongs to the selected strategy."""

    return (
        isinstance(selected_strategy, str)
        and selected_strategy == result_strategy
        and isinstance(result, dict)
    )


def backtest_strategy_changed(
    *,
    previous_strategy: object,
    selected_strategy: str,
) -> bool:
    """Return whether a selected strategy differs from stored state."""

    return previous_strategy != selected_strategy


def backtest_view_sections(
    *,
    strategy: str,
    has_result: bool,
) -> tuple[str, ...]:
    """Return the mutually exclusive section identity for a strategy view."""

    result_section = (
        f"{strategy} Results"
        if has_result
        else f"{strategy} Empty State"
    )

    if strategy == "Buy & Hold":
        return (
            "Buy & Hold Description",
            "Buy & Hold Configuration",
            result_section,
        )

    if strategy == "Threshold Rebalancing":
        return (
            "Threshold Rebalancing Description",
            "Threshold Rebalancing Configuration",
            result_section,
        )

    if strategy == "Strategy Comparison":
        return (
            "Strategy Comparison Description",
            "Strategy Comparison Configuration",
            result_section,
        )

    return ()


def backtest_state_keys_to_clear() -> tuple[str, ...]:
    """Return all result-related backtest state keys to clear."""

    return (
        BACKTEST_RESULT_STATE_KEY,
        LAST_RUN_STRATEGY_STATE_KEY,
        *LEGACY_BACKTEST_RESULT_STATE_KEYS,
    )


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


def _drawdown_history(
    result: JsonObject,
) -> list[JsonObject]:
    """Return backend-provided drawdown history records."""

    raw_history = result.get("drawdown_history")

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


def _history_has_field(
    history: list[JsonObject],
    field_names: tuple[str, ...],
) -> bool:
    """Return whether any history row includes one of the fields."""

    for row in history:
        for field_name in field_names:
            if field_name in row:
                return True

    return False


def _integer(
    value: int,
) -> str:
    """Format an integer count."""

    return f"{value:,}"


def _render_kpi(
    *,
    title: str,
    value: str,
    help_key: str,
) -> None:
    """Render a KPI with reusable backtest help copy."""

    render_kpi_card(
        title=title,
        value=value,
        subtitle=backtest_metric_help(help_key) or None,
    )


def _render_metric_group(
    *,
    title: str,
    metrics: list[tuple[str, str, str]],
) -> None:
    """Render one section of related backtest metrics."""

    if not metrics:
        return

    st.subheader(title)

    columns = st.columns(len(metrics))

    for column, (label, value, help_key) in zip(
        columns,
        metrics,
    ):
        with column:
            _render_kpi(
                title=label,
                value=value,
                help_key=help_key,
            )


def _render_performance_metrics(
    result: JsonObject,
) -> None:
    """Render grouped backtest performance metrics."""

    metrics = _metrics(result)

    if metrics is None:
        st.warning("Backtest metrics are unavailable.")
        return

    performance_metrics: list[tuple[str, str, str]] = []
    risk_metrics: list[tuple[str, str, str]] = []

    if "total_return" in metrics:
        performance_metrics.append(
            (
                "Total Return",
                format_percentage(metrics.get("total_return")),
                "total_return",
            )
        )

    if "annualized_return" in metrics:
        performance_metrics.append(
            (
                "Annualized Return",
                format_percentage(metrics.get("annualized_return")),
                "annualized_return",
            )
        )

    if "volatility" in metrics:
        risk_metrics.append(
            (
                "Volatility",
                format_percentage(metrics.get("volatility")),
                "volatility",
            )
        )

    if "sharpe_ratio" in metrics:
        risk_metrics.append(
            (
                "Sharpe Ratio",
                format_number(metrics.get("sharpe_ratio")),
                "sharpe_ratio",
            )
        )

    if "maximum_drawdown" in metrics:
        risk_metrics.append(
            (
                "Maximum Drawdown",
                format_percentage(metrics.get("maximum_drawdown")),
                "maximum_drawdown",
            )
        )

    _render_metric_group(
        title="Performance",
        metrics=performance_metrics,
    )
    _render_metric_group(
        title="Risk",
        metrics=risk_metrics,
    )

    if risk_metrics:
        st.caption(
            "Risk metrics help explain how consistently the strategy "
            "generated returns and how severely the portfolio declined "
            "during unfavorable periods."
        )


def _render_threshold_metrics(
    result: JsonObject,
) -> None:
    """Render threshold-specific fields returned in history rows."""

    history = _history(result)

    trading_metrics: list[tuple[str, str, str]] = []
    cost_metrics: list[tuple[str, str, str]] = []
    drift_metrics: list[tuple[str, str, str]] = []

    if _history_has_field(history, ("trade_count",)):
        trading_metrics.append(
            (
                "Number of Trades",
                _integer(_sum_history_int(history, "trade_count")),
                "number_of_trades",
            )
        )

    if _history_has_field(history, ("rebalanced",)):
        trading_metrics.append(
            (
                "Number of Rebalances",
                _integer(_sum_history_int(history, "rebalanced")),
                "number_of_rebalances",
            )
        )

    if _history_has_field(history, ("turnover",)):
        trading_metrics.append(
            (
                "Turnover",
                format_percentage(_sum_history_number(history, "turnover")),
                "turnover",
            )
        )

    if _history_has_field(history, ("transaction_cost",)):
        cost_metrics.append(
            (
                "Transaction Cost",
                format_currency(
                    _sum_history_number(history, "transaction_cost")
                ),
                "transaction_cost",
            )
        )

    if _history_has_field(history, ("estimated_tax_liability",)):
        cost_metrics.append(
            (
                "Estimated Tax",
                format_currency(
                    _sum_history_number(
                        history,
                        "estimated_tax_liability",
                    )
                ),
                "estimated_tax",
            )
        )

    drift_fields = (
        "max_absolute_drift",
        "maximum_drift",
        "breach_ratio",
    )

    if _history_has_field(history, drift_fields):
        drift_metrics.append(
            (
                "Maximum Drift",
                format_percentage(
                    _max_history_number(
                        history,
                        drift_fields,
                    )
                ),
                "maximum_drift",
            )
        )

    _render_metric_group(
        title="Trading Activity",
        metrics=trading_metrics,
    )
    if trading_metrics:
        st.caption(
            "More frequent rebalancing can keep allocation closer to "
            "target but may increase trading costs."
        )
    _render_metric_group(
        title="Costs",
        metrics=cost_metrics,
    )
    _render_metric_group(
        title="Drift",
        metrics=drift_metrics,
    )


def _render_history_charts(
    result: JsonObject,
) -> None:
    """Render available backtest history charts."""

    history = _history(result)

    st.subheader("Portfolio Growth Chart")

    with st.container(border=True):
        render_backtest_portfolio_history(
            history,
            title="Portfolio Value Over Time",
        )

    if not history:
        return

    drawdown_history = _drawdown_history(result)

    if not drawdown_history:
        return

    st.subheader("Drawdown Chart")
    st.caption(chart_help("drawdown"))
    with st.container(border=True):
        render_backtest_drawdown_history(drawdown_history)


def _render_comparison(
    comparison: JsonObject,
) -> None:
    """Render strategy-comparison output."""

    st.subheader("Strategy Comparison Results")
    st.caption(
        "Use return metrics to understand growth, risk metrics to "
        "understand variability and losses, and trading metrics to "
        "understand how actively each strategy changed the portfolio."
    )

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
        st.markdown("#### Simulation Configuration")
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
            help=(
                input_help("initial_portfolio_value")
            ),
        )
        risk_free_rate = st.number_input(
            "Risk-Free Rate",
            value=0.0,
            step=0.001,
            format="%.4f",
            help=(
                input_help("risk_free_rate")
            ),
        )
        periods_per_year = st.number_input(
            "Periods Per Year",
            min_value=1,
            value=252,
            step=1,
            help=(
                input_help("periods_per_year")
            ),
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
            st.markdown("#### Threshold Strategy Settings")
            st.caption(
                "Used only when running threshold rebalancing or the "
                "side-by-side comparison."
            )
            drift_band = st.number_input(
                "Rebalancing Threshold",
                min_value=0.001,
                value=0.05,
                step=0.005,
                format="%.4f",
                help=(
                    input_help("rebalance_threshold")
                ),
            )
            transaction_cost_rate = st.number_input(
                "Transaction Cost Rate",
                min_value=0.0,
                value=0.002,
                step=0.001,
                format="%.4f",
                help=(
                    input_help("transaction_cost_rate")
                ),
            )
            tax_rate = st.number_input(
                "Tax Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.20,
                step=0.01,
                format="%.4f",
                help=input_help("tax_rate"),
            )
            turnover_budget = st.number_input(
                "Turnover Budget",
                min_value=0.001,
                value=0.10,
                step=0.01,
                format="%.4f",
                help=input_help("turnover_budget"),
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


def _sync_strategy_state(
    strategy: str,
) -> None:
    """Clear stale results when the selected strategy changes."""

    previous_strategy = st.session_state.get(
        SELECTED_STRATEGY_STATE_KEY
    )

    if not backtest_strategy_changed(
        previous_strategy=previous_strategy,
        selected_strategy=strategy,
    ):
        return

    st.session_state[SELECTED_STRATEGY_STATE_KEY] = strategy
    _clear_result()


def _current_result(
    strategy: str,
) -> JsonObject | None:
    """Return the stored result owned by the selected strategy."""

    result_strategy = st.session_state.get(LAST_RUN_STRATEGY_STATE_KEY)
    result = st.session_state.get(BACKTEST_RESULT_STATE_KEY)

    if not should_render_backtest_result(
        selected_strategy=strategy,
        result_strategy=result_strategy,
        result=result,
    ):
        return None

    return _json_object(result)


def _store_result(
    *,
    strategy: str,
    result: JsonObject,
) -> None:
    """Store a successful result with explicit strategy ownership."""

    st.session_state[LAST_RUN_STRATEGY_STATE_KEY] = strategy
    st.session_state[BACKTEST_RESULT_STATE_KEY] = result


def _clear_result() -> None:
    """Remove any previously displayed backtest result."""

    for key in backtest_state_keys_to_clear():
        st.session_state.pop(key, None)


def _render_empty_state() -> None:
    """Render a clean empty result state."""

    with st.container(border=True):
        st.markdown("#### No results yet")
        st.caption(
            "Configure this strategy and run the backtest to see its "
            "results."
        )


def _run_strategy(
    *,
    strategy: str,
    client: FastApiClient,
    initial_portfolio_value: float,
    risk_free_rate: float,
    periods_per_year: int,
    drift_band: float,
    transaction_cost_rate: float,
    tax_rate: float,
    turnover_budget: float,
) -> JsonObject:
    """Execute the selected strategy through the FastAPI client."""

    if strategy == "Buy & Hold":
        return _run_buy_and_hold(
            client=client,
            initial_portfolio_value=initial_portfolio_value,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )

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
        return threshold_result

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

    return {
        "comparison": comparison,
        "buy_and_hold_result": buy_and_hold_result,
        "threshold_result": threshold_result,
    }


def _render_strategy_result(
    *,
    strategy: str,
    result: JsonObject,
) -> None:
    """Render the stored result for the selected strategy only."""

    if strategy == "Buy & Hold":
        st.subheader("Buy & Hold Results")
        _render_performance_metrics(result)
        _render_history_charts(
            result,
        )
        return

    if strategy == "Threshold Rebalancing":
        st.subheader("Threshold Rebalancing Results")
        _render_performance_metrics(result)
        _render_threshold_metrics(result)
        _render_history_charts(
            result,
        )
        return

    comparison = _json_object(result.get("comparison"))
    buy_and_hold_result = _json_object(result.get("buy_and_hold_result"))
    threshold_result = _json_object(result.get("threshold_result"))

    if (
        comparison is None
        or buy_and_hold_result is None
        or threshold_result is None
    ):
        st.warning("Strategy comparison results are unavailable.")
        return

    _render_comparison(comparison)

    st.subheader("Portfolio Growth Chart")
    with st.container(border=True):
        render_backtest_strategy_history(
            buy_and_hold_history=_history(buy_and_hold_result),
            threshold_history=_history(threshold_result),
        )

    buy_and_hold_drawdown = _drawdown_history(buy_and_hold_result)
    threshold_drawdown = _drawdown_history(threshold_result)

    if buy_and_hold_drawdown and threshold_drawdown:
        st.subheader("Drawdown Comparison")
        st.caption(chart_help("drawdown"))
        with st.container(border=True):
            render_backtest_strategy_drawdown_history(
                buy_and_hold_drawdown=buy_and_hold_drawdown,
                threshold_drawdown=threshold_drawdown,
            )


def _render_selected_strategy_view(
    *,
    strategy: str,
    client: FastApiClient,
) -> None:
    """Render exactly one selected strategy view."""

    if strategy == "Buy & Hold":
        _render_buy_and_hold_view(
            client=client,
        )
        return

    if strategy == "Threshold Rebalancing":
        _render_threshold_view(
            client=client,
        )
        return

    if strategy == "Strategy Comparison":
        _render_comparison_view(
            client=client,
        )
        return

    st.info("Select a strategy to continue.")


def _render_buy_and_hold_view(
    *,
    client: FastApiClient,
) -> None:
    """Render only the Buy & Hold configuration and result."""

    _render_strategy_description("Buy & Hold")
    (
        initial_portfolio_value,
        risk_free_rate,
        periods_per_year,
        drift_band,
        transaction_cost_rate,
        tax_rate,
        turnover_budget,
    ) = _render_configuration("Buy & Hold")
    _render_run_and_result(
        strategy="Buy & Hold",
        client=client,
        initial_portfolio_value=initial_portfolio_value,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        drift_band=drift_band,
        transaction_cost_rate=transaction_cost_rate,
        tax_rate=tax_rate,
        turnover_budget=turnover_budget,
    )


def _render_threshold_view(
    *,
    client: FastApiClient,
) -> None:
    """Render only the Threshold Rebalancing configuration and result."""

    _render_strategy_description("Threshold Rebalancing")
    (
        initial_portfolio_value,
        risk_free_rate,
        periods_per_year,
        drift_band,
        transaction_cost_rate,
        tax_rate,
        turnover_budget,
    ) = _render_configuration("Threshold Rebalancing")
    _render_run_and_result(
        strategy="Threshold Rebalancing",
        client=client,
        initial_portfolio_value=initial_portfolio_value,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        drift_band=drift_band,
        transaction_cost_rate=transaction_cost_rate,
        tax_rate=tax_rate,
        turnover_budget=turnover_budget,
    )


def _render_comparison_view(
    *,
    client: FastApiClient,
) -> None:
    """Render only the Strategy Comparison configuration and result."""

    _render_strategy_description("Strategy Comparison")
    (
        initial_portfolio_value,
        risk_free_rate,
        periods_per_year,
        drift_band,
        transaction_cost_rate,
        tax_rate,
        turnover_budget,
    ) = _render_configuration("Strategy Comparison")
    _render_run_and_result(
        strategy="Strategy Comparison",
        client=client,
        initial_portfolio_value=initial_portfolio_value,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        drift_band=drift_band,
        transaction_cost_rate=transaction_cost_rate,
        tax_rate=tax_rate,
        turnover_budget=turnover_budget,
    )


def _render_strategy_description(
    strategy: str,
) -> None:
    """Render the selected strategy description only."""

    st.caption(backtest_strategy_description(strategy))


def _render_run_and_result(
    *,
    strategy: str,
    client: FastApiClient,
    initial_portfolio_value: float,
    risk_free_rate: float,
    periods_per_year: int,
    drift_band: float,
    transaction_cost_rate: float,
    tax_rate: float,
    turnover_budget: float,
) -> None:
    """Run and render a result for the selected strategy only."""

    run_requested = st.button(
        "Run Backtest",
        type="primary",
        key=f"run_backtest_{strategy}",
    )

    if run_requested:
        _clear_result()

        try:
            with st.spinner("Running backtest..."):
                result = _run_strategy(
                    strategy=strategy,
                    client=client,
                    initial_portfolio_value=initial_portfolio_value,
                    risk_free_rate=risk_free_rate,
                    periods_per_year=periods_per_year,
                    drift_band=drift_band,
                    transaction_cost_rate=transaction_cost_rate,
                    tax_rate=tax_rate,
                    turnover_budget=turnover_budget,
                )
        except (ApiClientError, ValueError) as exc:
            st.error(f"Backtest failed: {exc}")
            result = None

        if result is not None:
            _store_result(
                strategy=strategy,
                result=result,
            )

    st.subheader("Results")

    current_result = _current_result(strategy)

    if current_result is None:
        _render_empty_state()
        return

    _render_strategy_result(
        strategy=strategy,
        result=current_result,
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

    render_page_header(
        title="Backtesting",
        description="Compare how portfolio strategies perform over time.",
        context=(
            "Simulation environment only. Results are not live portfolio "
            "execution instructions."
        ),
    )

    client = _build_client()

    st.subheader("Strategy Selection")

    with st.container(border=True):
        strategy = st.selectbox(
            label="Strategy",
            options=STRATEGIES,
            help=input_help("strategy"),
        )

        if not isinstance(strategy, str):
            st.info("Select a strategy to continue.")
            return

        _sync_strategy_state(strategy)

    _render_selected_strategy_view(
        strategy=strategy,
        client=client,
    )


if __name__ == "__main__":
    main()
