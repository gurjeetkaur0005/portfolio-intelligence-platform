from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

from streamlit_app.components.charts import (
    _format_asset_label,
    prepare_allocation_data,
    prepare_backtest_drawdown_data,
    prepare_backtest_portfolio_history_data,
    prepare_current_vs_target_allocation_data,
    prepare_holding_value_data,
    prepare_portfolio_value_data,
    prepare_rebalance_allocation_comparison_data,
    prepare_strategy_comparison_chart_data,
    prepare_target_allocation_data,
)
from streamlit_app.components.status import (
    approval_status_label,
    drift_label,
    drift_status_label,
    drift_status_tone,
    payload_status_label,
    status_label,
    status_tone,
)
from streamlit_app.components.tables import (
    prepare_holdings_table_data,
    prepare_rebalance_run_table_data,
    prepare_rebalance_trade_table_data,
    prepare_strategy_comparison_table_data,
)
from streamlit_app.services.display import display_label


sys.modules.setdefault(
    "streamlit",
    SimpleNamespace(session_state={}),
)

REBALANCING_PAGE = importlib.import_module(
    "streamlit_app.pages.3_Rebalancing"
)


def test_asset_label_formatting() -> None:
    """Asset identifiers are rendered as readable labels."""

    assert _format_asset_label("domestic_equity") == "Domestic Equity"
    assert (
        _format_asset_label("international_equity")
        == "International Equity"
    )
    assert _format_asset_label("fixed_income") == "Fixed Income"
    assert _format_asset_label("real_estate") == "Real Estate"


def test_status_label_and_tone_mapping() -> None:
    """Status helpers map known states to readable tones."""

    assert status_label("not_configured") == "Not Configured"
    assert status_tone("healthy") == "positive"
    assert status_tone("BUY") == "positive"
    assert status_tone("pending") == "warning"
    assert status_tone("Not Configured") == "warning"
    assert status_tone("SELL") == "negative"
    assert status_tone("failed") == "negative"
    assert status_tone("HOLD") == "neutral"
    assert status_tone("unknown") == "neutral"


def test_prepare_allocation_data_handles_empty_input() -> None:
    """Allocation preparation safely handles empty data."""

    assert prepare_allocation_data([]).empty
    assert prepare_target_allocation_data([]).empty
    assert prepare_current_vs_target_allocation_data([]).empty


def test_prepare_holding_value_data_sorts_descending() -> None:
    """Holding value chart data is sorted for presentation."""

    result = prepare_holding_value_data(
        [
            {
                "asset": "cash",
                "current_value": 10.0,
            },
            {
                "asset": "domestic_equity",
                "current_value": 100.0,
            },
        ]
    )

    assert list(result["asset_label"]) == [
        "Domestic Equity",
        "Cash",
    ]


def test_prepare_portfolio_value_data_handles_malformed_values() -> None:
    """Portfolio value data ignores malformed values."""

    result = prepare_portfolio_value_data(
        [
            {
                "portfolio_id": "P1",
                "portfolio_value": "bad",
            },
            {
                "portfolio_id": "P2",
                "portfolio_value": 100.0,
            },
        ]
    )

    assert list(result["portfolio_id"]) == ["P2"]


def test_prepare_holdings_table_data_formats_assets_and_weights() -> None:
    """Holdings table prep makes labels readable."""

    result = prepare_holdings_table_data(
        [
            {
                "asset": "fixed_income",
                "current_weight": 0.25,
                "target_weight": 0.30,
                "drift": -0.05,
                "current_value": 250.0,
                "cost_basis": 200.0,
            }
        ]
    )

    assert result.loc[0, "asset"] == "Fixed Income"
    assert result.loc[0, "current_weight"] == 25.0
    assert result.loc[0, "target_weight"] == 30.0
    assert result.loc[0, "drift"] == "-5.00%"


def test_prepare_target_allocation_data_uses_api_target_weight() -> None:
    """Target allocation prep consumes backend target weights."""

    result = prepare_target_allocation_data(
        [
            {
                "asset": "domestic_equity",
                "current_weight": 0.45,
                "target_weight": 0.40,
            }
        ]
    )

    assert list(result["asset_label"]) == ["Domestic Equity"]
    assert list(result["allocation_percent"]) == [40.0]


def test_prepare_current_vs_target_allocation_data() -> None:
    """Current vs target chart data uses returned allocation fields."""

    result = prepare_current_vs_target_allocation_data(
        [
            {
                "asset": "domestic_equity",
                "current_weight": 0.45,
                "target_weight": 0.40,
            }
        ]
    )

    assert list(result["asset_label"]) == [
        "Domestic Equity",
        "Domestic Equity",
    ]
    assert set(result["allocation_type"]) == {
        "Current Weight",
        "Target Weight",
    }
    assert set(result["weight_percent"]) == {
        45.0,
        40.0,
    }


def test_drift_formatting_uses_signed_percentages() -> None:
    """Drift helpers format backend drift values for display."""

    assert drift_label(0.012) == "+1.20%"
    assert drift_label(-0.008) == "-0.80%"
    assert drift_label(0) == "0.00%"
    assert drift_label("bad") == "Not available"


def test_drift_status_helpers_color_by_sign_only() -> None:
    """Drift status helpers classify by sign without thresholds."""

    assert drift_status_label(0.012) == "Positive Drift"
    assert drift_status_tone(0.012) == "positive"
    assert drift_status_label(-0.008) == "Negative Drift"
    assert drift_status_tone(-0.008) == "negative"
    assert drift_status_label(0) == "Near Target"
    assert drift_status_tone(0) == "neutral"


def test_prepare_rebalance_allocation_comparison_data() -> None:
    """Rebalance allocation chart data uses persisted weights."""

    result = prepare_rebalance_allocation_comparison_data(
        [
            {
                "asset": "domestic_equity",
                "current_weight": 0.45,
                "post_trade_weight": 0.40,
            }
        ]
    )

    assert list(result["asset_label"]) == [
        "Domestic Equity",
        "Domestic Equity",
    ]
    assert set(result["allocation_type"]) == {
        "Current Weight",
        "Post-Trade Weight",
    }
    assert set(result["weight_percent"]) == {
        45.0,
        40.0,
    }


def test_prepare_rebalance_allocation_requires_both_weights() -> None:
    """Allocation comparison skips rows with missing required weights."""

    result = prepare_rebalance_allocation_comparison_data(
        [
            {
                "asset": "cash",
                "current_weight": 0.10,
            }
        ]
    )

    assert result.empty


def test_prepare_rebalance_trade_table_data_formats_display_columns() -> None:
    """Trade table prep keeps only useful high-level columns."""

    result = prepare_rebalance_trade_table_data(
        [
            {
                "asset": "international_equity",
                "action": "SELL",
                "current_weight": 0.30,
                "trade_weight": -0.05,
                "post_trade_weight": 0.25,
                "trade_value": -5000.0,
                "estimated_transaction_cost": 10.0,
                "estimated_tax": 100.0,
                "threshold_severity": "high",
                "final_trigger_type": "threshold",
                "client_explanation": "Lower-priority detail.",
            }
        ]
    )

    assert list(result.columns) == [
        "asset",
        "action",
        "current_weight",
        "trade_weight",
        "post_trade_weight",
        "trade_value",
        "estimated_transaction_cost",
        "estimated_tax",
        "threshold_severity",
        "final_trigger_type",
    ]
    assert result.loc[0, "asset"] == "International Equity"
    assert result.loc[0, "trade_weight"] == -5.0


def test_rebalance_page_approval_status_uses_persisted_status() -> None:
    """Approval status display does not infer beyond persisted metadata."""

    assert (
        approval_status_label(
            required=True,
            status="pending",
        )
        == "pending"
    )
    assert (
        approval_status_label(
            required=False,
            status=None,
        )
        == "Not Required"
    )


def test_display_label_formats_metric_names() -> None:
    """Display labels make backend identifiers readable."""

    assert display_label("annualized_return") == "Annualized Return"
    assert display_label("maximum_drawdown") == "Maximum Drawdown"
    assert display_label("number_of_rebalances") == "Number Of Rebalances"


def test_payload_status_label_preserves_not_configured() -> None:
    """Health status mapping preserves real backend states."""

    assert payload_status_label({"status": "ready"}) == "Healthy"
    assert (
        payload_status_label({"status": "not_configured"})
        == "Not Configured"
    )
    assert payload_status_label(None) == "Unavailable"


def test_rebalance_submission_state_can_reset(
    monkeypatch,
) -> None:
    """Rebalance page submission state is not permanently disabled."""

    fake_streamlit = SimpleNamespace(session_state={})
    monkeypatch.setitem(
        sys.modules,
        "streamlit",
        fake_streamlit,
    )

    REBALANCING_PAGE._set_rebalance_request_active(True)
    assert REBALANCING_PAGE._rebalance_request_active() is True

    REBALANCING_PAGE._set_rebalance_request_active(False)
    assert REBALANCING_PAGE._rebalance_request_active() is False


def test_prepare_backtest_portfolio_history_data() -> None:
    """Backtest history prep keeps returned portfolio values."""

    result = prepare_backtest_portfolio_history_data(
        [
            {
                "date": "initial",
                "portfolio_value": 100_000.0,
            },
            {
                "date": "1",
                "portfolio_value": "bad",
            },
        ]
    )

    assert list(result["period_label"]) == ["initial"]
    assert list(result["portfolio_value"]) == [100_000.0]


def test_prepare_backtest_portfolio_history_handles_empty_data() -> None:
    """Backtest chart data safely handles missing history fields."""

    assert prepare_backtest_portfolio_history_data([]).empty
    assert prepare_backtest_drawdown_data(
        [
            {
                "date": "initial",
                "portfolio_value": 100_000.0,
            }
        ]
    ).empty


def test_prepare_backtest_drawdown_data_uses_returned_series() -> None:
    """Drawdown prep only uses a backend-provided drawdown series."""

    result = prepare_backtest_drawdown_data(
        [
            {
                "date": "1",
                "drawdown": -0.02,
            }
        ]
    )

    assert list(result["period_label"]) == ["1"]
    assert list(result["drawdown_percent"]) == [-2.0]


def test_prepare_strategy_comparison_chart_data_groups_scales() -> None:
    """Strategy comparison chart data separates incompatible scales."""

    result = prepare_strategy_comparison_chart_data(
        {
            "buy_and_hold": {
                "total_return": 0.10,
                "sharpe_ratio": 0.75,
                "transaction_costs": 0.0,
                "number_of_rebalances": 0,
            },
            "threshold_rebalancing": {
                "total_return": 0.12,
                "sharpe_ratio": 0.90,
                "transaction_costs": 10.0,
                "number_of_rebalances": 2,
            },
        }
    )

    assert set(result) == {
        "percentage",
        "ratio",
        "cost",
        "count",
    }
    assert set(result["percentage"]["metric"]) == {"Total Return"}
    assert set(result["count"]["value"]) == {0.0, 2.0}


def test_prepare_strategy_comparison_table_data_formats_values() -> None:
    """Comparison table prep formats side-by-side metric values."""

    result = prepare_strategy_comparison_table_data(
        {
            "buy_and_hold": {
                "total_return": 0.10,
                "annualized_return": 0.08,
                "volatility": 0.12,
                "maximum_drawdown": 0.05,
                "sharpe_ratio": 0.66,
                "transaction_costs": 0.0,
                "taxes_paid": 0.0,
                "total_implementation_cost": 0.0,
                "number_of_rebalances": 0,
            },
            "threshold_rebalancing": {
                "total_return": 0.11,
                "annualized_return": 0.09,
                "volatility": 0.10,
                "maximum_drawdown": 0.03,
                "sharpe_ratio": 0.90,
                "transaction_costs": 10.0,
                "taxes_paid": 20.0,
                "total_implementation_cost": 30.0,
                "number_of_rebalances": 2,
            },
        }
    )

    assert result.loc[0, "Metric"] == "Total Return"
    assert result.loc[0, "Buy & Hold"] == "10.00%"
    assert result.loc[0, "Threshold Rebalancing"] == "11.00%"


def test_prepare_rebalance_run_table_data_formats_numbers() -> None:
    """History run table prep keeps concise persisted run fields."""

    result = prepare_rebalance_run_table_data(
        [
            {
                "run_id": "RUN1",
                "status": "completed",
                "created_at": "2026-01-01T00:00:00",
                "portfolio_value": "100000",
                "transaction_cost": "25.5",
                "extra": "hidden",
            }
        ]
    )

    assert list(result.columns) == [
        "run_id",
        "status",
        "created_at",
        "portfolio_value",
        "transaction_cost",
    ]
    assert result.loc[0, "portfolio_value"] == 100000
