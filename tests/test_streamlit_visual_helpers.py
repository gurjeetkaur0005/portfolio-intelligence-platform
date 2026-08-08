from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

from streamlit_app.components.charts import (
    _format_asset_label,
    prepare_allocation_data,
    prepare_backtest_drawdown_data,
    prepare_backtest_portfolio_history_data,
    prepare_backtest_strategy_drawdown_data,
    prepare_backtest_strategy_history_data,
    prepare_current_vs_target_allocation_data,
    prepare_cost_tax_impact_data,
    prepare_drift_chart_data,
    prepare_drawdown_chart_data,
    prepare_holding_value_data,
    prepare_portfolio_value_data,
    prepare_rebalance_allocation_comparison_data,
    prepare_strategy_comparison_chart_data,
    prepare_target_allocation_data,
    prepare_trade_value_chart_data,
)
from streamlit_app.components.metrics import (
    backtest_metric_help,
    format_currency,
    format_percentage,
)
from streamlit_app.components.pagination import (
    page_offset,
    page_summary,
    reset_page_on_selection_change,
)
from streamlit_app.components.status import (
    approval_reason_label,
    approval_required_label,
    approval_reviewed_at_label,
    approval_reviewer_label,
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
from streamlit_app.services.api_client import PaginatedResponse
from streamlit_app.services.display import display_label
from streamlit_app.services.help_text import (
    chart_help,
    input_help,
    metric_help,
    status_help,
)


sys.modules.setdefault(
    "streamlit",
    SimpleNamespace(session_state={}),
)

REBALANCING_PAGE = importlib.import_module(
    "streamlit_app.pages.3_Rebalancing"
)
BACKTESTING_PAGE = importlib.import_module(
    "streamlit_app.pages.4_Backtesting"
)
DASHBOARD_PAGE = importlib.import_module(
    "streamlit_app.pages.1_Dashboard"
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


def test_prepare_allocation_data_carries_current_value_for_hover() -> None:
    """Current composition prep keeps backend current value for hover text."""

    result = prepare_allocation_data(
        [
            {
                "asset": "domestic_equity",
                "current_weight": 0.45,
                "current_value": "45000.25",
            }
        ]
    )

    assert list(result["asset_label"]) == ["Domestic Equity"]
    assert list(result["allocation_percent"]) == [45.0]
    assert list(result["current_value"]) == [45000.25]


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
        "Current Allocation",
        "Target Allocation",
    }
    assert set(result["weight_percent"]) == {
        45.0,
        40.0,
    }


def test_prepare_drift_chart_data_uses_api_drift() -> None:
    """Drift chart prep only formats backend-provided drift values."""

    result = prepare_drift_chart_data(
        [
            {
                "asset": "domestic_equity",
                "drift": 0.012,
            },
            {
                "asset": "fixed_income",
                "drift": -0.008,
            },
        ]
    )

    assert set(result["asset_label"]) == {
        "Domestic Equity",
        "Fixed Income",
    }
    assert set(result["drift_direction"]) == {
        "Overweight",
        "Underweight",
    }
    assert set(result["drift_percent"]) == {
        1.2,
        -0.8,
    }


def test_prepare_drift_chart_data_handles_empty_input() -> None:
    """Drift chart prep safely handles missing backend fields."""

    assert prepare_drift_chart_data([]).empty


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
        "Current Allocation",
        "Post-Trade Allocation",
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


def test_prepare_trade_value_chart_data_uses_signed_display_direction() -> None:
    """Trade chart prep keeps action direction with signed display bars."""

    result = prepare_trade_value_chart_data(
        [
            {
                "asset": "domestic_equity",
                "action": "SELL",
                "trade_value": -1250.25,
            },
            {
                "asset": "cash",
                "action": "BUY",
                "trade_value": 1250.25,
            },
        ]
    )

    assert set(result["action"]) == {
        "BUY",
        "SELL",
    }
    assert set(result["hover_trade_value"]) == {
        1250.25,
    }
    assert set(result["display_trade_value"]) == {
        -1250.25,
        1250.25,
    }


def test_prepare_cost_tax_impact_data_omits_zero_noise() -> None:
    """Cost/tax chart prep only keeps non-zero impact rows."""

    result = prepare_cost_tax_impact_data(
        [
            {
                "asset": "domestic_equity",
                "estimated_transaction_cost": 10.0,
                "estimated_tax": 0.0,
            }
        ]
    )

    assert list(result["impact_type"]) == ["Transaction Cost"]
    assert list(result["amount"]) == [10.0]


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
    assert result.loc[0, "trade_value"] == 5000.0


def test_rebalance_page_approval_status_uses_persisted_status() -> None:
    """Approval status display does not infer beyond persisted metadata."""

    assert (
        approval_status_label(
            required=True,
            status="pending",
        )
        == "Pending Review"
    )
    assert (
        approval_status_label(
            required=False,
            status=None,
        )
        == "Not Required"
    )


def test_approval_status_display_labels() -> None:
    """Approval status labels are professional display-only values."""

    assert (
        approval_status_label(
            required=True,
            status="PENDING",
        )
        == "Pending Review"
    )
    assert (
        approval_status_label(
            required=True,
            status="APPROVED",
        )
        == "Approved"
    )
    assert (
        approval_status_label(
            required=True,
            status="REJECTED",
        )
        == "Rejected"
    )
    assert (
        approval_status_label(
            required=True,
            status="needs_secondary_review",
        )
        == "Needs Secondary Review"
    )


def test_approval_required_display_label() -> None:
    """Approval requirement labels use safe display text."""

    assert approval_required_label(True) == "Yes"
    assert approval_required_label(False) == "No"
    assert approval_required_label(None) == "Unavailable"


def test_approval_reviewer_fallback() -> None:
    """Missing reviewers display a friendly fallback."""

    assert approval_reviewer_label(None) == "Not yet assigned"
    assert approval_reviewer_label("") == "Not yet assigned"
    assert approval_reviewer_label("advisor@example.com") == "advisor@example.com"


def test_approval_reviewed_at_fallback() -> None:
    """Missing review timestamps display a friendly fallback."""

    assert approval_reviewed_at_label(None) == "Not yet reviewed"
    assert approval_reviewed_at_label("") == "Not yet reviewed"
    assert (
        approval_reviewed_at_label("2026-08-08T12:00:00Z")
        == "2026-08-08T12:00:00Z"
    )


def test_critical_threshold_approval_reason_display() -> None:
    """Critical threshold reasons are rewritten only for display."""

    backend_reason = "Portfolio has a critical threshold breach."

    assert (
        approval_reason_label(backend_reason)
        == "Critical allocation drift requires human approval before execution."
    )
    assert backend_reason == "Portfolio has a critical threshold breach."


def test_approval_reason_handles_unknown_values_safely() -> None:
    """Unknown approval reason values do not leak null-style text."""

    assert approval_reason_label(None) == "Not available"
    assert approval_reason_label("") == "Not available"
    assert approval_reason_label("Custom review note.") == "Custom review note."


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


def test_help_text_helpers_return_professional_copy() -> None:
    """Reusable help text explains inputs, metrics, and statuses."""

    assert "portfolio" in input_help("portfolio").lower()
    assert "target" in metric_help("risk_category").lower()
    assert "target mix" in chart_help("current_vs_target").lower()
    assert "disabled" in status_help("not_configured")
    assert input_help("unknown") == ""
    assert metric_help("unknown") == ""
    assert chart_help("unknown") == ""


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


def test_pagination_summary_for_page_one() -> None:
    """Pagination displays the first backend page range."""

    page = PaginatedResponse(
        items=[],
        limit=20,
        offset=0,
        count=20,
    )

    summary = page_summary(
        page=page,
        page_index=0,
    )

    assert summary.label == "Showing 1-20"
    assert summary.page_number == 1
    assert summary.total_pages is None
    assert summary.previous_disabled is True
    assert summary.next_disabled is False
    assert page_offset(page_index=0) == 0


def test_pagination_summary_for_page_two() -> None:
    """Pagination displays the second backend page range."""

    page = PaginatedResponse(
        items=[],
        limit=20,
        offset=20,
        count=20,
    )

    summary = page_summary(
        page=page,
        page_index=1,
    )

    assert summary.label == "Showing 21-40"
    assert summary.page_number == 2
    assert summary.total_pages is None
    assert summary.previous_disabled is False
    assert summary.next_disabled is False
    assert page_offset(page_index=1) == 20


def test_pagination_summary_for_last_page() -> None:
    """A short page is treated as the last known page."""

    page = PaginatedResponse(
        items=[],
        limit=20,
        offset=40,
        count=8,
    )

    summary = page_summary(
        page=page,
        page_index=2,
    )

    assert summary.label == "Showing 41-48"
    assert summary.page_number == 3
    assert summary.total_pages is None
    assert summary.previous_disabled is False
    assert summary.next_disabled is True


def test_pagination_summary_for_single_page() -> None:
    """A single short page disables both navigation directions."""

    page = PaginatedResponse(
        items=[],
        limit=20,
        offset=0,
        count=3,
    )

    summary = page_summary(
        page=page,
        page_index=0,
    )

    assert summary.label == "Showing 1-3"
    assert summary.total_pages is None
    assert summary.previous_disabled is True
    assert summary.next_disabled is True


def test_pagination_summary_for_empty_dataset() -> None:
    """Empty pages display a friendly zero-record range."""

    page = PaginatedResponse(
        items=[],
        limit=20,
        offset=0,
        count=0,
    )

    summary = page_summary(
        page=page,
        page_index=0,
    )

    assert summary.label == "Showing 0 records"
    assert summary.total_pages is None
    assert summary.previous_disabled is True
    assert summary.next_disabled is True


def test_pagination_summary_uses_total_for_page_one() -> None:
    """Pagination displays total-aware first page metadata."""

    page = PaginatedResponse(
        items=[],
        limit=20,
        offset=0,
        count=20,
        total=500,
    )

    summary = page_summary(
        page=page,
        page_index=0,
    )

    assert summary.label == "Showing 1-20 of 500"
    assert summary.page_number == 1
    assert summary.total_pages == 25
    assert summary.previous_disabled is True
    assert summary.next_disabled is False


def test_pagination_summary_uses_total_for_page_two() -> None:
    """Pagination displays total-aware second page metadata."""

    page = PaginatedResponse(
        items=[],
        limit=20,
        offset=20,
        count=20,
        total=500,
    )

    summary = page_summary(
        page=page,
        page_index=1,
    )

    assert summary.label == "Showing 21-40 of 500"
    assert summary.page_number == 2
    assert summary.total_pages == 25
    assert summary.previous_disabled is False
    assert summary.next_disabled is False


def test_pagination_summary_uses_total_for_last_page() -> None:
    """Pagination disables next based on the backend total."""

    page = PaginatedResponse(
        items=[],
        limit=20,
        offset=500,
        count=3,
        total=503,
    )

    summary = page_summary(
        page=page,
        page_index=25,
    )

    assert summary.label == "Showing 501-503 of 503"
    assert summary.page_number == 26
    assert summary.total_pages == 26
    assert summary.previous_disabled is False
    assert summary.next_disabled is True


def test_pagination_summary_total_exact_multiple_last_page() -> None:
    """A full final page still disables Next when total is known."""

    page = PaginatedResponse(
        items=[],
        limit=20,
        offset=480,
        count=20,
        total=500,
    )

    summary = page_summary(
        page=page,
        page_index=24,
    )

    assert summary.label == "Showing 481-500 of 500"
    assert summary.total_pages == 25
    assert summary.next_disabled is True


def test_pagination_summary_total_empty_dataset() -> None:
    """A known empty dataset displays a friendly message."""

    page = PaginatedResponse(
        items=[],
        limit=20,
        offset=0,
        count=0,
        total=0,
    )

    summary = page_summary(
        page=page,
        page_index=0,
    )

    assert summary.label == "No records available"
    assert summary.page_number == 1
    assert summary.total_pages == 1
    assert summary.previous_disabled is True
    assert summary.next_disabled is True


def test_dashboard_portfolio_count_uses_total_when_available() -> None:
    """Dashboard shows true portfolio totals when the backend sends them."""

    page = PaginatedResponse(
        items=[],
        limit=20,
        offset=0,
        count=20,
        total=500,
    )

    assert DASHBOARD_PAGE._portfolio_count_label(page) == "500"
    assert (
        DASHBOARD_PAGE._portfolio_count_subtitle(page)
        == "Total listable portfolios."
    )
    assert (
        DASHBOARD_PAGE._portfolio_range_label(page)
        == "Showing 1-20 of 500"
    )


def test_dashboard_portfolio_count_falls_back_without_total() -> None:
    """Dashboard remains compatible with legacy paginated payloads."""

    page = PaginatedResponse(
        items=[],
        limit=20,
        offset=20,
        count=20,
    )

    assert DASHBOARD_PAGE._portfolio_count_label(page) == "20"
    assert (
        DASHBOARD_PAGE._portfolio_count_subtitle(page)
        == "Visible records in the current page."
    )
    assert DASHBOARD_PAGE._portfolio_range_label(page) == "Showing 21-40"


def test_portfolio_change_resets_history_page(
    monkeypatch,
) -> None:
    """Changing portfolio selection resets rebalance history paging."""

    fake_streamlit = SimpleNamespace(
        session_state={
            "history_selected_portfolio": "P1",
            "history_page": 2,
        }
    )
    monkeypatch.setitem(
        sys.modules,
        "streamlit",
        fake_streamlit,
    )

    reset_page_on_selection_change(
        selection_key="history_selected_portfolio",
        selected_value="P2",
        page_keys=("history_page",),
    )

    assert fake_streamlit.session_state["history_page"] == 0


def test_run_change_resets_trade_and_audit_pages(
    monkeypatch,
) -> None:
    """Changing run selection resets trade and audit paging."""

    fake_streamlit = SimpleNamespace(
        session_state={
            "history_selected_run": "RUN1",
            "trade_page": 3,
            "audit_page": 4,
        }
    )
    monkeypatch.setitem(
        sys.modules,
        "streamlit",
        fake_streamlit,
    )

    reset_page_on_selection_change(
        selection_key="history_selected_run",
        selected_value="RUN2",
        page_keys=(
            "trade_page",
            "audit_page",
        ),
    )

    assert fake_streamlit.session_state["trade_page"] == 0
    assert fake_streamlit.session_state["audit_page"] == 0


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
    assert prepare_drawdown_chart_data(
        [
            {
                "date": "initial",
                "portfolio_value": 100_000.0,
            }
        ]
    ).empty


def test_prepare_backtest_drawdown_data_uses_returned_series() -> None:
    """Drawdown prep only uses a backend-provided drawdown series."""

    result = prepare_drawdown_chart_data(
        [
            {
                "date": "1",
                "drawdown": -0.02,
            }
        ]
    )

    assert list(result["period_label"]) == ["1"]
    assert list(result["drawdown_percent"]) == [-2.0]
    assert list(prepare_backtest_drawdown_data(
        [
            {
                "date": "1",
                "drawdown": -0.02,
            }
        ]
    )["drawdown_percent"]) == [-2.0]


def test_backtest_strategy_descriptions_are_humanized() -> None:
    """Backtesting strategy descriptions match the selected strategy."""

    assert BACKTESTING_PAGE.backtest_strategy_description(
        "Buy & Hold"
    ) == (
        "Invest once and allow the portfolio allocation to move naturally "
        "with market performance."
    )
    assert BACKTESTING_PAGE.backtest_strategy_description(
        "Threshold Rebalancing"
    ) == (
        "Rebalance the portfolio when allocation drift exceeds the "
        "configured threshold."
    )
    assert BACKTESTING_PAGE.backtest_strategy_description(
        "Strategy Comparison"
    ) == (
        "Compare Buy & Hold and Threshold Rebalancing across return, "
        "risk, and trading activity."
    )


def test_unknown_backtest_strategy_description_is_empty() -> None:
    """Unknown strategy descriptions safely return no copy."""

    assert BACKTESTING_PAGE.backtest_strategy_description("Other") == ""


def test_backtest_metric_help_descriptions() -> None:
    """Backtest metric help uses centralized educational copy."""

    assert backtest_metric_help("sharpe_ratio") == (
        "How much return the strategy generated relative to the risk "
        "taken. Higher is generally better when comparing similar "
        "strategies."
    )
    assert backtest_metric_help("maximum_drift") == (
        "The largest difference observed between the portfolio's current "
        "allocation and its target allocation."
    )
    assert backtest_metric_help("unknown_metric") == ""


def test_backtest_metric_formatting_reuses_display_helpers() -> None:
    """Backtest values are formatted for display only."""

    assert format_percentage(0.1245) == "12.45%"
    assert format_percentage(0.183) == "18.30%"
    assert format_currency(1234.567) == "$1,234.57"


def test_backtest_strategy_change_invalidates_old_result() -> None:
    """A result cannot render after the selected strategy changes."""

    assert BACKTESTING_PAGE.backtest_strategy_changed(
        previous_strategy="Buy & Hold",
        selected_strategy="Threshold Rebalancing",
    )
    assert not BACKTESTING_PAGE.should_render_backtest_result(
        selected_strategy="Threshold Rebalancing",
        result_strategy="Buy & Hold",
        result={
            "metrics": {},
        },
    )


def test_backtest_sections_are_mutually_exclusive_for_buy_and_hold() -> None:
    """Buy & Hold view identity excludes other strategy sections."""

    sections = BACKTESTING_PAGE.backtest_view_sections(
        strategy="Buy & Hold",
        has_result=False,
    )

    assert "Buy & Hold Description" in sections
    assert "Buy & Hold Empty State" in sections
    assert not any(
        section.startswith("Threshold Rebalancing")
        or section.startswith("Strategy Comparison")
        for section in sections
    )


def test_backtest_sections_are_mutually_exclusive_for_threshold() -> None:
    """Threshold view identity excludes Buy & Hold sections."""

    sections = BACKTESTING_PAGE.backtest_view_sections(
        strategy="Threshold Rebalancing",
        has_result=True,
    )

    assert "Threshold Rebalancing Description" in sections
    assert "Threshold Rebalancing Results" in sections
    assert not any(
        section.startswith("Buy & Hold")
        or section.startswith("Strategy Comparison")
        for section in sections
    )


def test_backtest_sections_are_mutually_exclusive_for_comparison() -> None:
    """Comparison view identity excludes single-strategy sections."""

    sections = BACKTESTING_PAGE.backtest_view_sections(
        strategy="Strategy Comparison",
        has_result=False,
    )

    assert "Strategy Comparison Description" in sections
    assert "Strategy Comparison Empty State" in sections
    assert not any(
        section.startswith("Buy & Hold")
        or section.startswith("Threshold Rebalancing")
        for section in sections
    )


def test_backtest_state_keys_include_legacy_strategy_results() -> None:
    """Strategy changes clear current and legacy result keys."""

    assert BACKTESTING_PAGE.backtest_state_keys_to_clear() == (
        "backtest_result",
        "backtest_result_strategy",
        "buy_and_hold_result",
        "threshold_result",
        "comparison_result",
    )


def test_same_backtest_strategy_preserves_valid_result() -> None:
    """A result can render when it belongs to the selected strategy."""

    assert not BACKTESTING_PAGE.backtest_strategy_changed(
        previous_strategy="Buy & Hold",
        selected_strategy="Buy & Hold",
    )
    assert BACKTESTING_PAGE.should_render_backtest_result(
        selected_strategy="Buy & Hold",
        result_strategy="Buy & Hold",
        result={
            "metrics": {},
        },
    )


def test_backtest_result_identity_rejects_empty_result() -> None:
    """A missing result is never considered renderable."""

    assert not BACKTESTING_PAGE.should_render_backtest_result(
        selected_strategy="Buy & Hold",
        result_strategy="Buy & Hold",
        result=None,
    )


def test_prepare_backtest_strategy_history_data_overlays_histories() -> None:
    """Strategy comparison history keeps shared units on one chart."""

    result = prepare_backtest_strategy_history_data(
        buy_and_hold_history=[
            {
                "date": "initial",
                "portfolio_value": 100_000.0,
            }
        ],
        threshold_history=[
            {
                "date": "initial",
                "portfolio_value": 100_000.0,
            },
            {
                "date": "1",
                "portfolio_value": 101_000.0,
            },
        ],
    )

    assert set(result["strategy"]) == {
        "Buy & Hold",
        "Threshold Rebalancing",
    }
    assert list(result["portfolio_value"]) == [
        100_000.0,
        100_000.0,
        101_000.0,
    ]


def test_prepare_backtest_strategy_history_data_handles_empty_inputs() -> None:
    """Comparison history prep safely handles missing optional histories."""

    assert prepare_backtest_strategy_history_data(
        buy_and_hold_history=[],
        threshold_history=[],
    ).empty


def test_prepare_backtest_strategy_drawdown_data_overlays_histories() -> None:
    """Strategy comparison drawdown keeps backend series on one scale."""

    result = prepare_backtest_strategy_drawdown_data(
        buy_and_hold_drawdown=[
            {
                "period": 0,
                "drawdown": 0.0,
            },
            {
                "period": 1,
                "drawdown": -0.04,
            },
        ],
        threshold_drawdown=[
            {
                "period": 0,
                "drawdown": 0.0,
            },
            {
                "period": 1,
                "drawdown": -0.02,
            },
        ],
    )

    assert set(result["strategy"]) == {
        "Buy & Hold",
        "Threshold Rebalancing",
    }
    assert set(result["drawdown_percent"]) == {
        0.0,
        -4.0,
        -2.0,
    }


def test_prepare_backtest_strategy_drawdown_data_handles_empty_inputs() -> None:
    """Comparison drawdown chart omits empty backend drawdown histories."""

    assert prepare_backtest_strategy_drawdown_data(
        buy_and_hold_drawdown=[],
        threshold_drawdown=[],
    ).empty


def test_prepare_strategy_comparison_chart_data_groups_scales() -> None:
    """Strategy comparison chart data separates incompatible scales."""

    result = prepare_strategy_comparison_chart_data(
        {
            "buy_and_hold": {
                "total_return": 0.10,
                "volatility": 0.08,
                "sharpe_ratio": 0.75,
                "transaction_costs": 0.0,
                "number_of_rebalances": 0,
            },
            "threshold_rebalancing": {
                "total_return": 0.12,
                "volatility": 0.07,
                "sharpe_ratio": 0.90,
                "transaction_costs": 10.0,
                "number_of_rebalances": 2,
            },
        }
    )

    assert set(result) == {
        "return",
        "risk",
        "ratio",
        "cost",
        "count",
    }
    assert set(result["return"]["metric"]) == {"Total Return"}
    assert set(result["risk"]["metric"]) == {"Volatility"}
    assert set(result["count"]["value"]) == {0.0, 2.0}


def test_prepare_strategy_comparison_chart_data_handles_missing_metrics() -> None:
    """Comparison chart prep omits optional metrics that are absent."""

    result = prepare_strategy_comparison_chart_data(
        {
            "buy_and_hold": {
                "total_return": 0.10,
            },
            "threshold_rebalancing": {
                "total_return": 0.12,
            },
        }
    )

    assert set(result) == {
        "return",
    }
    assert set(result["return"]["metric"]) == {"Total Return"}


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
