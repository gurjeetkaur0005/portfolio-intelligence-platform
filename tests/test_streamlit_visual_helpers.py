from __future__ import annotations

from streamlit_app.components.charts import (
    _format_asset_label,
    prepare_allocation_data,
    prepare_holding_value_data,
    prepare_portfolio_value_data,
    prepare_rebalance_allocation_comparison_data,
)
from streamlit_app.components.status import (
    approval_status_label,
    status_label,
    status_tone,
)
from streamlit_app.components.tables import (
    prepare_holdings_table_data,
    prepare_rebalance_trade_table_data,
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
    assert status_tone("pending") == "warning"
    assert status_tone("failed") == "negative"
    assert status_tone("unknown") == "neutral"


def test_prepare_allocation_data_handles_empty_input() -> None:
    """Allocation preparation safely handles empty data."""

    assert prepare_allocation_data([]).empty


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
                "current_value": 250.0,
                "cost_basis": 200.0,
            }
        ]
    )

    assert result.loc[0, "asset"] == "Fixed Income"
    assert result.loc[0, "current_weight"] == 25.0


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
