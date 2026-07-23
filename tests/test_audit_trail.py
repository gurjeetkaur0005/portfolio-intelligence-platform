from __future__ import annotations

import pandas as pd
import pytest

from src.audit.audit_trail import create_audit_trail


@pytest.fixture
def sample_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "portfolio_id": [
                "P00001",
                "P00001",
                "P00002",
            ],
            "asset": [
                "domestic_equity",
                "fixed_income",
                "cash",
            ],
            "action": [
                "BUY",
                "SELL",
                "HOLD",
            ],
            "trade_value": [
                50_000.0,
                -25_000.0,
                0.0,
            ],
            "transaction_cost": [
                100.0,
                50.0,
                0.0,
            ],
            "estimated_tax_liability": [
                0.0,
                2_000.0,
                0.0,
            ],
            "final_trigger_type": [
                "threshold",
                "event",
                "calendar",
            ],
            "final_priority": [
                "high",
                "critical",
                "low",
            ],
            "approval_required": [
                False,
                True,
                False,
            ],
            "approval_status": [
                "NOT_REQUIRED",
                "PENDING",
                "NOT_REQUIRED",
            ],
            "approval_reason": [
                "Trade does not require approval.",
                "Critical event requires human approval.",
                "No trade execution is required.",
            ],
        }
    )


def test_create_audit_trail_adds_audit_columns(
    sample_trades: pd.DataFrame,
) -> None:
    result = create_audit_trail(sample_trades)

    assert "audit_id" in result.columns
    assert "audit_timestamp" in result.columns

def test_create_audit_trail_generates_expected_ids(
    sample_trades: pd.DataFrame,
) -> None:
    result = create_audit_trail(sample_trades)

    assert result["audit_id"].tolist() == [
        "AUD000001",
        "AUD000002",
        "AUD000003",
    ]

def test_create_audit_trail_generates_unique_ids(
    sample_trades: pd.DataFrame,
) -> None:
    result = create_audit_trail(sample_trades)

    assert result["audit_id"].is_unique


def test_create_audit_trail_populates_timestamp(
    sample_trades: pd.DataFrame,
) -> None:
    result = create_audit_trail(sample_trades)

    assert result["audit_timestamp"].notna().all()


def test_create_audit_trail_does_not_modify_input(
    sample_trades: pd.DataFrame,
) -> None:
    original = sample_trades.copy(deep=True)

    create_audit_trail(sample_trades)

    pd.testing.assert_frame_equal(
        sample_trades,
        original,
    )

def test_create_audit_trail_requires_dataframe() -> None:
    with pytest.raises(
        TypeError,
        match="pandas DataFrame",
    ):
        create_audit_trail([])

def test_create_audit_trail_preserves_row_count(
    sample_trades: pd.DataFrame,
) -> None:
    result = create_audit_trail(sample_trades)

    assert len(result) == len(sample_trades)

def test_create_audit_trail_audit_id_is_first_column(
    sample_trades: pd.DataFrame,
) -> None:
    result = create_audit_trail(sample_trades)

    assert result.columns[0] == "audit_id"


def test_create_audit_trail_timestamp_is_second_column(
    sample_trades: pd.DataFrame,
) -> None:
    result = create_audit_trail(sample_trades)

    assert result.columns[1] == "audit_timestamp"


def test_create_audit_trail_empty_dataframe() -> None:
    empty_trades = pd.DataFrame(
        columns=[
            "portfolio_id",
            "asset",
            "action",
            "trade_value",
            "transaction_cost",
            "estimated_tax_liability",
            "final_trigger_type",
            "final_priority",
            "approval_required",
            "approval_status",
            "approval_reason",
        ]
    )

    result = create_audit_trail(empty_trades)

    assert result.empty
