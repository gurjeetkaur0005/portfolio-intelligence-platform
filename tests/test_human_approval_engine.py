from __future__ import annotations

import math

import pandas as pd
import pytest

from src.override.human_approval_engine import (
    evaluate_trade_approvals,
)


@pytest.fixture
def valid_trades() -> pd.DataFrame:
    """
    Create valid trade data for approval-engine tests.
    """

    return pd.DataFrame(
        {
            "portfolio_id": [
                "P00001",
                "P00002",
                "P00003",
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
                -40_000.0,
                0.0,
            ],
            "prior_approval_required": [
                False,
                False,
                False,
            ],
            "threshold_severity": [
                "medium",
                "high",
                "none",
            ],
        }
    )


def test_small_trade_does_not_require_approval(
    valid_trades: pd.DataFrame,
) -> None:
    """
    A small valid trade should be automatically approved.
    """

    result = evaluate_trade_approvals(
        trades=valid_trades.iloc[[0]],
        approval_threshold=100_000.0,
    )

    assert bool(result.iloc[0]["approval_required"]) is False
    assert result.iloc[0]["approval_status"] == "NOT_REQUIRED"
    assert (
        result.iloc[0]["approval_reason"]
        == "Trade satisfies automatic approval rules."
    )


def test_hold_trade_does_not_require_approval(
    valid_trades: pd.DataFrame,
) -> None:
    """
    HOLD actions should not require execution approval.
    """

    result = evaluate_trade_approvals(
        trades=valid_trades.iloc[[2]],
        approval_threshold=100_000.0,
    )

    assert bool(result.iloc[0]["approval_required"]) is False
    assert result.iloc[0]["approval_status"] == "NOT_REQUIRED"
    assert (
        result.iloc[0]["approval_reason"]
        == "No trade execution is required."
    )


def test_client_prior_approval_requirement(
    valid_trades: pd.DataFrame,
) -> None:
    """
    A client approval requirement should make the trade pending.
    """

    trades = valid_trades.iloc[[0]].copy()
    trades.loc[:, "prior_approval_required"] = True

    result = evaluate_trade_approvals(
        trades=trades,
        approval_threshold=100_000.0,
    )

    assert bool(result.iloc[0]["approval_required"]) is True
    assert result.iloc[0]["approval_status"] == "PENDING"
    assert (
        "Client requires prior approval"
        in result.iloc[0]["approval_reason"]
    )


def test_large_buy_trade_requires_approval(
    valid_trades: pd.DataFrame,
) -> None:
    """
    A BUY at or above the threshold should require approval.
    """

    trades = valid_trades.iloc[[0]].copy()
    trades.loc[:, "trade_value"] = 150_000.0

    result = evaluate_trade_approvals(
        trades=trades,
        approval_threshold=100_000.0,
    )

    assert bool(result.iloc[0]["approval_required"]) is True
    assert (
        "Trade value exceeds the approval threshold"
        in result.iloc[0]["approval_reason"]
    )


def test_large_sell_trade_requires_approval(
    valid_trades: pd.DataFrame,
) -> None:
    """
    A large negative SELL value should use its absolute magnitude.
    """

    trades = valid_trades.iloc[[1]].copy()
    trades.loc[:, "trade_value"] = -150_000.0

    result = evaluate_trade_approvals(
        trades=trades,
        approval_threshold=100_000.0,
    )

    assert bool(result.iloc[0]["approval_required"]) is True
    assert (
        "Trade value exceeds the approval threshold"
        in result.iloc[0]["approval_reason"]
    )


def test_trade_equal_to_threshold_requires_approval(
    valid_trades: pd.DataFrame,
) -> None:
    """
    The approval threshold is inclusive.
    """

    trades = valid_trades.iloc[[0]].copy()
    trades.loc[:, "trade_value"] = 100_000.0

    result = evaluate_trade_approvals(
        trades=trades,
        approval_threshold=100_000.0,
    )

    assert bool(result.iloc[0]["approval_required"]) is True


def test_critical_threshold_requires_approval(
    valid_trades: pd.DataFrame,
) -> None:
    """
    A critical threshold breach should require human review.
    """

    trades = valid_trades.iloc[[0]].copy()
    trades.loc[:, "threshold_severity"] = "critical"

    result = evaluate_trade_approvals(
        trades=trades,
        approval_threshold=100_000.0,
    )

    assert bool(result.iloc[0]["approval_required"]) is True
    assert (
        "Portfolio has a critical threshold breach"
        in result.iloc[0]["approval_reason"]
    )


def test_multiple_approval_reasons_are_combined(
    valid_trades: pd.DataFrame,
) -> None:
    """
    All applicable approval reasons should be returned.
    """

    trades = valid_trades.iloc[[0]].copy()
    trades.loc[:, "prior_approval_required"] = True
    trades.loc[:, "trade_value"] = 200_000.0
    trades.loc[:, "threshold_severity"] = "critical"

    result = evaluate_trade_approvals(
        trades=trades,
        approval_threshold=100_000.0,
    )

    reason = result.iloc[0]["approval_reason"]

    assert "Client requires prior approval" in reason
    assert "Trade value exceeds the approval threshold" in reason
    assert "Portfolio has a critical threshold breach" in reason


def test_output_contains_approval_columns(
    valid_trades: pd.DataFrame,
) -> None:
    """
    The result should contain all approval decision columns.
    """

    result = evaluate_trade_approvals(
        trades=valid_trades,
    )

    expected_columns = {
        "approval_required",
        "approval_status",
        "approval_reason",
    }

    assert expected_columns.issubset(result.columns)


def test_missing_required_column_raises_error(
    valid_trades: pd.DataFrame,
) -> None:
    """
    Missing required trade data should raise a ValueError.
    """

    invalid_trades = valid_trades.drop(
        columns=["trade_value"],
    )

    with pytest.raises(
        ValueError,
        match="Trades are missing required columns",
    ):
        evaluate_trade_approvals(invalid_trades)


def test_invalid_action_raises_error(
    valid_trades: pd.DataFrame,
) -> None:
    """
    Unsupported trade actions should be rejected.
    """

    invalid_trades = valid_trades.copy()
    invalid_trades.loc[0, "action"] = "PURCHASE"

    with pytest.raises(
        ValueError,
        match="Invalid trade actions",
    ):
        evaluate_trade_approvals(invalid_trades)


@pytest.mark.parametrize(
    "invalid_trade_value",
    [
        "invalid",
        None,
        math.inf,
        -math.inf,
        math.nan,
    ],
)
def test_invalid_trade_values_raise_error(
    valid_trades: pd.DataFrame,
    invalid_trade_value: object,
) -> None:
    """
    Non-numeric and non-finite trade values should be rejected.
    """

    invalid_trades = valid_trades.copy()

    # Convert to object so intentionally invalid test values
    # can be inserted into the float column.
    invalid_trades["trade_value"] = invalid_trades[
        "trade_value"
    ].astype(object)

    invalid_trades.loc[
        0,
        "trade_value",
    ] = invalid_trade_value

    with pytest.raises(
        ValueError,
        match="finite numeric values",
    ):
        evaluate_trade_approvals(invalid_trades)


@pytest.mark.parametrize(
    ("action", "trade_value", "expected_message"),
    [
        (
            "BUY",
            -100.0,
            "BUY trades must have a positive trade value",
        ),
        (
            "SELL",
            100.0,
            "SELL trades must have a negative trade value",
        ),
        (
            "HOLD",
            100.0,
            "HOLD trades must have a zero trade value",
        ),
    ],
)
def test_invalid_action_sign_conventions_raise_error(
    valid_trades: pd.DataFrame,
    action: str,
    trade_value: float,
    expected_message: str,
) -> None:
    """
    Trade-value signs must agree with their corresponding actions.
    """

    invalid_trades = valid_trades.iloc[[0]].copy()
    invalid_trades.loc[:, "action"] = action
    invalid_trades.loc[:, "trade_value"] = trade_value

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        evaluate_trade_approvals(invalid_trades)


@pytest.mark.parametrize(
    "invalid_flag",
    [
        "Yes",
        "False",
        1,
        0,
        None,
    ],
)
def test_invalid_prior_approval_flag_raises_error(
    valid_trades: pd.DataFrame,
    invalid_flag: object,
) -> None:
    """
    Approval flags must be actual boolean values.
    """

    invalid_trades = valid_trades.copy()

    # Convert to object so intentionally invalid values
    # can be inserted into the boolean column.
    invalid_trades["prior_approval_required"] = (
        invalid_trades[
            "prior_approval_required"
        ].astype(object)
    )

    invalid_trades.loc[
        0,
        "prior_approval_required",
    ] = invalid_flag

    with pytest.raises(
        ValueError,
        match="must contain only boolean values",
    ):
        evaluate_trade_approvals(invalid_trades)


def test_invalid_threshold_severity_raises_error(
    valid_trades: pd.DataFrame,
) -> None:
    """
    Unknown threshold severity values should be rejected.
    """

    invalid_trades = valid_trades.copy()
    invalid_trades.loc[0, "threshold_severity"] = "urgent"

    with pytest.raises(
        ValueError,
        match="Invalid threshold severities",
    ):
        evaluate_trade_approvals(invalid_trades)


@pytest.mark.parametrize(
    "invalid_threshold",
    [
        0,
        -1,
        "invalid",
        True,
        math.inf,
        math.nan,
    ],
)
def test_invalid_approval_threshold_raises_error(
    valid_trades: pd.DataFrame,
    invalid_threshold: object,
) -> None:
    """
    The approval threshold must be a positive finite number.
    """

    with pytest.raises(
        (TypeError, ValueError),
    ):
        evaluate_trade_approvals(
            trades=valid_trades,
            approval_threshold=invalid_threshold,
        )


def test_input_dataframe_is_not_modified(
    valid_trades: pd.DataFrame,
) -> None:
    """
    The approval engine should not mutate its input DataFrame.
    """

    original_trades = valid_trades.copy(deep=True)

    evaluate_trade_approvals(
        trades=valid_trades,
    )

    pd.testing.assert_frame_equal(
        valid_trades,
        original_trades,
    )