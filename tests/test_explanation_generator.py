import copy

import pandas as pd
import pytest

from src.explanations.explanation_generator import (
    generate_trade_explanations,
)


def make_trade_list(
    **overrides,
) -> pd.DataFrame:
    """Build a valid explanation input DataFrame."""

    row = {
        "portfolio_id": "P001",
        "asset": "domestic_equity",
        "action": "BUY",
        "current_weight": 0.20,
        "trade_weight": 0.05,
        "post_trade_weight": 0.25,
        "trade_value": 50_000.0,
        "transaction_cost": 100.0,
        "estimated_tax_liability": 0.0,
        "threshold_breached": True,
        "threshold_severity": "medium",
        "breach_ratio": 1.25,
    }
    row.update(overrides)

    return pd.DataFrame([row])


@pytest.mark.parametrize(
    (
        "action",
        "trade_weight",
        "trade_value",
        "client_phrase",
    ),
    [
        ("BUY", 0.05, 50_000.0, "is increasing"),
        ("SELL", -0.05, -50_000.0, "is decreasing"),
        ("HOLD", 0.0, 0.0, "remains unchanged"),
    ],
)
def test_valid_actions_generate_explanations(
    action: str,
    trade_weight: float,
    trade_value: float,
    client_phrase: str,
) -> None:
    trade_list = make_trade_list(
        action=action,
        trade_weight=trade_weight,
        trade_value=trade_value,
        post_trade_weight=0.15 if action == "SELL" else 0.20,
        transaction_cost=0.0 if action == "HOLD" else 100.0,
    )

    result = generate_trade_explanations(trade_list)

    assert client_phrase in result.loc[0, "client_explanation"]
    assert "client_explanation" in result.columns
    assert "advisor_explanation" in result.columns
    assert "compliance_explanation" in result.columns


def test_sell_trade_value_is_formatted_with_clear_negative_sign() -> None:
    trade_list = make_trade_list(
        action="SELL",
        trade_weight=-0.05,
        trade_value=-50_000.0,
        post_trade_weight=0.15,
    )

    result = generate_trade_explanations(trade_list)

    assert "-$50,000.00" in result.loc[0, "advisor_explanation"]
    assert "-$50,000.00" in result.loc[0, "compliance_explanation"]


def test_hold_explanation_reads_naturally() -> None:
    trade_list = make_trade_list(
        action="HOLD",
        trade_weight=0.0,
        trade_value=0.0,
        transaction_cost=0.0,
        post_trade_weight=0.20,
        threshold_breached=False,
        threshold_severity="none",
        breach_ratio=1.0,
    )

    result = generate_trade_explanations(trade_list)
    explanation = result.loc[0, "client_explanation"]

    assert "remains unchanged at 20.00%" in explanation
    assert "changing from" not in explanation


def test_missing_required_columns_raise_value_error() -> None:
    trade_list = make_trade_list().drop(
        columns=["threshold_severity"]
    )

    with pytest.raises(
        ValueError,
        match="Trade list is missing required columns",
    ):
        generate_trade_explanations(trade_list)


def test_invalid_action_raises_value_error() -> None:
    trade_list = make_trade_list(action="REBUY")

    with pytest.raises(
        ValueError,
        match="Action must be one of BUY, SELL, or HOLD",
    ):
        generate_trade_explanations(trade_list)


def test_missing_numeric_values_raise_value_error() -> None:
    trade_list = make_trade_list(trade_value=None)

    with pytest.raises(
        ValueError,
        match="must not contain missing numeric values",
    ):
        generate_trade_explanations(trade_list)


@pytest.mark.parametrize(
    "column",
    [
        "current_weight",
        "trade_weight",
        "post_trade_weight",
        "trade_value",
        "transaction_cost",
        "estimated_tax_liability",
        "breach_ratio",
    ],
)
def test_infinite_values_raise_value_error(
    column: str,
) -> None:
    trade_list = make_trade_list()
    trade_list.loc[0, column] = float("inf")

    with pytest.raises(
        ValueError,
        match="must be finite values",
    ):
        generate_trade_explanations(trade_list)


def test_negative_infinity_raises_value_error() -> None:
    trade_list = make_trade_list(trade_value=float("-inf"))

    with pytest.raises(
        ValueError,
        match="must be finite values",
    ):
        generate_trade_explanations(trade_list)


def test_invalid_numeric_string_raises_value_error() -> None:
    trade_list = make_trade_list(trade_value="not-a-number")

    with pytest.raises(
        ValueError,
        match="numeric inputs must be valid numbers",
    ):
        generate_trade_explanations(trade_list)


@pytest.mark.parametrize(
    "overrides",
    [
        {"action": "BUY", "trade_weight": -0.05},
        {"action": "BUY", "trade_value": -50_000.0},
        {"action": "SELL", "trade_weight": 0.05},
        {"action": "SELL", "trade_value": 50_000.0},
        {"action": "HOLD", "trade_weight": 0.01},
        {"action": "HOLD", "trade_value": 1.0},
    ],
)
def test_invalid_trade_signs_raise_value_error(
    overrides: dict,
) -> None:
    trade_list = make_trade_list(**overrides)

    with pytest.raises(
        ValueError,
        match="Trade action is inconsistent",
    ):
        generate_trade_explanations(trade_list)


def test_tiny_hold_values_are_allowed() -> None:
    trade_list = make_trade_list(
        action="HOLD",
        trade_weight=1e-10,
        trade_value=-1e-10,
        post_trade_weight=0.20,
        threshold_breached=False,
        threshold_severity="none",
        breach_ratio=1.0,
    )

    result = generate_trade_explanations(trade_list)

    assert "remains unchanged" in result.loc[0, "client_explanation"]


@pytest.mark.parametrize(
    "threshold_breached",
    [
        "true",
        1,
        None,
    ],
)
def test_invalid_threshold_flag_raises_value_error(
    threshold_breached,
) -> None:
    trade_list = make_trade_list(
        threshold_breached=threshold_breached
    )

    with pytest.raises(
        ValueError,
        match="Threshold",
    ):
        generate_trade_explanations(trade_list)


def test_negative_breach_ratio_raises_value_error() -> None:
    trade_list = make_trade_list(
        threshold_breached=False,
        threshold_severity="none",
        breach_ratio=-0.10,
    )

    with pytest.raises(
        ValueError,
        match="Breach ratio must not be negative",
    ):
        generate_trade_explanations(trade_list)


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "threshold_breached": False,
            "threshold_severity": "medium",
            "breach_ratio": 1.25,
        },
        {
            "threshold_breached": True,
            "threshold_severity": "none",
            "breach_ratio": 1.25,
        },
        {
            "threshold_breached": True,
            "threshold_severity": "high",
            "breach_ratio": 1.25,
        },
    ],
)
def test_inconsistent_threshold_severity_raises_value_error(
    overrides: dict,
) -> None:
    trade_list = make_trade_list(**overrides)

    with pytest.raises(
        ValueError,
        match="Threshold breached, severity, and breach ratio",
    ):
        generate_trade_explanations(trade_list)


@pytest.mark.parametrize(
    (
        "breach_ratio",
        "threshold_breached",
        "threshold_severity",
    ),
    [
        (1.0, False, "none"),
        (1.000001, True, "medium"),
        (1.5, True, "high"),
        (1.999999, True, "high"),
        (2.0, True, "critical"),
    ],
)
def test_threshold_boundary_values_are_valid(
    breach_ratio: float,
    threshold_breached: bool,
    threshold_severity: str,
) -> None:
    trade_list = make_trade_list(
        breach_ratio=breach_ratio,
        threshold_breached=threshold_breached,
        threshold_severity=threshold_severity,
    )

    result = generate_trade_explanations(trade_list)

    assert result.loc[0, "threshold_severity"] == threshold_severity


def test_input_dataframe_is_not_mutated() -> None:
    trade_list = make_trade_list()
    original_records = copy.deepcopy(
        trade_list.to_dict("records")
    )

    generate_trade_explanations(trade_list)

    assert trade_list.to_dict("records") == original_records


def test_expected_audience_specific_explanation_content() -> None:
    trade_list = make_trade_list(
        action="SELL",
        trade_weight=-0.05,
        trade_value=-50_000.0,
        post_trade_weight=0.15,
        transaction_cost=100.0,
        estimated_tax_liability=500.0,
        threshold_severity="critical",
        breach_ratio=2.0,
    )

    result = generate_trade_explanations(trade_list)

    client = result.loc[0, "client_explanation"]
    advisor = result.loc[0, "advisor_explanation"]
    compliance = result.loc[0, "compliance_explanation"]

    assert "domestic equity" in client
    assert "decreasing" in client
    assert "$100.00" in client
    assert "$500.00" in client
    assert "Portfolio P001" in advisor
    assert "breach ratio of 2.00" in advisor
    assert "Estimated signed trade value: -$50,000.00" in advisor
    assert "Trade recommendation generated by the portfolio optimizer" in (
        compliance
    )
    assert "Breach ratio: 2.000000" in compliance
    assert "Recommended trade weight: -0.050000" in compliance
