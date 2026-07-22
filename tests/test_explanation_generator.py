import pandas as pd
import pytest

from src.explanations.explanation_generator import (
    generate_trade_explanations,
)


@pytest.fixture
def trade_list() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "portfolio_id": ["P001", "P001", "P001"],
            "asset": [
                "domestic_equity",
                "fixed_income",
                "cash",
            ],
            "action": ["BUY", "SELL", "HOLD"],
            "current_weight": [0.20, 0.50, 0.30],
            "trade_weight": [0.05, -0.05, 0.00],
            "post_trade_weight": [0.25, 0.45, 0.30],
            "trade_value": [50000, -50000, 0],
            "transaction_cost": [100, 100, 0],
            "estimated_tax_liability": [0, 500, 0],
        }
    )


def test_all_explanation_columns_are_created(
    trade_list: pd.DataFrame,
) -> None:
    result = generate_trade_explanations(trade_list)

    for column in [
        "client_explanation",
        "advisor_explanation",
        "compliance_explanation",
    ]:
        assert column in result.columns


def test_buy_sell_and_hold_use_appropriate_wording(
    trade_list: pd.DataFrame,
) -> None:
    result = generate_trade_explanations(trade_list)

    assert "increasing" in result.loc[0, "client_explanation"]
    assert "decreasing" in result.loc[1, "client_explanation"]
    assert "unchanged" in result.loc[2, "client_explanation"]
    assert "Action: BUY" in result.loc[0, "advisor_explanation"]
    assert "Action: SELL" in result.loc[1, "advisor_explanation"]
    assert "Action: HOLD" in result.loc[2, "advisor_explanation"]


def test_asset_names_with_underscores_are_readable(
    trade_list: pd.DataFrame,
) -> None:
    result = generate_trade_explanations(trade_list)

    assert "domestic equity" in result.loc[0, "client_explanation"]
    assert "fixed income" in result.loc[1, "advisor_explanation"]


def test_transaction_cost_and_tax_appear_in_client_and_advisor(
    trade_list: pd.DataFrame,
) -> None:
    result = generate_trade_explanations(trade_list)

    assert "$100.00" in result.loc[1, "client_explanation"]
    assert "$500.00" in result.loc[1, "client_explanation"]
    assert "Transaction cost: $100.00" in (
        result.loc[1, "advisor_explanation"]
    )
    assert "Tax liability: $500.00" in (
        result.loc[1, "advisor_explanation"]
    )


def test_compliance_explanation_is_traceable(
    trade_list: pd.DataFrame,
) -> None:
    result = generate_trade_explanations(trade_list)
    explanation = result.loc[1, "compliance_explanation"]

    assert "Recommendation source: portfolio optimizer" in explanation
    assert "Portfolio ID: P001" in explanation
    assert "Current weight: 0.500000" in explanation
    assert "Trade weight: -0.050000" in explanation
    assert "Post-trade weight: 0.450000" in explanation
    assert "Signed trade value: -50000.00" in explanation


def test_missing_columns_raise_value_error(
    trade_list: pd.DataFrame,
) -> None:
    invalid_trade_list = trade_list.drop(
        columns=["transaction_cost"]
    )

    with pytest.raises(
        ValueError,
        match="Trade list is missing required columns",
    ):
        generate_trade_explanations(invalid_trade_list)


def test_invalid_actions_raise_value_error(
    trade_list: pd.DataFrame,
) -> None:
    invalid_trade_list = trade_list.copy()
    invalid_trade_list.loc[0, "action"] = "REBUY"

    with pytest.raises(
        ValueError,
        match="Action must be one of BUY, SELL, or HOLD",
    ):
        generate_trade_explanations(invalid_trade_list)


def test_missing_numeric_values_raise_value_error(
    trade_list: pd.DataFrame,
) -> None:
    invalid_trade_list = trade_list.copy()
    invalid_trade_list.loc[0, "trade_value"] = None

    with pytest.raises(
        ValueError,
        match="Explanation inputs must not contain missing numeric values",
    ):
        generate_trade_explanations(invalid_trade_list)


def test_non_finite_weights_raise_value_error(
    trade_list: pd.DataFrame,
) -> None:
    invalid_trade_list = trade_list.copy()
    invalid_trade_list.loc[0, "current_weight"] = float("inf")

    with pytest.raises(
        ValueError,
        match="Weights must be finite numeric values",
    ):
        generate_trade_explanations(invalid_trade_list)


def test_input_dataframe_remains_unchanged(
    trade_list: pd.DataFrame,
) -> None:
    original = trade_list.copy(deep=True)

    generate_trade_explanations(trade_list)

    pd.testing.assert_frame_equal(trade_list, original)
