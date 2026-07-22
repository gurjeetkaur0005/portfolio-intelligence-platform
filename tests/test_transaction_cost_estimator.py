import pandas as pd
import pytest

from src.execution.transaction_cost_estimator import (
    estimate_transaction_costs,
)


@pytest.fixture
def trade_list() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asset": [
                "Domestic Equity",
                "Fixed Income",
                "Cash",
            ],
            "action": [
                "SELL",
                "BUY",
                "HOLD",
            ],
            "trade_weight": [
                -0.05,
                0.03,
                0.00,
            ],
        }
    )


def test_returns_dataframe(
    trade_list: pd.DataFrame,
) -> None:
    result = estimate_transaction_costs(
        trade_list=trade_list,
        portfolio_value=1_000_000,
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )


def test_buy_produces_positive_trade_value(
    trade_list: pd.DataFrame,
) -> None:
    result = estimate_transaction_costs(
        trade_list=trade_list,
        portfolio_value=1_000_000,
    )

    buy_trade = result.loc[result["action"] == "BUY"].iloc[0]

    assert buy_trade["trade_value"] == 30_000


def test_sell_produces_negative_trade_value(
    trade_list: pd.DataFrame,
) -> None:
    result = estimate_transaction_costs(
        trade_list=trade_list,
        portfolio_value=1_000_000,
    )

    sell_trade = result.loc[result["action"] == "SELL"].iloc[0]

    assert sell_trade["trade_value"] == -50_000


def test_hold_produces_zero_trade_value_and_cost(
    trade_list: pd.DataFrame,
) -> None:
    result = estimate_transaction_costs(
        trade_list=trade_list,
        portfolio_value=1_000_000,
    )

    hold_trade = result.loc[result["action"] == "HOLD"].iloc[0]

    assert hold_trade["trade_value"] == 0
    assert hold_trade["transaction_cost"] == 0


def test_transaction_costs_are_calculated_correctly(
    trade_list: pd.DataFrame,
) -> None:
    result = estimate_transaction_costs(
        trade_list=trade_list,
        portfolio_value=1_000_000,
        transaction_cost_rate=0.002,
    )

    assert list(result["transaction_cost"]) == [
        100,
        60,
        0,
    ]


def test_transaction_costs_are_always_non_negative(
    trade_list: pd.DataFrame,
) -> None:
    result = estimate_transaction_costs(
        trade_list=trade_list,
        portfolio_value=1_000_000,
    )

    assert (result["transaction_cost"] >= 0).all()


def test_custom_transaction_cost_rate(
    trade_list: pd.DataFrame,
) -> None:
    result = estimate_transaction_costs(
        trade_list=trade_list,
        portfolio_value=1_000_000,
        transaction_cost_rate=0.005,
    )

    assert list(result["transaction_cost"]) == [
        250,
        150,
        0,
    ]


def test_missing_required_column_raises_value_error(
    trade_list: pd.DataFrame,
) -> None:
    invalid_trade_list = trade_list.drop(
        columns=["trade_weight"]
    )

    with pytest.raises(
        ValueError,
        match="Trade list is missing required columns",
    ):
        estimate_transaction_costs(
            trade_list=invalid_trade_list,
            portfolio_value=1_000_000,
        )


def test_invalid_portfolio_value_raises_value_error(
    trade_list: pd.DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match="Portfolio value must be greater than zero.",
    ):
        estimate_transaction_costs(
            trade_list=trade_list,
            portfolio_value=0,
        )


def test_negative_transaction_cost_rate_raises_value_error(
    trade_list: pd.DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match="Transaction cost rate cannot be negative.",
    ):
        estimate_transaction_costs(
            trade_list=trade_list,
            portfolio_value=1_000_000,
            transaction_cost_rate=-0.002,
        )


def test_invalid_trade_weight_string_raises_value_error(
    trade_list: pd.DataFrame,
) -> None:
    invalid_trade_list = trade_list.copy()
    invalid_trade_list["trade_weight"] = invalid_trade_list[
        "trade_weight"
    ].astype(object)
    invalid_trade_list.loc[0, "trade_weight"] = "not-a-number"

    with pytest.raises(
        ValueError,
        match="Trade weights must be valid numbers",
    ):
        estimate_transaction_costs(
            trade_list=invalid_trade_list,
            portfolio_value=1_000_000,
        )


def test_infinite_trade_weight_raises_value_error(
    trade_list: pd.DataFrame,
) -> None:
    invalid_trade_list = trade_list.copy()
    invalid_trade_list.loc[0, "trade_weight"] = float("inf")

    with pytest.raises(
        ValueError,
        match="Trade weights must be finite numeric values",
    ):
        estimate_transaction_costs(
            trade_list=invalid_trade_list,
            portfolio_value=1_000_000,
        )


def test_original_trade_list_is_not_modified(
    trade_list: pd.DataFrame,
) -> None:
    original_columns = list(
        trade_list.columns
    )

    estimate_transaction_costs(
        trade_list=trade_list,
        portfolio_value=1_000_000,
    )

    assert list(trade_list.columns) == original_columns
