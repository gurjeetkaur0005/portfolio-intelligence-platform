import pandas as pd
import pytest

from src.pipeline.trade_enrichment import enrich_trade_data


@pytest.fixture
def trade_list() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asset": ["domestic_equity"],
            "action": ["BUY"],
            "current_weight": [0.20],
            "trade_weight": [0.05],
            "post_trade_weight": [0.25],
            "trade_value": [50_000],
            "transaction_cost": [100],
        }
    )


def test_enrich_trade_data_adds_tax_inputs(
    trade_list: pd.DataFrame,
) -> None:
    result = enrich_trade_data(
        trade_list=trade_list,
        portfolio_id="P001",
        portfolio_value=1_000_000,
        tax_rate=0.20,
    )

    assert result.loc[0, "portfolio_id"] == "P001"
    assert result.loc[0, "current_value"] == 200_000
    assert result.loc[0, "cost_basis"] == 180_000
    assert result.loc[0, "tax_rate"] == 0.20


def test_enrich_trade_data_does_not_mutate_input(
    trade_list: pd.DataFrame,
) -> None:
    original = trade_list.copy(deep=True)

    enrich_trade_data(
        trade_list=trade_list,
        portfolio_id="P001",
        portfolio_value=1_000_000,
        tax_rate=0.20,
    )

    pd.testing.assert_frame_equal(trade_list, original)


def test_missing_required_column_raises_value_error(
    trade_list: pd.DataFrame,
) -> None:
    invalid_trade_list = trade_list.drop(
        columns=["current_weight"]
    )

    with pytest.raises(
        ValueError,
        match="Trade list is missing required columns",
    ):
        enrich_trade_data(
            trade_list=invalid_trade_list,
            portfolio_id="P001",
            portfolio_value=1_000_000,
            tax_rate=0.20,
        )


def test_invalid_current_weight_string_raises_value_error(
    trade_list: pd.DataFrame,
) -> None:
    invalid_trade_list = trade_list.copy()
    invalid_trade_list["current_weight"] = invalid_trade_list[
        "current_weight"
    ].astype(object)
    invalid_trade_list.loc[0, "current_weight"] = "not-a-number"

    with pytest.raises(
        ValueError,
        match="Current weights must be valid numbers",
    ):
        enrich_trade_data(
            trade_list=invalid_trade_list,
            portfolio_id="P001",
            portfolio_value=1_000_000,
            tax_rate=0.20,
        )


def test_infinite_current_weight_raises_value_error(
    trade_list: pd.DataFrame,
) -> None:
    invalid_trade_list = trade_list.copy()
    invalid_trade_list.loc[0, "current_weight"] = float("inf")

    with pytest.raises(
        ValueError,
        match="Current weights must be finite numeric values",
    ):
        enrich_trade_data(
            trade_list=invalid_trade_list,
            portfolio_id="P001",
            portfolio_value=1_000_000,
            tax_rate=0.20,
        )


def test_invalid_portfolio_value_raises_value_error(
    trade_list: pd.DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match="Portfolio value must be greater than zero",
    ):
        enrich_trade_data(
            trade_list=trade_list,
            portfolio_id="P001",
            portfolio_value=0,
            tax_rate=0.20,
        )


def test_invalid_tax_rate_raises_value_error(
    trade_list: pd.DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match="Tax rate must be between 0 and 1",
    ):
        enrich_trade_data(
            trade_list=trade_list,
            portfolio_id="P001",
            portfolio_value=1_000_000,
            tax_rate=1.20,
        )
