import pandas as pd
import pytest

from src.optimization.tax_aware_optimizer import (
    estimate_trade_taxes,
)


def test_sell_trade_creates_correct_sell_value_and_tax() -> None:
    trade_list = pd.DataFrame(
        {
            "portfolio_id": ["P001"],
            "asset": ["equity"],
            "trade_value": [-20000],
            "current_value": [100000],
            "cost_basis": [80000],
            "tax_rate": [0.20],
        }
    )

    result = estimate_trade_taxes(trade_list)

    assert result.loc[0, "unrealized_gain"] == 20000
    assert result.loc[0, "sell_value"] == 20000
    assert result.loc[0, "sell_fraction"] == 0.20
    assert result.loc[0, "estimated_realized_gain"] == 4000
    assert result.loc[0, "estimated_tax_liability"] == 800
    assert result.loc[0, "after_tax_trade_value"] == 19200
    assert result.loc[0, "portfolio_estimated_tax"] == 800
    assert result.loc[0, "creates_tax_liability"]


def test_buy_trade_creates_no_tax() -> None:
    trade_list = pd.DataFrame(
        {
            "portfolio_id": ["P001"],
            "asset": ["fixed_income"],
            "trade_value": [20000],
            "current_value": [50000],
            "cost_basis": [45000],
            "tax_rate": [0.20],
        }
    )

    result = estimate_trade_taxes(trade_list)

    assert result.loc[0, "sell_value"] == 0
    assert result.loc[0, "sell_fraction"] == 0
    assert result.loc[0, "estimated_realized_gain"] == 0
    assert result.loc[0, "estimated_tax_liability"] == 0
    assert result.loc[0, "after_tax_trade_value"] == 0
    assert not result.loc[0, "creates_tax_liability"]


def test_loss_making_sell_creates_no_tax() -> None:
    trade_list = pd.DataFrame(
        {
            "portfolio_id": ["P001"],
            "asset": ["equity"],
            "trade_value": [-20000],
            "current_value": [80000],
            "cost_basis": [100000],
            "tax_rate": [0.20],
        }
    )

    result = estimate_trade_taxes(trade_list)

    assert result.loc[0, "unrealized_gain"] == -20000
    assert result.loc[0, "sell_value"] == 20000
    assert result.loc[0, "sell_fraction"] == 0.25
    assert result.loc[0, "estimated_realized_gain"] == -5000
    assert result.loc[0, "estimated_tax_liability"] == 0
    assert result.loc[0, "after_tax_trade_value"] == 20000
    assert not result.loc[0, "creates_tax_liability"]


def test_portfolio_estimated_tax_is_summed_for_all_trades() -> None:
    trade_list = pd.DataFrame(
        {
            "portfolio_id": ["P001", "P001"],
            "asset": ["equity", "real_estate"],
            "trade_value": [-20000, -10000],
            "current_value": [100000, 50000],
            "cost_basis": [80000, 40000],
            "tax_rate": [0.20, 0.20],
        }
    )

    result = estimate_trade_taxes(trade_list)

    assert result.loc[0, "estimated_tax_liability"] == 800
    assert result.loc[1, "estimated_tax_liability"] == 400

    assert result.loc[0, "portfolio_estimated_tax"] == 1200
    assert result.loc[1, "portfolio_estimated_tax"] == 1200


def test_asset_is_required_instead_of_asset_class() -> None:
    trade_list = pd.DataFrame(
        {
            "portfolio_id": ["P001"],
            "asset_class": ["equity"],
            "trade_value": [-20000],
            "current_value": [100000],
            "cost_basis": [80000],
            "tax_rate": [0.20],
        }
    )

    with pytest.raises(
        ValueError,
        match="Trade list is missing required columns",
    ):
        estimate_trade_taxes(trade_list)


def test_missing_required_columns_raises_error() -> None:
    trade_list = pd.DataFrame(
        {
            "portfolio_id": ["P001"],
            "asset": ["equity"],
            "trade_value": [-20000],
            "current_value": [100000],
            "cost_basis": [80000],
            # tax_rate is intentionally missing
        }
    )

    with pytest.raises(
        ValueError,
        match="Trade list is missing required columns",
    ):
        estimate_trade_taxes(trade_list)


def test_missing_numeric_values_raise_error() -> None:
    trade_list = pd.DataFrame(
        {
            "portfolio_id": ["P001"],
            "asset": ["equity"],
            "trade_value": [-20000],
            "current_value": [None],
            "cost_basis": [80000],
            "tax_rate": [0.20],
        }
    )

    with pytest.raises(
        ValueError,
        match="Tax estimation inputs must not contain missing values",
    ):
        estimate_trade_taxes(trade_list)


def test_invalid_tax_rate_raises_error() -> None:
    trade_list = pd.DataFrame(
        {
            "portfolio_id": ["P001"],
            "asset": ["equity"],
            "trade_value": [-20000],
            "current_value": [100000],
            "cost_basis": [80000],
            "tax_rate": [1.20],
        }
    )

    with pytest.raises(
        ValueError,
        match="Tax rate must be between 0 and 1",
    ):
        estimate_trade_taxes(trade_list)


def test_sell_value_exceeding_holding_raises_error() -> None:
    trade_list = pd.DataFrame(
        {
            "portfolio_id": ["P001"],
            "asset": ["equity"],
            "trade_value": [-120000],
            "current_value": [100000],
            "cost_basis": [80000],
            "tax_rate": [0.20],
        }
    )

    with pytest.raises(
        ValueError,
        match="Sell trade value cannot exceed the current holding value",
    ):
        estimate_trade_taxes(trade_list)


def test_negative_current_value_raises_error() -> None:
    trade_list = pd.DataFrame(
        {
            "portfolio_id": ["P001"],
            "asset": ["equity"],
            "trade_value": [-20000],
            "current_value": [-100000],
            "cost_basis": [80000],
            "tax_rate": [0.20],
        }
    )

    with pytest.raises(
        ValueError,
        match="Current value must be greater than zero",
    ):
        estimate_trade_taxes(trade_list)


def test_negative_cost_basis_raises_error() -> None:
    trade_list = pd.DataFrame(
        {
            "portfolio_id": ["P001"],
            "asset": ["equity"],
            "trade_value": [-20000],
            "current_value": [100000],
            "cost_basis": [-80000],
            "tax_rate": [0.20],
        }
    )

    with pytest.raises(
        ValueError,
        match="Cost basis must be greater than or equal to zero",
    ):
        estimate_trade_taxes(trade_list)
