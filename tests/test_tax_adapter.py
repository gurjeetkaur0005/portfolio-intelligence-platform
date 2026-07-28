from __future__ import annotations

import pandas as pd

from src.pipeline.tax_adapter import (
    estimate_taxes_allowing_zero_holding_buys,
)


def test_zero_holding_buy_is_treated_as_zero_tax() -> None:
    """
    BUY rows with no current holding should not realize gains.
    """

    trades = pd.DataFrame(
        {
            "portfolio_id": ["P00001"],
            "asset": ["commodities"],
            "trade_value": [10_000.0],
            "current_value": [0.0],
            "cost_basis": [0.0],
            "tax_rate": [0.20],
        }
    )

    result = estimate_taxes_allowing_zero_holding_buys(trades)

    assert result.loc[0, "estimated_tax_liability"] == 0.0
    assert result.loc[0, "portfolio_estimated_tax"] == 0.0
    assert not bool(result.loc[0, "creates_tax_liability"])


def test_taxable_sell_uses_tax_aware_optimizer() -> None:
    """
    SELL rows with holdings should preserve tax-aware optimizer math.
    """

    trades = pd.DataFrame(
        {
            "portfolio_id": ["P00001"],
            "asset": ["domestic_equity"],
            "trade_value": [-10_000.0],
            "current_value": [100_000.0],
            "cost_basis": [80_000.0],
            "tax_rate": [0.20],
        }
    )

    result = estimate_taxes_allowing_zero_holding_buys(trades)

    assert result.loc[0, "sell_value"] == 10_000.0
    assert result.loc[0, "estimated_realized_gain"] == 2_000.0
    assert result.loc[0, "estimated_tax_liability"] == 400.0
