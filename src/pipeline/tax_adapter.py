from __future__ import annotations

import pandas as pd

from src.optimization.tax_aware_optimizer import estimate_trade_taxes


def estimate_taxes_allowing_zero_holding_buys(
    enriched_trades: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate taxes while allowing zero-current-value BUY/HOLD rows.

    The tax-aware optimizer intentionally rejects non-positive current
    holdings because sell tax math requires an existing holding value.
    BUY and HOLD rows with no current holding do not realize gains, so the
    integration layer treats them as zero-tax rows before recombining the
    full trade list.
    """

    zero_holding_non_sell_mask = (
        (enriched_trades["current_value"] <= 0)
        & (enriched_trades["trade_value"] >= 0)
    )

    if not zero_holding_non_sell_mask.any():
        return estimate_trade_taxes(enriched_trades)

    taxable_trades = enriched_trades.loc[
        ~zero_holding_non_sell_mask
    ]
    zero_tax_trades = _add_zero_tax_columns(
        enriched_trades.loc[zero_holding_non_sell_mask]
    )

    if taxable_trades.empty:
        result = zero_tax_trades
    else:
        tax_aware_trades = estimate_trade_taxes(taxable_trades)
        result = pd.concat(
            [
                tax_aware_trades,
                zero_tax_trades,
            ],
            axis=0,
        ).sort_index()

    result["portfolio_estimated_tax"] = (
        result.groupby("portfolio_id")[
            "estimated_tax_liability"
        ].transform("sum")
    )

    return result.reset_index(drop=True)


def _add_zero_tax_columns(
    trades: pd.DataFrame,
) -> pd.DataFrame:
    """Add zero-tax estimation columns for non-taxable rows."""

    result = trades.copy()

    result["unrealized_gain"] = (
        result["current_value"]
        - result["cost_basis"]
    )
    result["sell_value"] = 0.0
    result["sell_fraction"] = 0.0
    result["estimated_realized_gain"] = 0.0
    result["estimated_tax_liability"] = 0.0
    result["creates_tax_liability"] = False
    result["after_tax_trade_value"] = 0.0
    result["portfolio_estimated_tax"] = 0.0

    return result
