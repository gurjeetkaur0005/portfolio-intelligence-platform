from __future__ import annotations

from decimal import Decimal

import pytest

from config.asset_classes import ASSET_CLASSES
from src.database.models import (
    ClientModel,
    PortfolioHoldingModel,
    PortfolioModel,
)
from src.services.portfolio_input_adapter import (
    PortfolioInputAdapter,
)


def _build_stored_portfolio() -> PortfolioModel:
    """Build one stored portfolio ORM graph."""

    client = ClientModel(
        client_id="C-STORED-001",
        risk_category="balanced",
    )
    portfolio = PortfolioModel(
        portfolio_id="P-STORED-001",
        portfolio_value=Decimal("1000000.00"),
        currency="USD",
    )
    portfolio.client = client

    for asset in ASSET_CLASSES:
        portfolio.holdings.append(
            PortfolioHoldingModel(
                asset=asset,
                current_weight=Decimal(
                    "0.4000000000"
                    if asset == "domestic_equity"
                    else "0.1200000000"
                ),
                current_value=Decimal("100000.00"),
                cost_basis=Decimal("90000.00"),
            )
        )

    return portfolio


def test_adapter_uses_stored_portfolio_and_holdings() -> None:
    portfolio = _build_stored_portfolio()

    result = PortfolioInputAdapter().build_input(
        portfolio
    )

    client_row = result.client_profiles.iloc[0]
    portfolio_row = result.portfolios.iloc[0]

    assert client_row["client_id"] == "C-STORED-001"
    assert client_row["portfolio_id"] == "P-STORED-001"
    assert portfolio_row["portfolio_id"] == "P-STORED-001"
    assert (
        portfolio_row["current_domestic_equity"]
        == pytest.approx(0.4)
    )
    assert (
        portfolio_row["current_value_domestic_equity"]
        == pytest.approx(100000.0)
    )
    assert (
        portfolio_row["cost_basis_domestic_equity"]
        == pytest.approx(90000.0)
    )


def test_adapter_rejects_unknown_holding_asset() -> None:
    portfolio = _build_stored_portfolio()
    portfolio.holdings.append(
        PortfolioHoldingModel(
            asset="unsupported_asset",
            current_weight=Decimal("0.1000000000"),
            current_value=Decimal("100.00"),
            cost_basis=Decimal("90.00"),
        )
    )

    with pytest.raises(
        ValueError,
        match="Unsupported holding asset",
    ):
        PortfolioInputAdapter().build_input(portfolio)
