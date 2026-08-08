from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
import pandas as pd
import pytest
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import selectinload

from scripts import seed_development_portfolios as seed_script
from src.api.main import app
from src.database.base import DatabaseBase
from src.database.models import (
    ClientModel,
    PortfolioHoldingModel,
    PortfolioModel,
)
from src.database.repositories import (
    PortfolioRepository,
    RebalanceRunRepository,
)
from src.database.session import (
    DatabaseSessionFactory,
    create_database_engine,
    create_database_session_factory,
    get_database_session,
)
from src.services.rebalance_application_service import (
    RebalanceApplicationService,
)
from src.database.persistence_service import (
    RebalancePersistenceService,
)
from src.optimization.tax_aware_optimizer import estimate_trade_taxes


@pytest.fixture
def database_engine(
    tmp_path: Path,
) -> Iterator[Engine]:
    """Create an isolated file-backed SQLite database."""

    engine = create_database_engine(
        f"sqlite+pysqlite:///{tmp_path / 'seed.db'}"
    )
    DatabaseBase.metadata.create_all(engine)

    try:
        yield engine
    finally:
        DatabaseBase.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture
def session_factory(
    database_engine: Engine,
) -> DatabaseSessionFactory:
    """Create a session factory for seed tests."""

    return create_database_session_factory(database_engine)


def test_seed_creates_ten_portfolios(
    session_factory: DatabaseSessionFactory,
) -> None:
    result = seed_script.seed_development_portfolios(
        session_factory
    )

    with session_factory() as session:
        portfolio_count = session.scalar(
            select(func.count()).select_from(PortfolioModel)
        )
        client_count = session.scalar(
            select(func.count()).select_from(ClientModel)
        )

    assert result.created == 10
    assert result.replaced == 0
    assert result.skipped == 0
    assert result.failed == 0
    assert portfolio_count == 10
    assert client_count == 10


def test_every_seeded_portfolio_has_six_holdings(
    session_factory: DatabaseSessionFactory,
) -> None:
    seed_script.seed_development_portfolios(session_factory)

    portfolios = _load_portfolios(session_factory)

    assert len(portfolios) == 10
    assert all(
        len(portfolio.holdings) == 6
        for portfolio in portfolios
    )
    assert {
        holding.asset
        for portfolio in portfolios
        for holding in portfolio.holdings
    } == set(seed_script.ASSET_CLASSES)


def test_seeded_holding_weights_sum_to_one(
    session_factory: DatabaseSessionFactory,
) -> None:
    seed_script.seed_development_portfolios(session_factory)

    portfolios = _load_portfolios(session_factory)

    for portfolio in portfolios:
        total_weight = sum(
            (
                holding.current_weight
                for holding in portfolio.holdings
            ),
            Decimal("0"),
        )
        assert total_weight == Decimal("1.0000000000")


def test_seeded_cost_basis_is_realistic_for_non_cash(
    session_factory: DatabaseSessionFactory,
) -> None:
    seed_script.seed_development_portfolios(session_factory)

    portfolios = _load_portfolios(session_factory)
    non_cash_holdings = [
        holding
        for portfolio in portfolios
        for holding in portfolio.holdings
        if holding.asset != "cash"
    ]

    assert any(
        holding.current_value != holding.cost_basis
        for holding in non_cash_holdings
    )
    assert any(
        holding.current_value > holding.cost_basis
        for holding in non_cash_holdings
    )
    assert any(
        holding.current_value < holding.cost_basis
        for holding in non_cash_holdings
    )
    assert all(
        holding.cost_basis >= Decimal("0.00")
        for holding in non_cash_holdings
    )


def test_seeded_cash_cost_basis_matches_current_value(
    session_factory: DatabaseSessionFactory,
) -> None:
    seed_script.seed_development_portfolios(session_factory)

    portfolios = _load_portfolios(session_factory)
    cash_holdings = [
        holding
        for portfolio in portfolios
        for holding in portfolio.holdings
        if holding.asset == "cash"
    ]

    assert cash_holdings
    assert all(
        holding.cost_basis == holding.current_value
        for holding in cash_holdings
    )


def test_seeded_current_values_remain_allocation_driven(
    session_factory: DatabaseSessionFactory,
) -> None:
    seed_script.seed_development_portfolios(session_factory)

    portfolios = _load_portfolios(session_factory)

    for portfolio in portfolios:
        for holding in portfolio.holdings:
            expected_current_value = (
                portfolio.portfolio_value
                * holding.current_weight
            ).quantize(
                seed_script.MONEY_QUANTUM,
                rounding=ROUND_HALF_UP,
            )

            assert holding.current_value == expected_current_value


def test_seeded_cost_basis_is_deterministic(
    session_factory: DatabaseSessionFactory,
) -> None:
    seed_script.seed_development_portfolios(session_factory)
    first_snapshot = _holding_snapshot(session_factory)

    result = seed_script.seed_development_portfolios(
        session_factory
    )
    second_snapshot = _holding_snapshot(session_factory)

    assert result.skipped == 10
    assert second_snapshot == first_snapshot


def test_seeded_profitable_sell_can_create_tax(
    session_factory: DatabaseSessionFactory,
) -> None:
    seed_script.seed_development_portfolios(session_factory)
    profitable_holding = _first_holding_with_gain(
        session_factory
    )

    result = estimate_trade_taxes(
        pd.DataFrame(
            [
                _tax_trade_row(
                    holding=profitable_holding,
                    trade_fraction=Decimal("-0.10"),
                )
            ]
        )
    )

    assert result.loc[0, "estimated_tax_liability"] > 0


def test_seeded_loss_sell_does_not_create_positive_tax(
    session_factory: DatabaseSessionFactory,
) -> None:
    seed_script.seed_development_portfolios(session_factory)
    loss_holding = _first_holding_with_loss(session_factory)

    result = estimate_trade_taxes(
        pd.DataFrame(
            [
                _tax_trade_row(
                    holding=loss_holding,
                    trade_fraction=Decimal("-0.10"),
                )
            ]
        )
    )

    assert result.loc[0, "estimated_tax_liability"] == 0


def test_running_seed_twice_creates_no_duplicates(
    session_factory: DatabaseSessionFactory,
) -> None:
    first_result = seed_script.seed_development_portfolios(
        session_factory
    )
    second_result = seed_script.seed_development_portfolios(
        session_factory
    )

    with session_factory() as session:
        portfolio_count = session.scalar(
            select(func.count()).select_from(PortfolioModel)
        )
        client_count = session.scalar(
            select(func.count()).select_from(ClientModel)
        )

    assert first_result.created == 10
    assert first_result.replaced == 0
    assert second_result.created == 0
    assert second_result.replaced == 0
    assert second_result.skipped == 10
    assert second_result.failed == 0
    assert portfolio_count == 10
    assert client_count == 10


def test_seed_replaces_stale_development_portfolio(
    session_factory: DatabaseSessionFactory,
) -> None:
    _insert_stale_development_portfolio(session_factory)

    result = seed_script.seed_development_portfolios(
        session_factory
    )

    portfolio = _load_portfolios(session_factory)[0]
    holding_value_total = sum(
        (
            holding.current_value
            for holding in portfolio.holdings
        ),
        Decimal("0"),
    )

    assert result.created == 9
    assert result.replaced == 1
    assert result.skipped == 0
    assert result.failed == 0
    assert portfolio.portfolio_id == "DEV-P00001"
    assert portfolio.portfolio_value == Decimal("500000.00")
    assert holding_value_total == portfolio.portfolio_value


def test_refreshed_seeded_portfolio_rebalances_successfully(
    session_factory: DatabaseSessionFactory,
) -> None:
    _insert_stale_development_portfolio(session_factory)
    seed_script.seed_development_portfolios(session_factory)

    with session_factory() as session:
        service = RebalanceApplicationService(
            portfolio_repository=PortfolioRepository(
                session
            ),
            persistence_service=RebalancePersistenceService(
                RebalanceRunRepository(session)
            ),
        )

        result = service.execute_rebalance(
            portfolio_id="DEV-P00001",
            transaction_cost_rate=Decimal("0.002"),
            run_id="RUN-SEED-REGRESSION",
        )

    assert result.portfolio_id == "DEV-P00001"
    assert result.run_id == "RUN-SEED-REGRESSION"
    assert result.trade_count == 6


def test_seed_failure_rolls_back_transaction(
    monkeypatch: pytest.MonkeyPatch,
    session_factory: DatabaseSessionFactory,
) -> None:
    def fail_build_holdings(
        *,
        portfolio_row: Any,
        portfolio_value: Decimal,
    ) -> list[object]:
        del portfolio_row, portfolio_value
        raise RuntimeError("Injected seed failure.")

    monkeypatch.setattr(
        seed_script,
        "_build_holdings",
        fail_build_holdings,
    )

    result = seed_script.seed_development_portfolios(
        session_factory
    )

    with session_factory() as session:
        portfolio_count = session.scalar(
            select(func.count()).select_from(PortfolioModel)
        )
        client_count = session.scalar(
            select(func.count()).select_from(ClientModel)
        )

    assert result.created == 0
    assert result.replaced == 0
    assert result.skipped == 0
    assert result.failed == 10
    assert portfolio_count == 0
    assert client_count == 0


def test_seed_includes_all_risk_categories(
    session_factory: DatabaseSessionFactory,
) -> None:
    seed_script.seed_development_portfolios(session_factory)

    with session_factory() as session:
        risk_categories = set(
            session.scalars(
                select(ClientModel.risk_category)
            ).all()
        )

    assert risk_categories == set(seed_script.RISK_CATEGORIES)


def test_seeded_portfolios_are_returned_by_read_api(
    session_factory: DatabaseSessionFactory,
) -> None:
    seed_script.seed_development_portfolios(session_factory)

    def override_database_session() -> Iterator[object]:
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_database_session] = (
        override_database_session
    )

    try:
        response = TestClient(app).get(
            "/portfolios?limit=20&offset=0"
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["count"] >= 10
    assert body["items"][:10] == [
        {
            "portfolio_id": f"{seed_script.SEED_PORTFOLIO_ID_PREFIX}"
            f"{index:05d}",
            "client_id": f"{seed_script.SEED_CLIENT_ID_PREFIX}"
            f"{index:05d}",
            "portfolio_value": float(
                500_000 + ((index - 1) * 125_000)
            ),
            "currency": seed_script.DEFAULT_CURRENCY,
        }
        for index in range(1, 11)
    ]


def test_new_database_rebalance_read_api_returns_completed_at(
    session_factory: DatabaseSessionFactory,
) -> None:
    seed_script.seed_development_portfolios(session_factory)

    def override_database_session() -> Iterator[object]:
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_database_session] = (
        override_database_session
    )

    try:
        client = TestClient(app)
        rebalance_response = client.post(
            "/portfolios/DEV-P00001/rebalance",
            json={"transaction_cost_rate": 0.002},
        )
        run_id = rebalance_response.json()["run_id"]
        read_response = client.get(
            f"/rebalances/{run_id}"
        )
    finally:
        app.dependency_overrides.clear()

    assert rebalance_response.status_code == 200
    assert read_response.status_code == 200

    body = read_response.json()
    assert body["completed_at"] is not None
    assert (
        datetime.fromisoformat(body["completed_at"])
        >= datetime.fromisoformat(body["created_at"])
    )


def _load_portfolios(
    session_factory: DatabaseSessionFactory,
) -> list[PortfolioModel]:
    """Return seeded portfolios with holdings loaded."""

    with session_factory() as session:
        return list(
            session.scalars(
                select(PortfolioModel)
                .options(
                    selectinload(
                        PortfolioModel.holdings
                    )
                )
                .order_by(PortfolioModel.portfolio_id)
            ).all()
        )


def _holding_snapshot(
    session_factory: DatabaseSessionFactory,
) -> list[tuple[str, str, Decimal, Decimal, Decimal]]:
    """Return stable holding values for determinism assertions."""

    return [
        (
            portfolio.portfolio_id,
            holding.asset,
            holding.current_weight,
            holding.current_value,
            holding.cost_basis,
        )
        for portfolio in _load_portfolios(session_factory)
        for holding in sorted(
            portfolio.holdings,
            key=lambda item: item.asset,
        )
    ]


def _first_holding_with_gain(
    session_factory: DatabaseSessionFactory,
) -> PortfolioHoldingModel:
    """Return the first seeded holding with unrealized gain."""

    for portfolio in _load_portfolios(session_factory):
        for holding in portfolio.holdings:
            if (
                holding.current_value > holding.cost_basis
                and holding.current_value > Decimal("0.00")
            ):
                return holding

    raise AssertionError("Expected at least one seeded gain holding.")


def _first_holding_with_loss(
    session_factory: DatabaseSessionFactory,
) -> PortfolioHoldingModel:
    """Return the first seeded holding with unrealized loss."""

    for portfolio in _load_portfolios(session_factory):
        for holding in portfolio.holdings:
            if (
                holding.current_value < holding.cost_basis
                and holding.current_value > Decimal("0.00")
            ):
                return holding

    raise AssertionError("Expected at least one seeded loss holding.")


def _tax_trade_row(
    *,
    holding: PortfolioHoldingModel,
    trade_fraction: Decimal,
) -> dict[str, object]:
    """Return one trade row for tax-estimation assertions."""

    return {
        "portfolio_id": "DEV-P-TAX-TEST",
        "asset": holding.asset,
        "trade_value": float(holding.current_value * trade_fraction),
        "current_value": float(holding.current_value),
        "cost_basis": float(holding.cost_basis),
        "tax_rate": 0.20,
    }


def _insert_stale_development_portfolio(
    session_factory: DatabaseSessionFactory,
) -> None:
    """Insert stale dev data that should be refreshed by the seed."""

    with session_factory() as session:
        with session.begin():
            client = ClientModel(
                client_id="DEV-C00001",
                risk_category="conservative",
            )
            portfolio = PortfolioModel(
                portfolio_id="DEV-P00001",
                portfolio_value=Decimal("1000000.00"),
                currency="USD",
            )
            client.portfolios.append(portfolio)

            weights = [
                Decimal("0.5000000000"),
                Decimal("0.1000000000"),
                Decimal("0.1000000000"),
                Decimal("0.1000000000"),
                Decimal("0.1000000000"),
                Decimal("0.1000000000"),
            ]

            for asset, weight in zip(
                seed_script.ASSET_CLASSES,
                weights,
            ):
                portfolio.holdings.append(
                    PortfolioHoldingModel(
                        asset=asset,
                        current_weight=weight,
                        current_value=Decimal("1.00"),
                        cost_basis=Decimal("0.90"),
                    )
                )

            session.add(client)
