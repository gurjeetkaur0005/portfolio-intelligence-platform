from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import selectinload

from scripts import seed_development_portfolios as seed_script
from src.api.main import app
from src.database.base import DatabaseBase
from src.database.models import (
    ClientModel,
    PortfolioModel,
)
from src.database.session import (
    DatabaseSessionFactory,
    create_database_engine,
    create_database_session_factory,
    get_database_session,
)


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
    assert second_result.created == 0
    assert second_result.skipped == 10
    assert second_result.failed == 0
    assert portfolio_count == 10
    assert client_count == 10


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
