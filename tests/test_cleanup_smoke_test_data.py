from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import Engine, func, select

from scripts.cleanup_smoke_test_data import cleanup_smoke_test_data
from src.database.base import DatabaseBase
from src.database.models import (
    ClientModel,
    PortfolioHoldingModel,
    PortfolioModel,
    RebalanceRunModel,
)
from src.database.session import (
    DatabaseSessionFactory,
    create_database_engine,
    create_database_session_factory,
)


@pytest.fixture
def database_engine(
    tmp_path: Path,
) -> Iterator[Engine]:
    """Create an isolated file-backed SQLite database."""

    engine = create_database_engine(
        f"sqlite+pysqlite:///{tmp_path / 'cleanup.db'}"
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
    """Create a session factory for cleanup tests."""

    return create_database_session_factory(database_engine)


def test_cleanup_removes_smoke_rows(
    session_factory: DatabaseSessionFactory,
) -> None:
    """Smoke portfolios and clients are removed with child records."""

    _insert_portfolio_graph(
        session_factory,
        client_id="C-SMOKE-001",
        portfolio_id="P-SMOKE-001",
    )

    result = cleanup_smoke_test_data(session_factory)

    assert result.deleted_portfolios == 1
    assert result.deleted_clients == 1
    assert result.deleted_holdings == 1
    assert result.deleted_rebalance_runs == 1
    assert not result.failed
    assert not _client_exists(session_factory, "C-SMOKE-001")
    assert not _portfolio_exists(session_factory, "P-SMOKE-001")
    assert _table_count(session_factory, PortfolioHoldingModel) == 0
    assert _table_count(session_factory, RebalanceRunModel) == 0


def test_cleanup_preserves_development_seed_rows(
    session_factory: DatabaseSessionFactory,
) -> None:
    """Development seed IDs are outside the smoke cleanup criteria."""

    _insert_portfolio_graph(
        session_factory,
        client_id="C-SMOKE-001",
        portfolio_id="P-SMOKE-001",
    )
    _insert_portfolio_graph(
        session_factory,
        client_id="DEV-C00001",
        portfolio_id="DEV-P00001",
    )

    cleanup_smoke_test_data(session_factory)

    assert _client_exists(session_factory, "DEV-C00001")
    assert _portfolio_exists(session_factory, "DEV-P00001")


def test_cleanup_preserves_unrelated_rows(
    session_factory: DatabaseSessionFactory,
) -> None:
    """Non-smoke clients and portfolios remain after cleanup."""

    _insert_portfolio_graph(
        session_factory,
        client_id="C-SMOKE-001",
        portfolio_id="P-SMOKE-001",
    )
    _insert_portfolio_graph(
        session_factory,
        client_id="CLIENT-001",
        portfolio_id="PORTFOLIO-001",
    )

    cleanup_smoke_test_data(session_factory)

    assert _client_exists(session_factory, "CLIENT-001")
    assert _portfolio_exists(session_factory, "PORTFOLIO-001")


def test_cleanup_rerunning_is_safe(
    session_factory: DatabaseSessionFactory,
) -> None:
    """Running cleanup repeatedly is idempotent."""

    _insert_portfolio_graph(
        session_factory,
        client_id="C-SMOKE-001",
        portfolio_id="P-SMOKE-001",
    )

    first_result = cleanup_smoke_test_data(session_factory)
    second_result = cleanup_smoke_test_data(session_factory)

    assert first_result.deleted_portfolios == 1
    assert first_result.deleted_clients == 1
    assert second_result.deleted_portfolios == 0
    assert second_result.deleted_clients == 0
    assert second_result.deleted_holdings == 0
    assert second_result.deleted_rebalance_runs == 0
    assert not second_result.failed


def _insert_portfolio_graph(
    session_factory: DatabaseSessionFactory,
    *,
    client_id: str,
    portfolio_id: str,
) -> None:
    """Insert one client, portfolio, holding, and rebalance run."""

    with session_factory() as session:
        with session.begin():
            client = ClientModel(
                client_id=client_id,
                risk_category="moderate",
            )
            portfolio = PortfolioModel(
                portfolio_id=portfolio_id,
                portfolio_value=Decimal("100000.00"),
                currency="USD",
            )
            portfolio.holdings.append(
                PortfolioHoldingModel(
                    asset="cash",
                    current_weight=Decimal("1.0000000000"),
                    current_value=Decimal("100000.00"),
                    cost_basis=Decimal("100000.00"),
                )
            )
            portfolio.rebalance_runs.append(
                RebalanceRunModel(
                    run_id=f"RUN-{portfolio_id}",
                    status="SUCCESS",
                    portfolio_value=Decimal("100000.00"),
                    transaction_cost_rate=Decimal("0.0020000000"),
                )
            )
            client.portfolios.append(portfolio)
            session.add(client)


def _client_exists(
    session_factory: DatabaseSessionFactory,
    client_id: str,
) -> bool:
    """Return whether a client exists."""

    with session_factory() as session:
        return (
            session.scalar(
                select(ClientModel).where(
                    ClientModel.client_id == client_id
                )
            )
            is not None
        )


def _portfolio_exists(
    session_factory: DatabaseSessionFactory,
    portfolio_id: str,
) -> bool:
    """Return whether a portfolio exists."""

    with session_factory() as session:
        return (
            session.scalar(
                select(PortfolioModel).where(
                    PortfolioModel.portfolio_id == portfolio_id
                )
            )
            is not None
        )


def _table_count(
    session_factory: DatabaseSessionFactory,
    model: type[object],
) -> int:
    """Return the row count for a model."""

    with session_factory() as session:
        count = session.scalar(
            select(func.count()).select_from(model)
        )

    if count is None:
        return 0

    return count
