from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import Engine, func, select

from scripts import cleanup_smoke_test_data as cleanup_script
from src.database.base import DatabaseBase
from src.database.models import (
    ApprovalModel,
    AuditRecordModel,
    ClientModel,
    PortfolioHoldingModel,
    PortfolioModel,
    RebalanceRunModel,
    TradeModel,
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

    result = cleanup_script.cleanup_smoke_test_data(session_factory)

    assert result.deleted_portfolios == 1
    assert result.deleted_clients == 1
    assert result.deleted_holdings == 1
    assert result.deleted_rebalance_runs == 1
    assert result.deleted_trades == 1
    assert result.deleted_approvals == 1
    assert result.deleted_audit_records == 1
    assert not result.failed
    assert not _client_exists(session_factory, "C-SMOKE-001")
    assert not _portfolio_exists(session_factory, "P-SMOKE-001")
    assert _table_count(session_factory, PortfolioHoldingModel) == 0
    assert _table_count(session_factory, RebalanceRunModel) == 0
    assert _table_count(session_factory, TradeModel) == 0
    assert _table_count(session_factory, ApprovalModel) == 0
    assert _table_count(session_factory, AuditRecordModel) == 0


def test_cleanup_recognizes_residual_smoke_portfolio_id(
    session_factory: DatabaseSessionFactory,
) -> None:
    """Residual P-SMOKE-C0314514 data is treated as smoke data."""

    _insert_portfolio_graph(
        session_factory,
        client_id="C-SMOKE-C0314514",
        portfolio_id="P-SMOKE-C0314514",
    )

    result = cleanup_script.cleanup_smoke_test_data(session_factory)

    assert result.deleted_portfolios == 1
    assert result.deleted_clients == 1
    assert result.deleted_holdings == 1
    assert result.deleted_rebalance_runs == 1
    assert result.deleted_trades == 1
    assert result.deleted_approvals == 1
    assert result.deleted_audit_records == 1
    assert not _portfolio_exists(session_factory, "P-SMOKE-C0314514")


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

    cleanup_script.cleanup_smoke_test_data(session_factory)

    assert _client_exists(session_factory, "DEV-C00001")
    assert _portfolio_exists(session_factory, "DEV-P00001")


def test_cleanup_preserves_all_development_seed_portfolios(
    session_factory: DatabaseSessionFactory,
) -> None:
    """All deterministic development portfolios are preserved."""

    _insert_portfolio_graph(
        session_factory,
        client_id="C-SMOKE-001",
        portfolio_id="P-SMOKE-001",
    )

    for index in range(1, 11):
        _insert_portfolio_graph(
            session_factory,
            client_id=f"DEV-C{index:05d}",
            portfolio_id=f"DEV-P{index:05d}",
        )

    cleanup_script.cleanup_smoke_test_data(session_factory)

    for index in range(1, 11):
        assert _portfolio_exists(
            session_factory,
            f"DEV-P{index:05d}",
        )


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

    cleanup_script.cleanup_smoke_test_data(session_factory)

    assert _client_exists(session_factory, "CLIENT-001")
    assert _portfolio_exists(session_factory, "PORTFOLIO-001")


def test_cleanup_preserves_mixed_client_with_legitimate_portfolio(
    session_factory: DatabaseSessionFactory,
) -> None:
    """A smoke client remains when it owns a legitimate portfolio."""

    _insert_client_with_portfolios(
        session_factory,
        client_id="C-SMOKE-MIXED",
        portfolio_ids=[
            "P-SMOKE-C0314514",
            "PORTFOLIO-001",
        ],
    )

    result = cleanup_script.cleanup_smoke_test_data(session_factory)

    assert result.deleted_portfolios == 1
    assert result.deleted_clients == 0
    assert _client_exists(session_factory, "C-SMOKE-MIXED")
    assert not _portfolio_exists(session_factory, "P-SMOKE-C0314514")
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

    first_result = cleanup_script.cleanup_smoke_test_data(session_factory)
    second_result = cleanup_script.cleanup_smoke_test_data(session_factory)

    assert first_result.deleted_portfolios == 1
    assert first_result.deleted_clients == 1
    assert second_result.deleted_portfolios == 0
    assert second_result.deleted_clients == 0
    assert second_result.deleted_holdings == 0
    assert second_result.deleted_rebalance_runs == 0
    assert not second_result.failed


def test_cleanup_rolls_back_on_failure(
    session_factory: DatabaseSessionFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cleanup failure leaves smoke rows untouched."""

    _insert_portfolio_graph(
        session_factory,
        client_id="C-SMOKE-001",
        portfolio_id="P-SMOKE-C0314514",
    )

    def fail_after_loading_clients(
        session: object,
    ) -> list[PortfolioModel]:
        raise RuntimeError("forced cleanup failure")

    monkeypatch.setattr(
        cleanup_script,
        "_load_smoke_portfolios",
        fail_after_loading_clients,
    )

    result = cleanup_script.cleanup_smoke_test_data(session_factory)

    assert result.failed
    assert result.deleted_portfolios == 0
    assert _client_exists(session_factory, "C-SMOKE-001")
    assert _portfolio_exists(session_factory, "P-SMOKE-C0314514")
    assert _table_count(session_factory, PortfolioHoldingModel) == 1
    assert _table_count(session_factory, RebalanceRunModel) == 1
    assert _table_count(session_factory, TradeModel) == 1
    assert _table_count(session_factory, ApprovalModel) == 1
    assert _table_count(session_factory, AuditRecordModel) == 1


def test_cleanup_refuses_non_local_database_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default cleanup refuses unsafe configured database URLs."""

    monkeypatch.setattr(
        cleanup_script,
        "get_database_url",
        lambda: "postgresql+psycopg://user:pass@db.example.com/prod",
    )

    result = cleanup_script.cleanup_smoke_test_data()

    assert result.failed
    assert result.deleted_portfolios == 0


def _insert_portfolio_graph(
    session_factory: DatabaseSessionFactory,
    *,
    client_id: str,
    portfolio_id: str,
) -> None:
    """Insert one client, portfolio, holding, and rebalance run."""

    _insert_client_with_portfolios(
        session_factory,
        client_id=client_id,
        portfolio_ids=[
            portfolio_id,
        ],
    )


def _insert_client_with_portfolios(
    session_factory: DatabaseSessionFactory,
    *,
    client_id: str,
    portfolio_ids: list[str],
) -> None:
    """Insert one client with complete portfolio graphs."""

    with session_factory() as session:
        with session.begin():
            client = ClientModel(
                client_id=client_id,
                risk_category="moderate",
            )

            for portfolio_id in portfolio_ids:
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
                rebalance_run = RebalanceRunModel(
                    run_id=f"RUN-{portfolio_id}",
                    status="SUCCESS",
                    portfolio_value=Decimal("100000.00"),
                    transaction_cost_rate=Decimal("0.0020000000"),
                )
                trade = TradeModel(
                    asset="cash",
                    action="HOLD",
                    current_weight=Decimal("1.0000000000"),
                    trade_weight=Decimal("0.0000000000"),
                    post_trade_weight=Decimal("1.0000000000"),
                    trade_value=Decimal("0.00"),
                    transaction_cost=Decimal("0.00"),
                    estimated_tax_liability=Decimal("0.00"),
                    threshold_breached=False,
                    threshold_severity="none",
                    breach_ratio=Decimal("0.0000000000"),
                    final_trigger_type="none",
                    final_priority="none",
                    contributing_triggers="[]",
                    client_explanation="No trade required.",
                    advisor_explanation="No trade required.",
                    compliance_explanation="No trade required.",
                )
                trade.approval = ApprovalModel(
                    approval_required=False,
                    approval_status="APPROVED",
                    approval_reason="No trade required.",
                )
                trade.audit_record = AuditRecordModel(
                    audit_id=f"AUDIT-{portfolio_id}",
                    event_type="TRADE_RECOMMENDATION",
                    details="No trade required.",
                )
                rebalance_run.trades.append(trade)
                portfolio.rebalance_runs.append(rebalance_run)
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
