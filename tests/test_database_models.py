from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy import Engine, event, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

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
    create_database_engine,
    create_database_session_factory,
)


EXPECTED_TABLES = {
    "clients",
    "portfolios",
    "portfolio_holdings",
    "rebalance_runs",
    "trades",
    "approvals",
    "audit_records",
}


@pytest.fixture
def database_engine() -> Engine:
    """Create an isolated in-memory SQLite database."""

    engine = create_database_engine(
        "sqlite+pysqlite:///:memory:"
    )

    # SQLite does not enforce foreign keys by default.
    # Enable them so ON DELETE CASCADE can be tested.
    @event.listens_for(engine, "connect")
    def enable_sqlite_foreign_keys(
        dbapi_connection: object,
        connection_record: object,
    ) -> None:
        del connection_record

        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    DatabaseBase.metadata.create_all(engine)

    try:
        yield engine
    finally:
        DatabaseBase.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture
def database_session(
    database_engine: Engine,
) -> Session:
    """Create one isolated database session."""

    session_factory = create_database_session_factory(
        database_engine
    )

    with session_factory() as session:
        yield session

        # Ensure failed tests do not leave an active transaction.
        session.rollback()


def _build_client_with_portfolio() -> tuple[
    ClientModel,
    PortfolioModel,
]:
    """Build a valid client and portfolio object graph."""

    client = ClientModel(
        client_id="C00001",
        risk_category="balanced",
    )

    portfolio = PortfolioModel(
        portfolio_id="P00001",
        portfolio_value=Decimal("1000000.00"),
        currency="USD",
    )

    client.portfolios.append(portfolio)

    return client, portfolio


def _build_rebalance_run(
    portfolio: PortfolioModel,
) -> RebalanceRunModel:
    """Build a valid rebalance run."""

    rebalance_run = RebalanceRunModel(
        run_id="RUN00001",
        status="SUCCESS",
        portfolio_value=Decimal("1000000.00"),
        transaction_cost_rate=Decimal("0.0020000000"),
    )

    portfolio.rebalance_runs.append(rebalance_run)

    return rebalance_run


def _build_trade(
    rebalance_run: RebalanceRunModel,
    *,
    asset: str = "domestic_equity",
) -> TradeModel:
    """Build a valid trade linked to a rebalance run."""

    trade = TradeModel(
        asset=asset,
        action="BUY",
        current_weight=Decimal("0.3000000000"),
        trade_weight=Decimal("0.1000000000"),
        post_trade_weight=Decimal("0.4000000000"),
        trade_value=Decimal("100000.00"),
        transaction_cost=Decimal("200.00"),
        estimated_tax_liability=Decimal("0.00"),
        threshold_breached=True,
        threshold_severity="high",
        breach_ratio=Decimal("1.5000000000"),
        final_trigger_type="threshold",
        final_priority="high",
        contributing_triggers="threshold",
        client_explanation="Client explanation.",
        advisor_explanation="Advisor explanation.",
        compliance_explanation="Compliance explanation.",
    )

    rebalance_run.trades.append(trade)

    return trade


def test_expected_database_tables_are_registered() -> None:
    """All ORM models should be registered in shared metadata."""

    assert EXPECTED_TABLES.issubset(
        DatabaseBase.metadata.tables
    )


def test_expected_tables_are_created(
    database_engine: Engine,
) -> None:
    """All expected tables should exist in the test database."""

    inspector = inspect(database_engine)

    assert set(inspector.get_table_names()) == EXPECTED_TABLES


def test_client_can_be_persisted(
    database_session: Session,
) -> None:
    """A valid client should be stored successfully."""

    client = ClientModel(
        client_id="C00001",
        risk_category="balanced",
    )

    database_session.add(client)
    database_session.commit()

    assert client.id is not None
    assert client.client_id == "C00001"
    assert client.risk_category == "balanced"
    assert client.created_at is not None
    assert client.updated_at is not None


def test_portfolio_can_be_linked_to_client(
    database_session: Session,
) -> None:
    """A portfolio should be linked in both ORM directions."""

    client, portfolio = _build_client_with_portfolio()

    database_session.add(client)
    database_session.commit()

    assert portfolio.id is not None
    assert portfolio.client_id == client.id
    assert portfolio.client is client
    assert client.portfolios == [portfolio]


def test_holding_can_be_linked_to_portfolio(
    database_session: Session,
) -> None:
    """A holding should be linked to its portfolio."""

    client, portfolio = _build_client_with_portfolio()

    holding = PortfolioHoldingModel(
        asset="domestic_equity",
        current_weight=Decimal("0.4000000000"),
        current_value=Decimal("400000.00"),
        cost_basis=Decimal("360000.00"),
    )

    portfolio.holdings.append(holding)

    database_session.add(client)
    database_session.commit()

    assert holding.id is not None
    assert holding.portfolio_id == portfolio.id
    assert holding.portfolio is portfolio
    assert portfolio.holdings == [holding]


def test_rebalance_run_can_be_linked_to_portfolio(
    database_session: Session,
) -> None:
    """A rebalance run should belong to one portfolio."""

    client, portfolio = _build_client_with_portfolio()
    rebalance_run = _build_rebalance_run(portfolio)

    database_session.add(client)
    database_session.commit()

    assert rebalance_run.id is not None
    assert rebalance_run.portfolio_id == portfolio.id
    assert rebalance_run.portfolio is portfolio
    assert portfolio.rebalance_runs == [rebalance_run]


def test_trade_can_be_linked_to_rebalance_run(
    database_session: Session,
) -> None:
    """A trade should belong to one rebalance run."""

    client, portfolio = _build_client_with_portfolio()
    rebalance_run = _build_rebalance_run(portfolio)
    trade = _build_trade(rebalance_run)

    database_session.add(client)
    database_session.commit()

    assert trade.id is not None
    assert trade.rebalance_run_id == rebalance_run.id
    assert trade.rebalance_run is rebalance_run
    assert rebalance_run.trades == [trade]


def test_approval_can_be_linked_to_trade(
    database_session: Session,
) -> None:
    """One approval record should be linked to one trade."""

    client, portfolio = _build_client_with_portfolio()
    rebalance_run = _build_rebalance_run(portfolio)
    trade = _build_trade(rebalance_run)

    approval = ApprovalModel(
        approval_required=True,
        approval_status="PENDING",
        approval_reason="Human review is required.",
    )

    trade.approval = approval

    database_session.add(client)
    database_session.commit()

    assert approval.id is not None
    assert approval.trade_id == trade.id
    assert approval.trade is trade
    assert trade.approval is approval


def test_audit_record_can_be_linked_to_trade(
    database_session: Session,
) -> None:
    """One audit record should be linked to one trade."""

    client, portfolio = _build_client_with_portfolio()
    rebalance_run = _build_rebalance_run(portfolio)
    trade = _build_trade(rebalance_run)

    audit_record = AuditRecordModel(
        audit_id="AUDIT00001",
        details="Trade recommendation generated.",
    )

    trade.audit_record = audit_record

    database_session.add(client)
    database_session.commit()

    assert audit_record.id is not None
    assert audit_record.trade_id == trade.id
    assert audit_record.trade is trade
    assert trade.audit_record is audit_record


def test_duplicate_client_business_id_is_rejected(
    database_session: Session,
) -> None:
    """Duplicate client business identifiers should be rejected."""

    database_session.add_all(
        [
            ClientModel(
                client_id="C00001",
                risk_category="balanced",
            ),
            ClientModel(
                client_id="C00001",
                risk_category="aggressive",
            ),
        ]
    )

    with pytest.raises(IntegrityError):
        database_session.commit()

    database_session.rollback()


def test_duplicate_portfolio_asset_is_rejected(
    database_session: Session,
) -> None:
    """A portfolio cannot contain the same asset twice."""

    client, portfolio = _build_client_with_portfolio()

    portfolio.holdings.extend(
        [
            PortfolioHoldingModel(
                asset="cash",
                current_weight=Decimal("0.0500000000"),
                current_value=Decimal("50000.00"),
                cost_basis=Decimal("50000.00"),
            ),
            PortfolioHoldingModel(
                asset="cash",
                current_weight=Decimal("0.0500000000"),
                current_value=Decimal("50000.00"),
                cost_basis=Decimal("50000.00"),
            ),
        ]
    )

    database_session.add(client)

    with pytest.raises(IntegrityError):
        database_session.commit()

    database_session.rollback()


def test_duplicate_trade_asset_for_same_run_is_rejected(
    database_session: Session,
) -> None:
    """One rebalance run cannot contain duplicate asset trades."""

    client, portfolio = _build_client_with_portfolio()
    rebalance_run = _build_rebalance_run(portfolio)

    _build_trade(
        rebalance_run,
        asset="cash",
    )
    _build_trade(
        rebalance_run,
        asset="cash",
    )

    database_session.add(client)

    with pytest.raises(IntegrityError):
        database_session.commit()

    database_session.rollback()


def test_duplicate_approval_for_trade_is_rejected(
    database_session: Session,
) -> None:
    """A trade should have at most one approval row."""

    client, portfolio = _build_client_with_portfolio()
    rebalance_run = _build_rebalance_run(portfolio)
    trade = _build_trade(rebalance_run)

    database_session.add(client)
    database_session.commit()

    first_approval = ApprovalModel(
        trade_id=trade.id,
        approval_required=True,
        approval_status="PENDING",
        approval_reason="First approval.",
    )

    second_approval = ApprovalModel(
        trade_id=trade.id,
        approval_required=True,
        approval_status="PENDING",
        approval_reason="Second approval.",
    )

    database_session.add_all(
        [
            first_approval,
            second_approval,
        ]
    )

    with pytest.raises(IntegrityError):
        database_session.commit()

    database_session.rollback()


def test_negative_portfolio_value_is_rejected(
    database_session: Session,
) -> None:
    """Portfolio values must be positive."""

    client = ClientModel(
        client_id="C00001",
        risk_category="balanced",
    )

    client.portfolios.append(
        PortfolioModel(
            portfolio_id="P00001",
            portfolio_value=Decimal("-100.00"),
        )
    )

    database_session.add(client)

    with pytest.raises(IntegrityError):
        database_session.commit()

    database_session.rollback()


def test_holding_weight_above_one_is_rejected(
    database_session: Session,
) -> None:
    """Holding weights must not exceed one."""

    client, portfolio = _build_client_with_portfolio()

    portfolio.holdings.append(
        PortfolioHoldingModel(
            asset="domestic_equity",
            current_weight=Decimal("1.1000000000"),
            current_value=Decimal("1100000.00"),
            cost_basis=Decimal("1000000.00"),
        )
    )

    database_session.add(client)

    with pytest.raises(IntegrityError):
        database_session.commit()

    database_session.rollback()


def test_invalid_trade_action_is_rejected(
    database_session: Session,
) -> None:
    """Only BUY, SELL, and HOLD trade actions should be accepted."""

    client, portfolio = _build_client_with_portfolio()
    rebalance_run = _build_rebalance_run(portfolio)
    trade = _build_trade(rebalance_run)

    trade.action = "INVALID"

    database_session.add(client)

    with pytest.raises(IntegrityError):
        database_session.commit()

    database_session.rollback()


def test_negative_transaction_cost_is_rejected(
    database_session: Session,
) -> None:
    """Transaction costs cannot be negative."""

    client, portfolio = _build_client_with_portfolio()
    rebalance_run = _build_rebalance_run(portfolio)
    trade = _build_trade(rebalance_run)

    trade.transaction_cost = Decimal("-1.00")

    database_session.add(client)

    with pytest.raises(IntegrityError):
        database_session.commit()

    database_session.rollback()


def test_deleting_client_deletes_all_related_models(
    database_session: Session,
) -> None:
    """Deleting a client should cascade through the entire hierarchy."""

    client, portfolio = _build_client_with_portfolio()

    portfolio.holdings.append(
        PortfolioHoldingModel(
            asset="cash",
            current_weight=Decimal("0.0500000000"),
            current_value=Decimal("50000.00"),
            cost_basis=Decimal("50000.00"),
        )
    )

    rebalance_run = _build_rebalance_run(portfolio)
    trade = _build_trade(rebalance_run)

    trade.approval = ApprovalModel(
        approval_required=True,
        approval_status="PENDING",
        approval_reason="Human review required.",
    )

    trade.audit_record = AuditRecordModel(
        audit_id="AUDIT00001",
        details="Trade recommendation generated.",
    )

    database_session.add(client)
    database_session.commit()

    database_session.delete(client)
    database_session.commit()

    assert database_session.query(ClientModel).count() == 0
    assert database_session.query(PortfolioModel).count() == 0
    assert (
        database_session.query(
            PortfolioHoldingModel
        ).count()
        == 0
    )
    assert (
        database_session.query(
            RebalanceRunModel
        ).count()
        == 0
    )
    assert database_session.query(TradeModel).count() == 0
    assert database_session.query(ApprovalModel).count() == 0
    assert database_session.query(AuditRecordModel).count() == 0