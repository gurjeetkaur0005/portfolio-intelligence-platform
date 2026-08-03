from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from src.database.base import DatabaseBase
from src.database.models import (
    ApprovalModel,
    AuditRecordModel,
    PortfolioHoldingModel,
    PortfolioModel,
    RebalanceRunModel,
    TradeModel,
)
from src.database.repositories import (
    DuplicateRecordError,
    PortfolioRepository,
    RebalanceRunRepository,
    RecordNotFoundError,
)
from src.database.session import (
    create_database_engine,
    create_database_session_factory,
)


@pytest.fixture
def database_engine() -> Engine:
    """Create an isolated in-memory database."""

    engine = create_database_engine(
        "sqlite+pysqlite:///:memory:"
    )

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
    """Create one isolated session."""

    factory = create_database_session_factory(
        database_engine
    )

    with factory() as session:
        yield session


def _build_rebalance_run(
    portfolio: PortfolioModel,
) -> RebalanceRunModel:
    """Build one valid rebalance model graph."""

    run = RebalanceRunModel(
        run_id="RUN000001",
        status="SUCCESS",
        portfolio_value=Decimal("1000000.00"),
        transaction_cost_rate=Decimal(
            "0.0020000000"
        ),
    )

    trade = TradeModel(
        asset="domestic_equity",
        action="SELL",
        current_weight=Decimal("0.6000000000"),
        trade_weight=Decimal("-0.0200000000"),
        post_trade_weight=Decimal("0.5800000000"),
        trade_value=Decimal("-20000.00"),
        transaction_cost=Decimal("40.00"),
        estimated_tax_liability=Decimal("400.00"),
        threshold_breached=True,
        threshold_severity="high",
        breach_ratio=Decimal("1.5000000000"),
        final_trigger_type="threshold",
        final_priority="high",
        contributing_triggers="threshold",
        client_explanation="Client explanation.",
        advisor_explanation="Advisor explanation.",
        compliance_explanation=(
            "Compliance explanation."
        ),
    )

    trade.approval = ApprovalModel(
        approval_required=False,
        approval_status="NOT_REQUIRED",
        approval_reason="Automatic approval.",
    )

    trade.audit_record = AuditRecordModel(
        audit_id="AUD000001",
        event_type="TRADE_RECOMMENDATION",
        details="Trade recorded.",
    )

    run.trades.append(trade)
    portfolio.rebalance_runs.append(run)

    return run


def test_repository_creates_client(
    database_session: Session,
) -> None:
    repository = PortfolioRepository(
        database_session
    )

    client = repository.create_client(
        client_id="C00001",
        risk_category="balanced",
    )

    assert client.id is not None
    assert client.client_id == "C00001"


def test_repository_returns_client_by_business_id(
    database_session: Session,
) -> None:
    repository = PortfolioRepository(
        database_session
    )

    repository.create_client(
        client_id="C00001",
        risk_category="balanced",
    )

    client = repository.get_client_by_business_id(
        "C00001"
    )

    assert client is not None
    assert client.risk_category == "balanced"


def test_require_client_raises_when_missing(
    database_session: Session,
) -> None:
    repository = PortfolioRepository(
        database_session
    )

    with pytest.raises(
        RecordNotFoundError,
        match="was not found",
    ):
        repository.require_client_by_business_id(
            "UNKNOWN"
        )


def test_duplicate_client_is_rejected(
    database_session: Session,
) -> None:
    repository = PortfolioRepository(
        database_session
    )

    repository.create_client(
        client_id="C00001",
        risk_category="balanced",
    )

    with pytest.raises(DuplicateRecordError):
        repository.create_client(
            client_id="C00001",
            risk_category="aggressive",
        )


def test_repository_creates_portfolio(
    database_session: Session,
) -> None:
    repository = PortfolioRepository(
        database_session
    )

    client = repository.create_client(
        client_id="C00001",
        risk_category="balanced",
    )

    portfolio = repository.create_portfolio(
        client=client,
        portfolio_id="P00001",
        portfolio_value=Decimal("1000000.00"),
    )

    assert portfolio.id is not None
    assert portfolio.client_id == client.id
    assert portfolio.currency == "USD"


def test_repository_replaces_holdings(
    database_session: Session,
) -> None:
    repository = PortfolioRepository(
        database_session
    )

    client = repository.create_client(
        client_id="C00001",
        risk_category="balanced",
    )

    portfolio = repository.create_portfolio(
        client=client,
        portfolio_id="P00001",
        portfolio_value=Decimal("1000000.00"),
    )

    holding = PortfolioHoldingModel(
        asset="cash",
        current_weight=Decimal("0.0500000000"),
        current_value=Decimal("50000.00"),
        cost_basis=Decimal("50000.00"),
    )

    repository.replace_holdings(
        portfolio=portfolio,
        holdings=[holding],
    )

    loaded = (
        repository.require_portfolio_by_business_id(
            "P00001"
        )
    )

    assert len(loaded.holdings) == 1
    assert loaded.holdings[0].asset == "cash"


def test_rebalance_repository_saves_complete_graph(
    database_session: Session,
) -> None:
    portfolio_repository = PortfolioRepository(
        database_session
    )

    client = portfolio_repository.create_client(
        client_id="C00001",
        risk_category="balanced",
    )

    portfolio = portfolio_repository.create_portfolio(
        client=client,
        portfolio_id="P00001",
        portfolio_value=Decimal("1000000.00"),
    )

    run = _build_rebalance_run(portfolio)

    repository = RebalanceRunRepository(
        database_session
    )

    saved_run = repository.save_rebalance_run(
        run
    )

    assert saved_run.id is not None
    assert len(saved_run.trades) == 1
    assert saved_run.trades[0].approval is not None
    assert (
        saved_run.trades[0].audit_record
        is not None
    )


def test_rebalance_repository_loads_complete_graph(
    database_session: Session,
) -> None:
    portfolio_repository = PortfolioRepository(
        database_session
    )

    client = portfolio_repository.create_client(
        client_id="C00001",
        risk_category="balanced",
    )

    portfolio = portfolio_repository.create_portfolio(
        client=client,
        portfolio_id="P00001",
        portfolio_value=Decimal("1000000.00"),
    )

    repository = RebalanceRunRepository(
        database_session
    )

    repository.save_rebalance_run(
        _build_rebalance_run(portfolio)
    )

    loaded = repository.require_by_run_id(
        "RUN000001"
    )

    assert len(loaded.trades) == 1
    assert loaded.trades[0].approval is not None
    assert loaded.trades[0].audit_record is not None


def test_rebalance_repository_lists_trades(
    database_session: Session,
) -> None:
    portfolio_repository = PortfolioRepository(
        database_session
    )

    client = portfolio_repository.create_client(
        client_id="C00001",
        risk_category="balanced",
    )

    portfolio = portfolio_repository.create_portfolio(
        client=client,
        portfolio_id="P00001",
        portfolio_value=Decimal("1000000.00"),
    )

    repository = RebalanceRunRepository(
        database_session
    )

    repository.save_rebalance_run(
        _build_rebalance_run(portfolio)
    )

    trades = repository.list_trades(
        "RUN000001"
    )

    assert len(trades) == 1
    assert trades[0].asset == "domestic_equity"


def test_invalid_currency_is_rejected(
    database_session: Session,
) -> None:
    repository = PortfolioRepository(
        database_session
    )

    client = repository.create_client(
        client_id="C00001",
        risk_category="balanced",
    )

    with pytest.raises(
        ValueError,
        match="three letters",
    ):
        repository.create_portfolio(
            client=client,
            portfolio_id="P00001",
            portfolio_value=Decimal("1000000.00"),
            currency="US",
        )