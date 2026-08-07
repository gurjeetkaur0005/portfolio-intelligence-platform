from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.database.models import (
    ApprovalModel,
    AuditRecordModel,
    ClientModel,
    PortfolioHoldingModel,
    PortfolioModel,
    RebalanceRunModel,
    TradeModel,
)
from src.database.repositories import RecordNotFoundError
from src.services.portfolio_read_application_service import (
    PortfolioReadApplicationService,
)


TIMESTAMP = datetime(
    2026,
    8,
    5,
    12,
    30,
    tzinfo=timezone.utc,
)


class FakePortfolioRepository:
    """Repository fake for portfolio read tests."""

    def __init__(
        self,
        portfolio: PortfolioModel | None,
    ) -> None:
        self.portfolio = portfolio
        self.list_called = False
        self.required_portfolio_id: str | None = None
        self.limit: int | None = None
        self.offset: int | None = None

    def list_portfolios(
        self,
        *,
        limit: int,
        offset: int,
    ) -> list[PortfolioModel]:
        self.list_called = True
        self.limit = limit
        self.offset = offset

        if self.portfolio is None:
            return []

        return [self.portfolio]

    def require_portfolio_by_business_id(
        self,
        portfolio_id: str,
    ) -> PortfolioModel:
        self.required_portfolio_id = portfolio_id

        if self.portfolio is None:
            raise RecordNotFoundError(
                f"Portfolio {portfolio_id!r} was not found."
            )

        return self.portfolio


class FakeRebalanceRunRepository:
    """Repository fake for rebalance read tests."""

    def __init__(
        self,
        run: RebalanceRunModel | None,
    ) -> None:
        self.run = run
        self.required_run_id: str | None = None
        self.listed_portfolio_database_id: int | None = None
        self.trades_run_id: str | None = None
        self.audit_run_id: str | None = None
        self.limit: int | None = None
        self.offset: int | None = None

    def require_by_run_id(
        self,
        run_id: str,
    ) -> RebalanceRunModel:
        self.required_run_id = run_id

        if self.run is None:
            raise RecordNotFoundError(
                f"Rebalance run {run_id!r} was not found."
            )

        return self.run

    def list_trades(
        self,
        run_id: str,
        *,
        limit: int,
        offset: int,
    ) -> list[TradeModel]:
        self.require_by_run_id(run_id)
        self.trades_run_id = run_id
        self.limit = limit
        self.offset = offset

        if self.run is None:
            return []

        return list(self.run.trades)

    def list_by_portfolio_database_id(
        self,
        portfolio_database_id: int,
        *,
        limit: int,
        offset: int,
    ) -> list[RebalanceRunModel]:
        self.listed_portfolio_database_id = (
            portfolio_database_id
        )
        self.limit = limit
        self.offset = offset

        if self.run is None:
            return []

        return [self.run]

    def list_audit_records(
        self,
        run_id: str,
        *,
        limit: int,
        offset: int,
    ) -> list[AuditRecordModel]:
        self.require_by_run_id(run_id)
        self.audit_run_id = run_id
        self.limit = limit
        self.offset = offset

        if self.run is None:
            return []

        return [
            trade.audit_record
            for trade in self.run.trades
            if trade.audit_record is not None
        ]


def _build_portfolio() -> PortfolioModel:
    """Build a portfolio model graph without a database."""

    client = ClientModel(
        client_id="C00001",
        risk_category="balanced",
    )
    portfolio = PortfolioModel(
        portfolio_id="P00001",
        portfolio_value=Decimal("1000000.00"),
        currency="USD",
    )
    portfolio.id = 7
    portfolio.client = client
    portfolio.holdings.append(
        PortfolioHoldingModel(
            asset="cash",
            current_weight=Decimal("0.0500000000"),
            current_value=Decimal("50000.00"),
            cost_basis=Decimal("50000.00"),
        )
    )

    return portfolio


def _build_run(
    portfolio: PortfolioModel,
) -> RebalanceRunModel:
    """Build a rebalance model graph without a database."""

    run = RebalanceRunModel(
        run_id="RUN000001",
        status="success",
        portfolio_value=Decimal("1000000.00"),
        transaction_cost_rate=Decimal("0.0020000000"),
        started_at=TIMESTAMP,
        completed_at=TIMESTAMP,
    )
    run.portfolio = portfolio

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
        compliance_explanation="Compliance explanation.",
    )
    trade.approval = ApprovalModel(
        approval_required=False,
        approval_status="NOT_REQUIRED",
        approval_reason="Automatic approval.",
    )
    trade.audit_record = AuditRecordModel(
        audit_id="AUD000001",
        audit_timestamp=TIMESTAMP,
        event_type="TRADE_RECOMMENDATION",
        details="Trade recorded.",
    )

    run.trades.append(trade)

    return run


def _build_service(
    *,
    portfolio: PortfolioModel | None,
    run: RebalanceRunModel | None,
) -> tuple[
    PortfolioReadApplicationService,
    FakePortfolioRepository,
    FakeRebalanceRunRepository,
]:
    """Build a read service with fake repositories."""

    portfolio_repository = FakePortfolioRepository(
        portfolio
    )
    rebalance_repository = FakeRebalanceRunRepository(
        run
    )

    return (
        PortfolioReadApplicationService(
            portfolio_repository=portfolio_repository,
            rebalance_repository=rebalance_repository,
        ),
        portfolio_repository,
        rebalance_repository,
    )


def test_list_portfolios_uses_repository() -> None:
    portfolio = _build_portfolio()
    service, portfolio_repository, _ = _build_service(
        portfolio=portfolio,
        run=None,
    )

    result = service.list_portfolios(
        limit=10,
        offset=5,
    )

    assert portfolio_repository.list_called is True
    assert portfolio_repository.limit == 10
    assert portfolio_repository.offset == 5
    assert result.items[0].portfolio_id == "P00001"
    assert result.items[0].client_id == "C00001"
    assert result.limit == 10
    assert result.offset == 5
    assert result.count == 1


def test_get_portfolio_returns_holdings() -> None:
    portfolio = _build_portfolio()
    service, portfolio_repository, _ = _build_service(
        portfolio=portfolio,
        run=None,
    )

    result = service.get_portfolio("P00001")

    assert (
        portfolio_repository.required_portfolio_id
        == "P00001"
    )
    assert len(result.holdings) == 1
    assert result.holdings[0].asset == "cash"


def test_get_portfolio_preserves_404() -> None:
    service, _, _ = _build_service(
        portfolio=None,
        run=None,
    )

    with pytest.raises(RecordNotFoundError):
        service.get_portfolio("MISSING")


def test_list_portfolio_rebalances_uses_repositories() -> None:
    portfolio = _build_portfolio()
    run = _build_run(portfolio)
    service, _, rebalance_repository = _build_service(
        portfolio=portfolio,
        run=run,
    )

    result = service.list_portfolio_rebalances(
        "P00001",
        limit=7,
        offset=2,
    )

    assert (
        rebalance_repository.listed_portfolio_database_id
        == 7
    )
    assert rebalance_repository.limit == 7
    assert rebalance_repository.offset == 2
    assert result.items[0].run_id == "RUN000001"
    assert (
        result.items[0].transaction_cost
        == Decimal("40.00")
    )


def test_get_rebalance_returns_summary_totals() -> None:
    portfolio = _build_portfolio()
    run = _build_run(portfolio)
    service, _, rebalance_repository = _build_service(
        portfolio=portfolio,
        run=run,
    )

    result = service.get_rebalance("RUN000001")

    assert (
        rebalance_repository.required_run_id
        == "RUN000001"
    )
    assert result.trade_count == 1
    assert result.transaction_cost == Decimal("40.00")
    assert (
        result.estimated_tax_liability
        == Decimal("400.00")
    )
    assert result.approval_required_count == 0
    assert result.pending_approval_count == 0


def test_list_rebalance_trades_uses_repository() -> None:
    portfolio = _build_portfolio()
    run = _build_run(portfolio)
    service, _, rebalance_repository = _build_service(
        portfolio=portfolio,
        run=run,
    )

    result = service.list_rebalance_trades(
        "RUN000001",
        limit=3,
        offset=1,
    )

    assert (
        rebalance_repository.required_run_id
        == "RUN000001"
    )
    assert (
        rebalance_repository.trades_run_id
        == "RUN000001"
    )
    assert rebalance_repository.limit == 3
    assert rebalance_repository.offset == 1
    assert result.items[0].action == "SELL"
    assert (
        result.items[0].current_weight
        == Decimal("0.6000000000")
    )
    assert (
        result.items[0].post_trade_weight
        == Decimal("0.5800000000")
    )
    assert result.items[0].estimated_tax == Decimal("400.00")
    assert result.items[0].threshold_breached is True
    assert result.items[0].threshold_severity == "high"
    assert (
        result.items[0].breach_ratio
        == Decimal("1.5000000000")
    )
    assert result.items[0].final_trigger_type == "threshold"
    assert result.items[0].final_priority == "high"
    assert result.items[0].client_explanation == "Client explanation."
    assert result.items[0].approval is not None
    assert result.items[0].approval.required is False


def test_list_rebalance_audit_returns_audit_entries() -> None:
    portfolio = _build_portfolio()
    run = _build_run(portfolio)
    service, _, rebalance_repository = _build_service(
        portfolio=portfolio,
        run=run,
    )

    result = service.list_rebalance_audit(
        "RUN000001",
        limit=4,
        offset=2,
    )

    assert rebalance_repository.audit_run_id == "RUN000001"
    assert rebalance_repository.limit == 4
    assert rebalance_repository.offset == 2
    assert result.items[0].audit_id == "AUD000001"
    assert result.items[0].approval_status == "NOT_REQUIRED"
    assert result.items[0].timestamp == TIMESTAMP
    assert result.items[0].event_type == "TRADE_RECOMMENDATION"
    assert result.items[0].audit_message == "Trade recorded."
    assert result.items[0].asset == "domestic_equity"
    assert result.items[0].action == "SELL"
    assert (
        result.items[0].approval_reason
        == "Automatic approval."
    )


def test_list_rebalance_trades_handles_missing_approval() -> None:
    portfolio = _build_portfolio()
    run = _build_run(portfolio)
    run.trades[0].approval = None
    service, _, _ = _build_service(
        portfolio=portfolio,
        run=run,
    )

    result = service.list_rebalance_trades(
        "RUN000001",
        limit=20,
        offset=0,
    )

    assert result.items[0].approval is None
