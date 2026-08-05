from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Protocol

from src.database.models import (
    AuditRecordModel,
    PortfolioModel,
    RebalanceRunModel,
    TradeModel,
)


class PortfolioReadRepositoryProtocol(Protocol):
    """Describe portfolio read operations."""

    def list_portfolios(self) -> list[PortfolioModel]:
        """Return all persisted portfolios."""
        ...

    def require_portfolio_by_business_id(
        self,
        portfolio_id: str,
    ) -> PortfolioModel:
        """Return one portfolio or raise RecordNotFoundError."""
        ...


class RebalanceReadRepositoryProtocol(Protocol):
    """Describe rebalance read operations."""

    def require_by_run_id(
        self,
        run_id: str,
    ) -> RebalanceRunModel:
        """Return one rebalance run or raise RecordNotFoundError."""
        ...

    def list_trades(
        self,
        run_id: str,
    ) -> list[TradeModel]:
        """Return all trades for one rebalance run."""
        ...

    def list_by_portfolio_database_id(
        self,
        portfolio_database_id: int,
    ) -> list[RebalanceRunModel]:
        """Return runs for one portfolio database ID."""
        ...


@dataclass(frozen=True, slots=True)
class PortfolioSummary:
    """Represent a persisted portfolio summary."""

    portfolio_id: str
    client_id: str
    portfolio_value: Decimal
    currency: str


@dataclass(frozen=True, slots=True)
class PortfolioHolding:
    """Represent one persisted portfolio holding."""

    asset: str
    current_weight: Decimal
    current_value: Decimal
    cost_basis: Decimal


@dataclass(frozen=True, slots=True)
class PortfolioDetail:
    """Represent a portfolio with holdings."""

    portfolio_id: str
    client_id: str
    portfolio_value: Decimal
    currency: str
    holdings: tuple[PortfolioHolding, ...]


@dataclass(frozen=True, slots=True)
class RebalanceRunSummary:
    """Represent summarized rebalance run data."""

    run_id: str
    status: str
    created_at: datetime
    transaction_cost: Decimal
    portfolio_value: Decimal


@dataclass(frozen=True, slots=True)
class RebalanceRunDetail:
    """Represent one rebalance run summary."""

    run_id: str
    portfolio_id: str
    status: str
    created_at: datetime
    completed_at: datetime | None
    portfolio_value: Decimal
    transaction_cost_rate: Decimal
    trade_count: int
    transaction_cost: Decimal
    estimated_tax_liability: Decimal


@dataclass(frozen=True, slots=True)
class RebalanceTradeDetail:
    """Represent one persisted rebalance trade."""

    asset: str
    action: str
    trade_weight: Decimal
    trade_value: Decimal
    estimated_tax: Decimal
    estimated_transaction_cost: Decimal


@dataclass(frozen=True, slots=True)
class RebalanceAuditEntry:
    """Represent one persisted audit entry."""

    approval_status: str | None
    timestamp: datetime
    audit_message: str


class PortfolioReadApplicationService:
    """Coordinate read-only portfolio and rebalance queries."""

    def __init__(
        self,
        portfolio_repository: PortfolioReadRepositoryProtocol,
        rebalance_repository: RebalanceReadRepositoryProtocol,
    ) -> None:
        """Initialize the read service."""

        self._portfolio_repository = portfolio_repository
        self._rebalance_repository = rebalance_repository

    def list_portfolios(
        self,
    ) -> list[PortfolioSummary]:
        """Return summaries for all portfolios."""

        return [
            _portfolio_summary(portfolio)
            for portfolio in (
                self._portfolio_repository.list_portfolios()
            )
        ]

    def get_portfolio(
        self,
        portfolio_id: str,
    ) -> PortfolioDetail:
        """Return one portfolio with holdings."""

        portfolio = (
            self._portfolio_repository
            .require_portfolio_by_business_id(
                portfolio_id
            )
        )

        return _portfolio_detail(portfolio)

    def list_portfolio_rebalances(
        self,
        portfolio_id: str,
    ) -> list[RebalanceRunSummary]:
        """Return rebalance summaries for one portfolio."""

        portfolio = (
            self._portfolio_repository
            .require_portfolio_by_business_id(
                portfolio_id
            )
        )

        runs = (
            self._rebalance_repository
            .list_by_portfolio_database_id(
                _required_database_id(
                    portfolio.id,
                    "portfolio",
                )
            )
        )

        return [
            _rebalance_run_summary(run)
            for run in runs
        ]

    def get_rebalance(
        self,
        run_id: str,
    ) -> RebalanceRunDetail:
        """Return one rebalance run summary."""

        return _rebalance_run_detail(
            self._rebalance_repository.require_by_run_id(
                run_id
            )
        )

    def list_rebalance_trades(
        self,
        run_id: str,
    ) -> list[RebalanceTradeDetail]:
        """Return all trades for one rebalance run."""

        self._rebalance_repository.require_by_run_id(
            run_id
        )

        return [
            _rebalance_trade_detail(trade)
            for trade in self._rebalance_repository.list_trades(
                run_id
            )
        ]

    def list_rebalance_audit(
        self,
        run_id: str,
    ) -> list[RebalanceAuditEntry]:
        """Return audit entries for one rebalance run."""

        run = self._rebalance_repository.require_by_run_id(
            run_id
        )

        return [
            _rebalance_audit_entry(
                audit_record=trade.audit_record,
                approval_status=(
                    trade.approval.approval_status
                    if trade.approval is not None
                    else None
                ),
            )
            for trade in run.trades
            if trade.audit_record is not None
        ]


def _portfolio_summary(
    portfolio: PortfolioModel,
) -> PortfolioSummary:
    """Build a portfolio summary DTO."""

    return PortfolioSummary(
        portfolio_id=portfolio.portfolio_id,
        client_id=portfolio.client.client_id,
        portfolio_value=portfolio.portfolio_value,
        currency=portfolio.currency,
    )


def _portfolio_detail(
    portfolio: PortfolioModel,
) -> PortfolioDetail:
    """Build a portfolio detail DTO."""

    holdings = tuple(
        PortfolioHolding(
            asset=holding.asset,
            current_weight=holding.current_weight,
            current_value=holding.current_value,
            cost_basis=holding.cost_basis,
        )
        for holding in portfolio.holdings
    )

    return PortfolioDetail(
        portfolio_id=portfolio.portfolio_id,
        client_id=portfolio.client.client_id,
        portfolio_value=portfolio.portfolio_value,
        currency=portfolio.currency,
        holdings=holdings,
    )


def _rebalance_run_summary(
    run: RebalanceRunModel,
) -> RebalanceRunSummary:
    """Build a rebalance summary DTO."""

    return RebalanceRunSummary(
        run_id=run.run_id,
        status=run.status,
        created_at=run.started_at,
        transaction_cost=_total_transaction_cost(run),
        portfolio_value=run.portfolio_value,
    )


def _rebalance_run_detail(
    run: RebalanceRunModel,
) -> RebalanceRunDetail:
    """Build a rebalance detail DTO."""

    return RebalanceRunDetail(
        run_id=run.run_id,
        portfolio_id=run.portfolio.portfolio_id,
        status=run.status,
        created_at=run.started_at,
        completed_at=run.completed_at,
        portfolio_value=run.portfolio_value,
        transaction_cost_rate=run.transaction_cost_rate,
        trade_count=len(run.trades),
        transaction_cost=_total_transaction_cost(run),
        estimated_tax_liability=_total_estimated_tax(run),
    )


def _rebalance_trade_detail(
    trade: TradeModel,
) -> RebalanceTradeDetail:
    """Build a rebalance trade DTO."""

    return RebalanceTradeDetail(
        asset=trade.asset,
        action=trade.action,
        trade_weight=trade.trade_weight,
        trade_value=trade.trade_value,
        estimated_tax=trade.estimated_tax_liability,
        estimated_transaction_cost=trade.transaction_cost,
    )


def _rebalance_audit_entry(
    *,
    audit_record: AuditRecordModel | None,
    approval_status: str | None,
) -> RebalanceAuditEntry:
    """Build a rebalance audit DTO."""

    if audit_record is None:
        raise ValueError(
            "audit_record must not be None."
        )

    return RebalanceAuditEntry(
        approval_status=approval_status,
        timestamp=audit_record.audit_timestamp,
        audit_message=audit_record.details,
    )


def _total_transaction_cost(
    run: RebalanceRunModel,
) -> Decimal:
    """Return the stored transaction-cost total."""

    return sum(
        (trade.transaction_cost for trade in run.trades),
        Decimal("0"),
    )


def _total_estimated_tax(
    run: RebalanceRunModel,
) -> Decimal:
    """Return the stored estimated-tax total."""

    return sum(
        (
            trade.estimated_tax_liability
            for trade in run.trades
        ),
        Decimal("0"),
    )


def _required_database_id(
    value: int | None,
    model_name: str,
) -> int:
    """Return a persisted model identifier."""

    if value is None:
        raise ValueError(
            f"{model_name} must have a database identifier."
        )

    return value
