from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Final

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)

from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import DatabaseBase


IDENTIFIER_LENGTH: Final = 50
RISK_CATEGORY_LENGTH: Final = 50
ASSET_NAME_LENGTH: Final = 100
CURRENCY_LENGTH: Final = 3

MONEY_PRECISION: Final = 18
MONEY_SCALE: Final = 2

WEIGHT_PRECISION: Final = 12
WEIGHT_SCALE: Final = 10
ACTION_LENGTH: Final = 10
STATUS_LENGTH: Final = 30
TRIGGER_TYPE_LENGTH: Final = 50
PRIORITY_LENGTH: Final = 30
SEVERITY_LENGTH: Final = 30
WORKFLOW_STATUS_LENGTH: Final = 30

TEXT_LENGTH: Final = 2_000

def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


class ClientModel(DatabaseBase):
    """Persist one portfolio-management client."""

    __tablename__ = "clients"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    client_id: Mapped[str] = mapped_column(
        String(IDENTIFIER_LENGTH),
        nullable=False,
        unique=True,
        index=True,
    )

    risk_category: Mapped[str] = mapped_column(
        String(RISK_CATEGORY_LENGTH),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    portfolios: Mapped[list[PortfolioModel]] = relationship(
        back_populates="client",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        """Return a developer-friendly model representation."""

        return (
            "ClientModel("
            f"id={self.id!r}, "
            f"client_id={self.client_id!r}, "
            f"risk_category={self.risk_category!r}"
            ")"
        )


class PortfolioModel(DatabaseBase):
    """Persist one client portfolio."""

    __tablename__ = "portfolios"

    __table_args__ = (
        CheckConstraint(
            "portfolio_value > 0",
            name="ck_portfolios_positive_value",
        ),
        Index(
            "ix_portfolios_client_id_created_at",
            "client_id",
            "created_at",
        ),
    )

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    portfolio_id: Mapped[str] = mapped_column(
        String(IDENTIFIER_LENGTH),
        nullable=False,
        unique=True,
        index=True,
    )

    client_id: Mapped[int] = mapped_column(
        ForeignKey(
            "clients.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    portfolio_value: Mapped[Decimal] = mapped_column(
        Numeric(
            precision=MONEY_PRECISION,
            scale=MONEY_SCALE,
        ),
        nullable=False,
    )

    currency: Mapped[str] = mapped_column(
        String(CURRENCY_LENGTH),
        nullable=False,
        default="USD",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    client: Mapped[ClientModel] = relationship(
        back_populates="portfolios",
    )

    holdings: Mapped[list[PortfolioHoldingModel]] = relationship(
        back_populates="portfolio",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    rebalance_runs: Mapped[list[RebalanceRunModel]] = relationship(
        back_populates="portfolio",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        """Return a developer-friendly model representation."""

        return (
            "PortfolioModel("
            f"id={self.id!r}, "
            f"portfolio_id={self.portfolio_id!r}, "
            f"client_id={self.client_id!r}, "
            f"portfolio_value={self.portfolio_value!r}"
            ")"
        )


class PortfolioHoldingModel(DatabaseBase):
    """Persist one asset holding inside a portfolio."""

    __tablename__ = "portfolio_holdings"

    __table_args__ = (
        UniqueConstraint(
            "portfolio_id",
            "asset",
            name="uq_portfolio_holdings_portfolio_asset",
        ),
        CheckConstraint(
            "current_weight >= 0 AND current_weight <= 1",
            name="ck_portfolio_holdings_weight_range",
        ),
        CheckConstraint(
            "current_value >= 0",
            name="ck_portfolio_holdings_non_negative_value",
        ),
        CheckConstraint(
            "cost_basis >= 0",
            name="ck_portfolio_holdings_non_negative_cost_basis",
        ),
    )

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    portfolio_id: Mapped[int] = mapped_column(
        ForeignKey(
            "portfolios.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    asset: Mapped[str] = mapped_column(
        String(ASSET_NAME_LENGTH),
        nullable=False,
    )

    current_weight: Mapped[Decimal] = mapped_column(
        Numeric(
            precision=WEIGHT_PRECISION,
            scale=WEIGHT_SCALE,
        ),
        nullable=False,
    )

    current_value: Mapped[Decimal] = mapped_column(
        Numeric(
            precision=MONEY_PRECISION,
            scale=MONEY_SCALE,
        ),
        nullable=False,
    )

    cost_basis: Mapped[Decimal] = mapped_column(
        Numeric(
            precision=MONEY_PRECISION,
            scale=MONEY_SCALE,
        ),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    portfolio: Mapped[PortfolioModel] = relationship(
        back_populates="holdings",
    )

    def __repr__(self) -> str:
        """Return a developer-friendly model representation."""

        return (
            "PortfolioHoldingModel("
            f"id={self.id!r}, "
            f"portfolio_id={self.portfolio_id!r}, "
            f"asset={self.asset!r}, "
            f"current_weight={self.current_weight!r}"
            ")"
        )
class RebalanceRunModel(DatabaseBase):
    """Persist one execution of the rebalance workflow."""

    __tablename__ = "rebalance_runs"

    __table_args__ = (
        CheckConstraint(
            "portfolio_value > 0",
            name="ck_rebalance_runs_positive_portfolio_value",
        ),
        CheckConstraint(
            (
                "transaction_cost_rate >= 0 "
                "AND transaction_cost_rate <= 1"
            ),
            name="ck_rebalance_runs_cost_rate_range",
        ),
        Index(
            "ix_rebalance_runs_portfolio_started_at",
            "portfolio_id",
            "started_at",
        ),
    )

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    run_id: Mapped[str] = mapped_column(
        String(IDENTIFIER_LENGTH),
        nullable=False,
        unique=True,
        index=True,
    )

    portfolio_id: Mapped[int] = mapped_column(
        ForeignKey(
            "portfolios.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(WORKFLOW_STATUS_LENGTH),
        nullable=False,
    )

    portfolio_value: Mapped[Decimal] = mapped_column(
        Numeric(
            precision=MONEY_PRECISION,
            scale=MONEY_SCALE,
        ),
        nullable=False,
    )

    transaction_cost_rate: Mapped[Decimal] = mapped_column(
        Numeric(
            precision=WEIGHT_PRECISION,
            scale=WEIGHT_SCALE,
        ),
        nullable=False,
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    portfolio: Mapped[PortfolioModel] = relationship(
        back_populates="rebalance_runs",
    )

    trades: Mapped[list[TradeModel]] = relationship(
        back_populates="rebalance_run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""

        return (
            "RebalanceRunModel("
            f"id={self.id!r}, "
            f"run_id={self.run_id!r}, "
            f"portfolio_id={self.portfolio_id!r}, "
            f"status={self.status!r}"
            ")"
        )


class TradeModel(DatabaseBase):
    """Persist one asset-level trade recommendation."""

    __tablename__ = "trades"

    __table_args__ = (
        UniqueConstraint(
            "rebalance_run_id",
            "asset",
            name="uq_trades_run_asset",
        ),
        CheckConstraint(
            "action IN ('BUY', 'SELL', 'HOLD')",
            name="ck_trades_valid_action",
        ),
        CheckConstraint(
            "current_weight >= 0 AND current_weight <= 1",
            name="ck_trades_current_weight_range",
        ),
        CheckConstraint(
            "post_trade_weight >= 0 AND post_trade_weight <= 1",
            name="ck_trades_post_trade_weight_range",
        ),
        CheckConstraint(
            "transaction_cost >= 0",
            name="ck_trades_non_negative_transaction_cost",
        ),
        CheckConstraint(
            "estimated_tax_liability >= 0",
            name="ck_trades_non_negative_tax",
        ),
        Index(
            "ix_trades_run_action",
            "rebalance_run_id",
            "action",
        ),
    )

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    rebalance_run_id: Mapped[int] = mapped_column(
        ForeignKey(
            "rebalance_runs.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    asset: Mapped[str] = mapped_column(
        String(ASSET_NAME_LENGTH),
        nullable=False,
    )

    action: Mapped[str] = mapped_column(
        String(ACTION_LENGTH),
        nullable=False,
    )

    current_weight: Mapped[Decimal] = mapped_column(
        Numeric(
            precision=WEIGHT_PRECISION,
            scale=WEIGHT_SCALE,
        ),
        nullable=False,
    )

    trade_weight: Mapped[Decimal] = mapped_column(
        Numeric(
            precision=WEIGHT_PRECISION,
            scale=WEIGHT_SCALE,
        ),
        nullable=False,
    )

    post_trade_weight: Mapped[Decimal] = mapped_column(
        Numeric(
            precision=WEIGHT_PRECISION,
            scale=WEIGHT_SCALE,
        ),
        nullable=False,
    )

    trade_value: Mapped[Decimal] = mapped_column(
        Numeric(
            precision=MONEY_PRECISION,
            scale=MONEY_SCALE,
        ),
        nullable=False,
    )

    transaction_cost: Mapped[Decimal] = mapped_column(
        Numeric(
            precision=MONEY_PRECISION,
            scale=MONEY_SCALE,
        ),
        nullable=False,
    )

    estimated_tax_liability: Mapped[Decimal] = mapped_column(
        Numeric(
            precision=MONEY_PRECISION,
            scale=MONEY_SCALE,
        ),
        nullable=False,
    )

    threshold_breached: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )

    threshold_severity: Mapped[str] = mapped_column(
        String(SEVERITY_LENGTH),
        nullable=False,
    )

    breach_ratio: Mapped[Decimal] = mapped_column(
        Numeric(
            precision=WEIGHT_PRECISION,
            scale=WEIGHT_SCALE,
        ),
        nullable=False,
    )

    final_trigger_type: Mapped[str] = mapped_column(
        String(TRIGGER_TYPE_LENGTH),
        nullable=False,
    )

    final_priority: Mapped[str] = mapped_column(
        String(PRIORITY_LENGTH),
        nullable=False,
    )

    contributing_triggers: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    client_explanation: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    advisor_explanation: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    compliance_explanation: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    rebalance_run: Mapped[RebalanceRunModel] = relationship(
        back_populates="trades",
    )

    approval: Mapped[ApprovalModel | None] = relationship(
        back_populates="trade",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )

    audit_record: Mapped[AuditRecordModel | None] = relationship(
        back_populates="trade",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""

        return (
            "TradeModel("
            f"id={self.id!r}, "
            f"rebalance_run_id={self.rebalance_run_id!r}, "
            f"asset={self.asset!r}, "
            f"action={self.action!r}"
            ")"
        )


class ApprovalModel(DatabaseBase):
    """Persist the human-approval decision for one trade."""

    __tablename__ = "approvals"

    __table_args__ = (
        UniqueConstraint(
            "trade_id",
            name="uq_approvals_trade_id",
        ),
    )

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    trade_id: Mapped[int] = mapped_column(
        ForeignKey(
            "trades.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    approval_required: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )

    approval_status: Mapped[str] = mapped_column(
        String(STATUS_LENGTH),
        nullable=False,
    )

    approval_reason: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    reviewed_by: Mapped[str | None] = mapped_column(
        String(IDENTIFIER_LENGTH),
        nullable=True,
    )

    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    trade: Mapped[TradeModel] = relationship(
        back_populates="approval",
    )

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""

        return (
            "ApprovalModel("
            f"id={self.id!r}, "
            f"trade_id={self.trade_id!r}, "
            f"approval_status={self.approval_status!r}"
            ")"
        )


class AuditRecordModel(DatabaseBase):
    """Persist the immutable audit record for one trade."""

    __tablename__ = "audit_records"

    __table_args__ = (
        UniqueConstraint(
            "trade_id",
            name="uq_audit_records_trade_id",
        ),
    )

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    audit_id: Mapped[str] = mapped_column(
        String(IDENTIFIER_LENGTH),
        nullable=False,
        unique=True,
        index=True,
    )

    trade_id: Mapped[int] = mapped_column(
        ForeignKey(
            "trades.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    audit_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    event_type: Mapped[str] = mapped_column(
        String(TRIGGER_TYPE_LENGTH),
        nullable=False,
        default="TRADE_RECOMMENDATION",
    )

    details: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    trade: Mapped[TradeModel] = relationship(
        back_populates="audit_record",
    )

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""

        return (
            "AuditRecordModel("
            f"id={self.id!r}, "
            f"audit_id={self.audit_id!r}, "
            f"trade_id={self.trade_id!r}"
            ")"
        )
