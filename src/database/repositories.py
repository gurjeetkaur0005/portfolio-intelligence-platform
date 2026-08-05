from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, selectinload

from src.database.models import (
    AuditRecordModel,
    ClientModel,
    PortfolioHoldingModel,
    PortfolioModel,
    RebalanceRunModel,
    TradeModel,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


class RepositoryError(RuntimeError):
    """Raised when a repository operation cannot be completed."""


class DuplicateRecordError(RepositoryError):
    """Raised when a unique database record already exists."""


class RecordNotFoundError(RepositoryError):
    """Raised when a requested database record does not exist."""


class PortfolioRepository:
    """
    Persist and retrieve clients, portfolios, and holdings.

    The repository owns SQLAlchemy query and transaction details.
    It does not perform portfolio calculations.
    """

    def __init__(
        self,
        session: Session,
    ) -> None:
        """Initialize the repository with one short-lived session."""

        if not isinstance(session, Session):
            raise TypeError(
                "session must be a SQLAlchemy Session."
            )

        self._session = session

    def create_client(
        self,
        *,
        client_id: str,
        risk_category: str,
    ) -> ClientModel:
        """
        Create and persist one client.

        Raises:
            DuplicateRecordError:
                If client_id already exists.
            RepositoryError:
                If persistence fails.
        """

        client = ClientModel(
            client_id=_validate_non_empty_string(
                client_id,
                "client_id",
            ),
            risk_category=_validate_non_empty_string(
                risk_category,
                "risk_category",
            ),
        )

        self._save(client)

        return client

    def get_client_by_business_id(
        self,
        client_id: str,
    ) -> ClientModel | None:
        """Return one client by its external business ID."""

        normalized_client_id = _validate_non_empty_string(
            client_id,
            "client_id",
        )

        statement = (
            select(ClientModel)
            .where(
                ClientModel.client_id
                == normalized_client_id
            )
            .options(
                selectinload(
                    ClientModel.portfolios
                )
            )
        )

        return self._session.scalar(statement)

    def require_client_by_business_id(
        self,
        client_id: str,
    ) -> ClientModel:
        """
        Return one client or raise RecordNotFoundError.
        """

        client = self.get_client_by_business_id(
            client_id
        )

        if client is None:
            raise RecordNotFoundError(
                f"Client {client_id!r} was not found."
            )

        return client

    def create_portfolio(
        self,
        *,
        client: ClientModel,
        portfolio_id: str,
        portfolio_value: Decimal,
        currency: str = "USD",
    ) -> PortfolioModel:
        """Create and persist one portfolio for a client."""

        if not isinstance(client, ClientModel):
            raise TypeError(
                "client must be a ClientModel."
            )

        normalized_portfolio_value = (
            _validate_positive_decimal(
                portfolio_value,
                "portfolio_value",
            )
        )

        portfolio = PortfolioModel(
            portfolio_id=_validate_non_empty_string(
                portfolio_id,
                "portfolio_id",
            ),
            portfolio_value=normalized_portfolio_value,
            currency=_validate_currency(currency),
        )

        client.portfolios.append(portfolio)

        self._save(portfolio)

        return portfolio

    def get_portfolio_by_business_id(
        self,
        portfolio_id: str,
    ) -> PortfolioModel | None:
        """Return a portfolio with its holdings."""

        normalized_portfolio_id = (
            _validate_non_empty_string(
                portfolio_id,
                "portfolio_id",
            )
        )

        statement = (
            select(PortfolioModel)
            .where(
                PortfolioModel.portfolio_id
                == normalized_portfolio_id
            )
            .options(
                selectinload(
                    PortfolioModel.holdings
                ),
                selectinload(
                    PortfolioModel.rebalance_runs
                ),
                selectinload(
                    PortfolioModel.client
                ),
            )
        )

        return self._session.scalar(statement)

    def list_portfolios(
        self,
        *,
        limit: int,
        offset: int,
    ) -> list[PortfolioModel]:
        """Return all portfolios with their clients."""

        _validate_pagination(limit=limit, offset=offset)

        statement = (
            select(PortfolioModel)
            .options(
                selectinload(
                    PortfolioModel.client
                )
            )
            .order_by(
                PortfolioModel.portfolio_id,
            )
            .limit(limit)
            .offset(offset)
        )

        return list(
            self._session.scalars(statement).all()
        )

    def require_portfolio_by_business_id(
        self,
        portfolio_id: str,
    ) -> PortfolioModel:
        """
        Return one portfolio or raise RecordNotFoundError.
        """

        portfolio = self.get_portfolio_by_business_id(
            portfolio_id
        )

        if portfolio is None:
            raise RecordNotFoundError(
                f"Portfolio {portfolio_id!r} was not found."
            )

        return portfolio

    def replace_holdings(
        self,
        *,
        portfolio: PortfolioModel,
        holdings: Sequence[
            PortfolioHoldingModel
        ],
    ) -> PortfolioModel:
        """
        Replace all holdings belonging to one portfolio.

        This method persists already-created ORM holding models.
        It does not calculate weights, values, or cost basis.
        """

        if not isinstance(portfolio, PortfolioModel):
            raise TypeError(
                "portfolio must be a PortfolioModel."
            )

        validated_holdings = list(holdings)

        for holding in validated_holdings:
            if not isinstance(
                holding,
                PortfolioHoldingModel,
            ):
                raise TypeError(
                    "holdings must contain "
                    "PortfolioHoldingModel objects."
                )

        portfolio.holdings.clear()
        portfolio.holdings.extend(
            validated_holdings
        )

        self._save(portfolio)

        return portfolio

    def _save(
        self,
        model: object,
    ) -> None:
        """Save one ORM graph with commit and rollback."""

        try:
            self._session.add(model)
            self._session.commit()
        except IntegrityError as error:
            self._session.rollback()
            logger.exception(
                "portfolio_repository_integrity_failure "
                "model=%s",
                type(model).__name__,
            )

            raise DuplicateRecordError(
                "The record violates a database "
                "uniqueness or integrity constraint."
            ) from error
        except SQLAlchemyError as error:
            self._session.rollback()
            logger.exception(
                "portfolio_repository_database_failure "
                "model=%s",
                type(model).__name__,
            )

            raise RepositoryError(
                "The database operation failed."
            ) from error


class RebalanceRunRepository:
    """
    Persist and retrieve rebalance runs and their trades.

    A complete ORM graph may contain:

    RebalanceRunModel
        → TradeModel
            → ApprovalModel
            → AuditRecordModel
    """

    def __init__(
        self,
        session: Session,
    ) -> None:
        """Initialize the repository."""

        if not isinstance(session, Session):
            raise TypeError(
                "session must be a SQLAlchemy Session."
            )

        self._session = session

    def save_rebalance_run(
        self,
        rebalance_run: RebalanceRunModel,
    ) -> RebalanceRunModel:
        """
        Persist one complete rebalance workflow graph.

        SQLAlchemy cascades the operation to trades, approvals,
        and audit records.
        """

        if not isinstance(
            rebalance_run,
            RebalanceRunModel,
        ):
            raise TypeError(
                "rebalance_run must be a "
                "RebalanceRunModel."
            )

        try:
            self._session.add(rebalance_run)
            self._session.commit()
        except IntegrityError as error:
            self._session.rollback()
            logger.exception(
                "rebalance_repository_integrity_failure "
                "run_id=%s",
                rebalance_run.run_id,
            )

            raise DuplicateRecordError(
                "The rebalance run violates a database "
                "uniqueness or integrity constraint."
            ) from error
        except SQLAlchemyError as error:
            self._session.rollback()
            logger.exception(
                "rebalance_repository_database_failure "
                "run_id=%s",
                rebalance_run.run_id,
            )

            raise RepositoryError(
                "The rebalance run could not be saved."
            ) from error

        return rebalance_run

    def get_by_run_id(
        self,
        run_id: str,
    ) -> RebalanceRunModel | None:
        """
        Return a complete rebalance run graph.

        Related trades, approvals, and audit records are loaded
        efficiently through select-in loading.
        """

        normalized_run_id = _validate_non_empty_string(
            run_id,
            "run_id",
        )

        statement = (
            select(RebalanceRunModel)
            .where(
                RebalanceRunModel.run_id
                == normalized_run_id
            )
            .options(
                selectinload(
                    RebalanceRunModel.trades
                ).selectinload(
                    TradeModel.approval
                ),
                selectinload(
                    RebalanceRunModel.trades
                ).selectinload(
                    TradeModel.audit_record
                ),
            )
        )

        return self._session.scalar(statement)

    def require_by_run_id(
        self,
        run_id: str,
    ) -> RebalanceRunModel:
        """
        Return one rebalance run or raise RecordNotFoundError.
        """

        rebalance_run = self.get_by_run_id(
            run_id
        )

        if rebalance_run is None:
            raise RecordNotFoundError(
                f"Rebalance run {run_id!r} was not found."
            )

        return rebalance_run

    def list_trades(
        self,
        run_id: str,
        *,
        limit: int,
        offset: int,
    ) -> list[TradeModel]:
        """Return all trades belonging to one rebalance run."""

        rebalance_run = self.require_by_run_id(
            run_id
        )
        _validate_pagination(limit=limit, offset=offset)

        statement = (
            select(TradeModel)
            .where(
                TradeModel.rebalance_run_id
                == rebalance_run.id
            )
            .options(
                selectinload(
                    TradeModel.approval
                ),
                selectinload(
                    TradeModel.audit_record
                ),
            )
            .order_by(TradeModel.id)
            .limit(limit)
            .offset(offset)
        )

        return list(
            self._session.scalars(statement).all()
        )

    def list_by_portfolio_database_id(
        self,
        portfolio_database_id: int,
        *,
        limit: int,
        offset: int,
    ) -> list[RebalanceRunModel]:
        """Return rebalance runs for one database portfolio ID."""

        if not isinstance(portfolio_database_id, int):
            raise TypeError(
                "portfolio_database_id must be an integer."
            )
        _validate_pagination(limit=limit, offset=offset)

        statement = (
            select(RebalanceRunModel)
            .where(
                RebalanceRunModel.portfolio_id
                == portfolio_database_id
            )
            .options(
                selectinload(
                    RebalanceRunModel.trades
                )
            )
            .order_by(
                RebalanceRunModel.started_at.desc(),
                RebalanceRunModel.run_id,
            )
            .limit(limit)
            .offset(offset)
        )

        return list(
            self._session.scalars(statement).all()
        )

    def list_audit_records(
        self,
        run_id: str,
        *,
        limit: int,
        offset: int,
    ) -> list[AuditRecordModel]:
        """Return audit records for one rebalance run."""

        rebalance_run = self.require_by_run_id(
            run_id
        )
        _validate_pagination(limit=limit, offset=offset)

        statement = (
            select(AuditRecordModel)
            .join(TradeModel)
            .where(
                TradeModel.rebalance_run_id
                == rebalance_run.id
            )
            .options(
                selectinload(
                    AuditRecordModel.trade
                ).selectinload(
                    TradeModel.approval
                )
            )
            .order_by(
                AuditRecordModel.audit_timestamp.desc(),
                AuditRecordModel.id.desc(),
            )
            .limit(limit)
            .offset(offset)
        )

        return list(
            self._session.scalars(statement).all()
        )


def _validate_non_empty_string(
    value: str,
    field_name: str,
) -> str:
    """Validate and normalize a required string."""

    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} must be a string."
        )

    normalized_value = value.strip()

    if not normalized_value:
        raise ValueError(
            f"{field_name} must not be empty."
        )

    return normalized_value


def _validate_positive_decimal(
    value: Decimal,
    field_name: str,
) -> Decimal:
    """Validate a positive Decimal value."""

    if not isinstance(value, Decimal):
        raise TypeError(
            f"{field_name} must be a Decimal."
        )

    if not value.is_finite():
        raise ValueError(
            f"{field_name} must be finite."
        )

    if value <= 0:
        raise ValueError(
            f"{field_name} must be positive."
        )

    return value


def _validate_pagination(
    *,
    limit: int,
    offset: int,
) -> None:
    """Validate repository pagination controls."""

    if not isinstance(limit, int):
        raise TypeError("limit must be an integer.")

    if not isinstance(offset, int):
        raise TypeError("offset must be an integer.")

    if limit < 1:
        raise ValueError("limit must be greater than zero.")

    if offset < 0:
        raise ValueError(
            "offset must be greater than or equal to zero."
        )


def _validate_currency(
    currency: str,
) -> str:
    """Validate and normalize a three-letter currency code."""

    normalized_currency = _validate_non_empty_string(
        currency,
        "currency",
    ).upper()

    if len(normalized_currency) != 3:
        raise ValueError(
            "currency must contain exactly three letters."
        )

    if not normalized_currency.isalpha():
        raise ValueError(
            "currency must contain letters only."
        )

    return normalized_currency
