from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.orm import Session

from src.database.models import (
    ClientModel,
    PortfolioModel,
    RebalanceRunModel,
    TradeModel,
)
from src.database.config import get_database_url
from src.database.session import (
    DatabaseSessionFactory,
    get_database_session_factory,
)
from src.utils.logger import get_logger


SMOKE_CLIENT_ID_PATTERN = "C-SMOKE-%"
SMOKE_PORTFOLIO_ID_PATTERN = "P-SMOKE-%"
SMOKE_PORTFOLIO_ID_PREFIX = "P-SMOKE-"
DEV_PORTFOLIO_ID_PREFIX = "DEV-P"
LOCAL_DATABASE_HOSTS = {
    "localhost",
    "127.0.0.1",
    "::1",
}
UNSAFE_DATABASE_NAME_PARTS = {
    "prod",
    "production",
}

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CleanupResult:
    """Represent the outcome of a smoke-test cleanup run."""

    deleted_portfolios: int
    deleted_clients: int
    deleted_holdings: int
    deleted_rebalance_runs: int
    deleted_trades: int = 0
    deleted_approvals: int = 0
    deleted_audit_records: int = 0
    failed: bool = False


def cleanup_smoke_test_data(
    session_factory: DatabaseSessionFactory | None = None,
) -> CleanupResult:
    """
    Remove deterministic smoke-test records from the development database.

    Only records using the smoke-test naming convention are removed.
    Development seed records such as DEV-P00001 through DEV-P00010 are
    outside the cleanup criteria and are preserved.
    """

    if session_factory is None:
        try:
            database_url = get_database_url()
        except ValueError:
            logger.error("smoke_test_cleanup_missing_database_url")
            return CleanupResult(
                deleted_portfolios=0,
                deleted_clients=0,
                deleted_holdings=0,
                deleted_rebalance_runs=0,
                failed=True,
            )

        if not _is_local_development_database_url(database_url):
            logger.error("smoke_test_cleanup_refused_non_local_database")
            return CleanupResult(
                deleted_portfolios=0,
                deleted_clients=0,
                deleted_holdings=0,
                deleted_rebalance_runs=0,
                failed=True,
            )

        resolved_session_factory = get_database_session_factory()
    else:
        resolved_session_factory = session_factory

    with resolved_session_factory() as session:
        try:
            with session.begin():
                result = _cleanup_smoke_records(session)
        except Exception:
            session.rollback()
            logger.exception("smoke_test_cleanup_failed")
            return CleanupResult(
                deleted_portfolios=0,
                deleted_clients=0,
                deleted_holdings=0,
                deleted_rebalance_runs=0,
                failed=True,
            )

    logger.info(
        "smoke_test_cleanup_complete deleted_portfolios=%s "
        "deleted_clients=%s deleted_holdings=%s "
        "deleted_rebalance_runs=%s deleted_trades=%s "
        "deleted_approvals=%s deleted_audit_records=%s",
        result.deleted_portfolios,
        result.deleted_clients,
        result.deleted_holdings,
        result.deleted_rebalance_runs,
        result.deleted_trades,
        result.deleted_approvals,
        result.deleted_audit_records,
    )

    return result


def _cleanup_smoke_records(
    session: Session,
) -> CleanupResult:
    """Stage smoke-test deletes in the active transaction."""

    smoke_clients = _load_smoke_clients(session)
    smoke_portfolios = _load_smoke_portfolios(session)

    portfolios_by_database_id = {
        portfolio.id: portfolio
        for portfolio in smoke_portfolios
    }

    deletable_clients: list[ClientModel] = []
    skipped_client_ids: list[str] = []

    for client in smoke_clients:
        if all(
            _is_smoke_portfolio_id(portfolio.portfolio_id)
            for portfolio in client.portfolios
        ):
            deletable_clients.append(client)

            for portfolio in client.portfolios:
                portfolios_by_database_id[portfolio.id] = portfolio
            continue

        skipped_client_ids.append(client.client_id)

    deletable_client_database_ids = {
        client.id
        for client in deletable_clients
    }

    portfolios_to_delete = list(portfolios_by_database_id.values())
    independent_portfolios = [
        portfolio
        for portfolio in portfolios_to_delete
        if portfolio.client_id not in deletable_client_database_ids
    ]

    portfolio_ids = sorted(
        portfolio.portfolio_id
        for portfolio in portfolios_to_delete
    )
    client_ids = sorted(
        client.client_id
        for client in deletable_clients
    )

    logger.info(
        "smoke_test_cleanup_planned portfolio_ids=%s client_ids=%s",
        portfolio_ids,
        client_ids,
    )

    if skipped_client_ids:
        logger.warning(
            "smoke_test_cleanup_skipped_clients_with_non_smoke_portfolios "
            "client_ids=%s",
            sorted(skipped_client_ids),
        )

    deleted_holdings = sum(
        len(portfolio.holdings)
        for portfolio in portfolios_to_delete
    )
    deleted_rebalance_runs = sum(
        len(portfolio.rebalance_runs)
        for portfolio in portfolios_to_delete
    )
    deleted_trades = sum(
        len(rebalance_run.trades)
        for portfolio in portfolios_to_delete
        for rebalance_run in portfolio.rebalance_runs
    )
    deleted_approvals = sum(
        1
        for portfolio in portfolios_to_delete
        for rebalance_run in portfolio.rebalance_runs
        for trade in rebalance_run.trades
        if trade.approval is not None
    )
    deleted_audit_records = sum(
        1
        for portfolio in portfolios_to_delete
        for rebalance_run in portfolio.rebalance_runs
        for trade in rebalance_run.trades
        if trade.audit_record is not None
    )

    for portfolio in independent_portfolios:
        session.delete(portfolio)

    for client in deletable_clients:
        session.delete(client)

    return CleanupResult(
        deleted_portfolios=len(portfolios_to_delete),
        deleted_clients=len(deletable_clients),
        deleted_holdings=deleted_holdings,
        deleted_rebalance_runs=deleted_rebalance_runs,
        deleted_trades=deleted_trades,
        deleted_approvals=deleted_approvals,
        deleted_audit_records=deleted_audit_records,
    )


def _load_smoke_clients(
    session: Session,
) -> list[ClientModel]:
    """Return smoke-test clients with portfolios loaded."""

    statement = (
        select(ClientModel)
        .where(ClientModel.client_id.like(SMOKE_CLIENT_ID_PATTERN))
        .options(
            selectinload(ClientModel.portfolios).selectinload(
                PortfolioModel.holdings
            ),
            selectinload(ClientModel.portfolios).selectinload(
                PortfolioModel.rebalance_runs
            ).selectinload(
                RebalanceRunModel.trades
            ).selectinload(
                TradeModel.approval
            ),
            selectinload(ClientModel.portfolios).selectinload(
                PortfolioModel.rebalance_runs
            ).selectinload(
                RebalanceRunModel.trades
            ).selectinload(
                TradeModel.audit_record
            ),
        )
        .order_by(ClientModel.client_id)
    )

    return list(session.scalars(statement).all())


def _load_smoke_portfolios(
    session: Session,
) -> list[PortfolioModel]:
    """Return smoke-test portfolios with cascaded children loaded."""

    statement = (
        select(PortfolioModel)
        .where(
            PortfolioModel.portfolio_id.like(
                SMOKE_PORTFOLIO_ID_PATTERN
            )
        )
        .options(
            selectinload(PortfolioModel.holdings),
            selectinload(PortfolioModel.rebalance_runs).selectinload(
                RebalanceRunModel.trades
            ).selectinload(
                TradeModel.approval
            ),
            selectinload(PortfolioModel.rebalance_runs).selectinload(
                RebalanceRunModel.trades
            ).selectinload(
                TradeModel.audit_record
            ),
        )
        .order_by(PortfolioModel.portfolio_id)
    )

    return list(session.scalars(statement).all())


def _is_smoke_portfolio_id(
    portfolio_id: str,
) -> bool:
    """Return whether a portfolio ID matches the smoke-test prefix."""

    return (
        portfolio_id.startswith(SMOKE_PORTFOLIO_ID_PREFIX)
        and not portfolio_id.startswith(DEV_PORTFOLIO_ID_PREFIX)
    )


def _is_local_development_database_url(
    database_url: str,
) -> bool:
    """Return whether a database URL is safe for local cleanup."""

    parsed_url = urlparse(database_url)

    if parsed_url.scheme.startswith("sqlite"):
        return True

    if not parsed_url.scheme.startswith("postgresql"):
        return False

    hostname = parsed_url.hostname

    if hostname not in LOCAL_DATABASE_HOSTS:
        return False

    database_name = parsed_url.path.rsplit("/", maxsplit=1)[-1].lower()

    return not any(
        unsafe_part in database_name
        for unsafe_part in UNSAFE_DATABASE_NAME_PARTS
    )


def main() -> None:
    """Run the smoke-test cleanup script."""

    result = cleanup_smoke_test_data()

    if result.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
