from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.orm import Session

from src.database.models import (
    ClientModel,
    PortfolioModel,
    RebalanceRunModel,
)
from src.database.session import (
    DatabaseSessionFactory,
    get_database_session_factory,
)
from src.utils.logger import get_logger


SMOKE_CLIENT_ID_PATTERN = "C-SMOKE-%"
SMOKE_PORTFOLIO_ID_PATTERN = "P-SMOKE-%"
SMOKE_PORTFOLIO_ID_PREFIX = "P-SMOKE-"

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CleanupResult:
    """Represent the outcome of a smoke-test cleanup run."""

    deleted_portfolios: int
    deleted_clients: int
    deleted_holdings: int
    deleted_rebalance_runs: int
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

    resolved_session_factory = (
        session_factory
        if session_factory is not None
        else get_database_session_factory()
    )

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
        "deleted_rebalance_runs=%s",
        result.deleted_portfolios,
        result.deleted_clients,
        result.deleted_holdings,
        result.deleted_rebalance_runs,
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

    for portfolio in independent_portfolios:
        session.delete(portfolio)

    for client in deletable_clients:
        session.delete(client)

    return CleanupResult(
        deleted_portfolios=len(portfolios_to_delete),
        deleted_clients=len(deletable_clients),
        deleted_holdings=deleted_holdings,
        deleted_rebalance_runs=deleted_rebalance_runs,
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
            ),
        )
        .order_by(PortfolioModel.portfolio_id)
    )

    return list(session.scalars(statement).all())


def _is_smoke_portfolio_id(
    portfolio_id: str,
) -> bool:
    """Return whether a portfolio ID matches the smoke-test prefix."""

    return portfolio_id.startswith(SMOKE_PORTFOLIO_ID_PREFIX)


def main() -> None:
    """Run the smoke-test cleanup script."""

    result = cleanup_smoke_test_data()

    if result.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
