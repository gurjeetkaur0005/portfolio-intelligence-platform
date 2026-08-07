from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Protocol
from uuid import uuid4

import pandas as pd

from src.database.models import (
    PortfolioModel,
    RebalanceRunModel,
    utc_now,
)
from src.pipeline.rebalance_pipeline import (
    run_rebalance_pipeline_for_inputs,
)
from src.services.portfolio_input_adapter import (
    DeterministicPortfolioInput,
    PortfolioInputAdapter,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


class PortfolioRepositoryProtocol(Protocol):
    """Describe the portfolio repository operation used by the service."""

    def require_portfolio_by_business_id(
        self,
        portfolio_id: str,
    ) -> PortfolioModel:
        """Return a persisted portfolio or raise RecordNotFoundError."""
        ...


class PortfolioInputAdapterProtocol(Protocol):
    """Describe persisted-portfolio input adaptation."""

    def build_input(
        self,
        portfolio: PortfolioModel,
    ) -> DeterministicPortfolioInput:
        """Build deterministic pipeline input."""
        ...


class RebalancePipelineProtocol(Protocol):
    """Describe the deterministic portfolio engine entry point."""

    def __call__(
        self,
        *,
        client_profiles: pd.DataFrame,
        portfolios: pd.DataFrame,
        portfolio_value: float = 1_000_000.0,
        transaction_cost_rate: float = 0.002,
    ) -> pd.DataFrame:
        """Run deterministic rebalancing for supplied inputs."""
        ...


class RebalancePersistenceProtocol(Protocol):
    """Describe the persistence operation used by the service."""

    def persist_rebalance_result(
        self,
        *,
        portfolio: PortfolioModel,
        trade_results: pd.DataFrame,
        portfolio_value: Decimal,
        transaction_cost_rate: Decimal,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        run_id: str | None = None,
        status: str = "SUCCESS",
    ) -> RebalanceRunModel:
        """Persist one complete rebalance result."""
        ...


class RebalanceApplicationServiceError(RuntimeError):
    """Base exception for application-service failures."""


class RebalanceExecutionError(
    RebalanceApplicationServiceError
):
    """Raised when the deterministic workflow cannot complete."""


class RebalancePersistenceError(
    RebalanceApplicationServiceError
):
    """Raised when a completed workflow cannot be persisted."""


@dataclass(frozen=True, slots=True)
class PersistedRebalanceResult:
    """Store the outcome of a persisted rebalance workflow."""

    portfolio_id: str
    run_id: str
    workflow_status: str
    workflow_name: str
    workflow_message: str
    trade_count: int
    database_run_id: int


class RebalanceApplicationService:
    """
    Coordinate portfolio loading, rebalancing, and persistence.

    This service does not calculate portfolio drift, optimize weights,
    create trades, estimate costs, calculate taxes, or generate
    explanations. Those responsibilities remain in the existing
    deterministic workflow.
    """

    def __init__(
        self,
        portfolio_repository: PortfolioRepositoryProtocol,
        persistence_service: RebalancePersistenceProtocol,
        portfolio_input_adapter: (
            PortfolioInputAdapterProtocol | None
        ) = None,
        portfolio_engine: RebalancePipelineProtocol = (
            run_rebalance_pipeline_for_inputs
        ),
    ) -> None:
        """Initialize the application service dependencies."""

        self._portfolio_repository = portfolio_repository
        self._persistence_service = persistence_service
        self._portfolio_input_adapter = (
            portfolio_input_adapter
            if portfolio_input_adapter is not None
            else PortfolioInputAdapter()
        )
        self._portfolio_engine = portfolio_engine

    def execute_rebalance(
        self,
        *,
        portfolio_id: str,
        transaction_cost_rate: Decimal,
        run_id: str | None = None,
    ) -> PersistedRebalanceResult:
        """
        Execute and persist one portfolio rebalance workflow.

        Args:
            portfolio_id:
                External business identifier of the persisted portfolio.
            transaction_cost_rate:
                Transaction-cost rate expressed as a decimal.
            run_id:
                Optional external identifier for the persisted run.

        Returns:
            Structured information about the persisted workflow.

        Raises:
            TypeError:
                If arguments have invalid types.
            ValueError:
                If values violate the service contract.
            RecordNotFoundError:
                If the requested portfolio does not exist.
            RebalanceExecutionError:
                If the deterministic workflow fails.
            RebalancePersistenceError:
                If persistence fails after successful execution.
        """

        normalized_portfolio_id = _validate_non_empty_string(
            portfolio_id,
            "portfolio_id",
        )
        normalized_transaction_cost_rate = _validate_rate(
            transaction_cost_rate,
            "transaction_cost_rate",
        )
        normalized_run_id = (
            _validate_non_empty_string(
                run_id,
                "run_id",
            )
            if run_id is not None
            else _generate_run_id()
        )

        portfolio = (
            self._portfolio_repository
            .require_portfolio_by_business_id(
                normalized_portfolio_id
            )
        )
        normalized_portfolio_value = _validate_positive_decimal(
            portfolio.portfolio_value,
            "portfolio.portfolio_value",
        )
        started_at = utc_now()
        logger.info(
            "rebalance_start portfolio_id=%s run_id=%s",
            normalized_portfolio_id,
            normalized_run_id,
        )

        deterministic_input = (
            self._portfolio_input_adapter.build_input(
                portfolio
            )
        )

        trade_results = _execute_portfolio_engine(
            portfolio_engine=self._portfolio_engine,
            deterministic_input=deterministic_input,
            portfolio_value=normalized_portfolio_value,
            transaction_cost_rate=(
                normalized_transaction_cost_rate
            ),
            portfolio_id=normalized_portfolio_id,
        )

        completed_at = utc_now()

        try:
            persisted_run = (
                self._persistence_service
                .persist_rebalance_result(
                    portfolio=portfolio,
                    trade_results=trade_results,
                    portfolio_value=(
                        normalized_portfolio_value
                    ),
                    transaction_cost_rate=(
                        normalized_transaction_cost_rate
                    ),
                    run_id=normalized_run_id,
                    status="success",
                    started_at=started_at,
                    completed_at=completed_at,
                )
            )
        except Exception as error:
            logger.exception(
                "rebalance_persistence_failed portfolio_id=%s "
                "run_id=%s",
                normalized_portfolio_id,
                normalized_run_id,
            )
            raise RebalancePersistenceError(
                "The rebalance workflow completed, but its "
                "result could not be persisted."
            ) from error

        if persisted_run.id is None:
            logger.error(
                "rebalance_persistence_missing_database_id "
                "portfolio_id=%s run_id=%s",
                normalized_portfolio_id,
                normalized_run_id,
            )
            raise RebalancePersistenceError(
                "The persisted rebalance run did not receive "
                "a database identifier."
            )

        logger.info(
            "rebalance_complete portfolio_id=%s run_id=%s "
            "database_run_id=%s trade_count=%s",
            normalized_portfolio_id,
            persisted_run.run_id,
            persisted_run.id,
            len(trade_results),
        )

        return PersistedRebalanceResult(
            portfolio_id=normalized_portfolio_id,
            run_id=persisted_run.run_id,
            workflow_status="success",
            workflow_name="portfolio_rebalancing",
            workflow_message=(
                "The database-backed rebalance workflow "
                "completed successfully."
            ),
            trade_count=len(trade_results),
            database_run_id=persisted_run.id,
        )


def _execute_portfolio_engine(
    *,
    portfolio_engine: RebalancePipelineProtocol,
    deterministic_input: DeterministicPortfolioInput,
    portfolio_value: Decimal,
    transaction_cost_rate: Decimal,
    portfolio_id: str,
) -> pd.DataFrame:
    """Run and validate the database-backed deterministic workflow."""

    try:
        trade_results = portfolio_engine(
            client_profiles=(
                deterministic_input.client_profiles
            ),
            portfolios=deterministic_input.portfolios,
            portfolio_value=float(portfolio_value),
            transaction_cost_rate=float(
                transaction_cost_rate
            ),
        )
    except Exception as error:
        logger.exception(
            "rebalance_engine_failed portfolio_id=%s",
            portfolio_id,
        )
        raise RebalanceExecutionError(
            "The rebalance workflow could not be completed."
        ) from error

    if not isinstance(trade_results, pd.DataFrame):
        raise RebalanceExecutionError(
            "The rebalance workflow result must be "
            "a pandas DataFrame."
        )

    if trade_results.empty:
        raise RebalanceExecutionError(
            "The rebalance workflow returned no trade rows."
        )

    if "portfolio_id" not in trade_results.columns:
        raise RebalanceExecutionError(
            "The rebalance result is missing the "
            "portfolio_id column."
        )

    portfolio_rows = trade_results.loc[
        trade_results["portfolio_id"].astype(str)
        == portfolio_id
    ].copy(deep=True)

    if portfolio_rows.empty:
        raise RebalanceExecutionError(
            "The rebalance workflow returned no rows for "
            "the requested portfolio."
        )

    return portfolio_rows.reset_index(
        drop=True
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
    """Validate a positive finite Decimal value."""

    if not isinstance(value, Decimal):
        raise TypeError(
            f"{field_name} must be a Decimal."
        )

    if not value.is_finite():
        raise ValueError(
            f"{field_name} must be finite."
        )

    if value <= Decimal("0"):
        raise ValueError(
            f"{field_name} must be positive."
        )

    return value


def _validate_rate(
    value: Decimal,
    field_name: str,
) -> Decimal:
    """Validate a finite rate between zero and one."""

    if not isinstance(value, Decimal):
        raise TypeError(
            f"{field_name} must be a Decimal."
        )

    if not value.is_finite():
        raise ValueError(
            f"{field_name} must be finite."
        )

    if not Decimal("0") <= value <= Decimal("1"):
        raise ValueError(
            f"{field_name} must be between 0 and 1."
        )

    return value


def _generate_run_id() -> str:
    """Generate an external rebalance-run identifier."""

    return f"RUN-{uuid4().hex.upper()}"
