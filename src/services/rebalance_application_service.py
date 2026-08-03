from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol
from uuid import uuid4

import pandas as pd

from src.agents.orchestrator_agent import (
    AgentExecutionStatus,
    OrchestratorRequest,
    OrchestratorResponse,
)
from src.database.models import (
    PortfolioModel,
    RebalanceRunModel,
)
from src.database.repositories import (
    RecordNotFoundError,
)


class PortfolioRepositoryProtocol(Protocol):
    """Describe the portfolio repository operation used by the service."""

    def require_portfolio_by_business_id(
        self,
        portfolio_id: str,
    ) -> PortfolioModel:
        """Return a persisted portfolio or raise RecordNotFoundError."""
        ...


class OrchestratorProtocol(Protocol):
    """Describe the deterministic rebalance operation."""

    def execute_rebalance(
        self,
        request: OrchestratorRequest,
    ) -> OrchestratorResponse:
        """Execute the existing deterministic rebalance workflow."""
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
    """Raised when the orchestrator cannot complete the workflow."""


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
        orchestrator: OrchestratorProtocol,
        persistence_service: RebalancePersistenceProtocol,
    ) -> None:
        """Initialize the application service dependencies."""

        self._portfolio_repository = portfolio_repository
        self._orchestrator = orchestrator
        self._persistence_service = persistence_service

    def execute_rebalance(
        self,
        *,
        portfolio_id: str,
        portfolio_value: Decimal,
        transaction_cost_rate: Decimal,
        run_id: str | None = None,
    ) -> PersistedRebalanceResult:
        """
        Execute and persist one portfolio rebalance workflow.

        Args:
            portfolio_id:
                External business identifier of the persisted portfolio.
            portfolio_value:
                Monetary value used by the deterministic workflow.
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
        normalized_portfolio_value = _validate_positive_decimal(
            portfolio_value,
            "portfolio_value",
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

        orchestrator_request = OrchestratorRequest(
            number_of_clients=1,
            evaluation_date=None,
            portfolio_value=float(
                normalized_portfolio_value
            ),
            transaction_cost_rate=float(
                normalized_transaction_cost_rate
            ),
        )

        orchestrator_response = (
            self._orchestrator.execute_rebalance(
                orchestrator_request
            )
        )

        trade_results = _extract_successful_result(
            orchestrator_response
        )

        portfolio_trade_results = _extract_portfolio_rows(
            trade_results=trade_results,
            portfolio_id=normalized_portfolio_id,
        )

        try:
            persisted_run = (
                self._persistence_service
                .persist_rebalance_result(
                    portfolio=portfolio,
                    trade_results=portfolio_trade_results,
                    portfolio_value=(
                        normalized_portfolio_value
                    ),
                    transaction_cost_rate=(
                        normalized_transaction_cost_rate
                    ),
                    run_id=normalized_run_id,
                    status=orchestrator_response.status.value,
                )
            )
        except Exception as error:
            raise RebalancePersistenceError(
                "The rebalance workflow completed, but its "
                "result could not be persisted."
            ) from error

        if persisted_run.id is None:
            raise RebalancePersistenceError(
                "The persisted rebalance run did not receive "
                "a database identifier."
            )

        return PersistedRebalanceResult(
            portfolio_id=normalized_portfolio_id,
            run_id=persisted_run.run_id,
            workflow_status=(
                orchestrator_response.status.value
            ),
            workflow_name=(
                orchestrator_response.workflow_name
            ),
            workflow_message=(
                orchestrator_response.message
            ),
            trade_count=len(
                portfolio_trade_results
            ),
            database_run_id=persisted_run.id,
        )


def _extract_successful_result(
    response: OrchestratorResponse,
) -> pd.DataFrame:
    """Return a copied successful orchestrator DataFrame."""

    if not isinstance(response, OrchestratorResponse):
        raise TypeError(
            "orchestrator must return an "
            "OrchestratorResponse."
        )

    if response.status == AgentExecutionStatus.FAILED:
        raise RebalanceExecutionError(
            response.message
        )

    if response.result is None:
        raise RebalanceExecutionError(
            "The rebalance workflow completed without "
            "returning trade results."
        )

    if not isinstance(response.result, pd.DataFrame):
        raise RebalanceExecutionError(
            "The rebalance workflow result must be "
            "a pandas DataFrame."
        )

    if response.result.empty:
        raise RebalanceExecutionError(
            "The rebalance workflow returned no trade rows."
        )

    return response.result.copy(deep=True)


def _extract_portfolio_rows(
    *,
    trade_results: pd.DataFrame,
    portfolio_id: str,
) -> pd.DataFrame:
    """Return trade rows belonging to the requested portfolio."""

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
        available_portfolio_ids = {
            str(value)
            for value in trade_results[
                "portfolio_id"
            ].dropna()
        }

        if len(available_portfolio_ids) == 1:
            generated_portfolio_id = next(
                iter(available_portfolio_ids)
            )

            portfolio_rows = trade_results.loc[
                trade_results["portfolio_id"].astype(str)
                == generated_portfolio_id
            ].copy(deep=True)

            portfolio_rows.loc[
                :,
                "portfolio_id",
            ] = portfolio_id
        else:
            raise RebalanceExecutionError(
                "The rebalance workflow did not return "
                "a unique portfolio result."
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