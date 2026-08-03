from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

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
from src.services.rebalance_application_service import (
    PersistedRebalanceResult,
    RebalanceApplicationService,
    RebalanceExecutionError,
    RebalancePersistenceError,
)


class FakePortfolioRepository:
    """Return a predefined portfolio."""

    def __init__(
        self,
        portfolio: PortfolioModel | None,
    ) -> None:
        self.portfolio = portfolio
        self.received_portfolio_id: str | None = None

    def require_portfolio_by_business_id(
        self,
        portfolio_id: str,
    ) -> PortfolioModel:
        self.received_portfolio_id = portfolio_id

        if self.portfolio is None:
            raise RecordNotFoundError(
                f"Portfolio {portfolio_id!r} was not found."
            )

        return self.portfolio


class FakeOrchestrator:
    """Return a predefined orchestrator response."""

    def __init__(
        self,
        response: OrchestratorResponse,
    ) -> None:
        self.response = response
        self.received_request: (
            OrchestratorRequest | None
        ) = None

    def execute_rebalance(
        self,
        request: OrchestratorRequest,
    ) -> OrchestratorResponse:
        self.received_request = request

        return self.response


class FakePersistenceService:
    """Record persistence arguments and return a run."""

    def __init__(
        self,
        persisted_run: RebalanceRunModel,
        error: Exception | None = None,
    ) -> None:
        self.persisted_run = persisted_run
        self.error = error

        self.received_portfolio: (
            PortfolioModel | None
        ) = None
        self.received_trade_results: (
            pd.DataFrame | None
        ) = None
        self.received_portfolio_value: (
            Decimal | None
        ) = None
        self.received_transaction_cost_rate: (
            Decimal | None
        ) = None
        self.received_run_id: str | None = None
        self.received_status: str | None = None

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
        if self.error is not None:
            raise self.error

        self.received_portfolio = portfolio
        self.received_trade_results = (
            trade_results.copy(deep=True)
        )
        self.received_portfolio_value = (
            portfolio_value
        )
        self.received_transaction_cost_rate = (
            transaction_cost_rate
        )
        self.received_run_id = run_id
        self.received_status = status

        return self.persisted_run


def _build_portfolio() -> PortfolioModel:
    """Build one persisted portfolio-like model."""

    portfolio = PortfolioModel(
        portfolio_id="P-STORED-001",
        portfolio_value=Decimal("1000000.00"),
        currency="USD",
    )

    portfolio.id = 10

    return portfolio


def _build_trade_results() -> pd.DataFrame:
    """Build deterministic trade results."""

    return pd.DataFrame(
        [
            {
                "portfolio_id": "P00001",
                "asset": "domestic_equity",
                "action": "SELL",
                "trade_value": -20_000.0,
            },
            {
                "portfolio_id": "P00001",
                "asset": "fixed_income",
                "action": "BUY",
                "trade_value": 20_000.0,
            },
        ]
    )


def _build_success_response(
    result: pd.DataFrame | None = None,
) -> OrchestratorResponse:
    """Build one successful orchestrator response."""

    return OrchestratorResponse(
        status=AgentExecutionStatus.SUCCESS,
        workflow_name="portfolio_rebalancing",
        message="Workflow completed successfully.",
        result=(
            _build_trade_results()
            if result is None
            else result
        ),
    )


def _build_persisted_run() -> RebalanceRunModel:
    """Build one persisted rebalance run."""

    run = RebalanceRunModel(
        run_id="RUN000001",
        status="success",
        portfolio_value=Decimal("1000000.00"),
        transaction_cost_rate=Decimal(
            "0.0020000000"
        ),
    )

    run.id = 25

    return run


def test_execute_rebalance_returns_persisted_result() -> None:
    portfolio = _build_portfolio()

    portfolio_repository = FakePortfolioRepository(
        portfolio
    )
    orchestrator = FakeOrchestrator(
        _build_success_response()
    )
    persistence_service = FakePersistenceService(
        _build_persisted_run()
    )

    service = RebalanceApplicationService(
        portfolio_repository=portfolio_repository,
        orchestrator=orchestrator,
        persistence_service=persistence_service,
    )

    result = service.execute_rebalance(
        portfolio_id="P-STORED-001",
        portfolio_value=Decimal("1000000.00"),
        transaction_cost_rate=Decimal("0.002"),
        run_id="RUN000001",
    )

    assert isinstance(
        result,
        PersistedRebalanceResult,
    )
    assert result.portfolio_id == "P-STORED-001"
    assert result.run_id == "RUN000001"
    assert result.workflow_status == "success"
    assert result.trade_count == 2
    assert result.database_run_id == 25


def test_service_loads_requested_portfolio() -> None:
    portfolio_repository = FakePortfolioRepository(
        _build_portfolio()
    )

    service = RebalanceApplicationService(
        portfolio_repository=portfolio_repository,
        orchestrator=FakeOrchestrator(
            _build_success_response()
        ),
        persistence_service=FakePersistenceService(
            _build_persisted_run()
        ),
    )

    service.execute_rebalance(
        portfolio_id=" P-STORED-001 ",
        portfolio_value=Decimal("1000000.00"),
        transaction_cost_rate=Decimal("0.002"),
    )

    assert (
        portfolio_repository.received_portfolio_id
        == "P-STORED-001"
    )


def test_service_passes_values_to_orchestrator() -> None:
    orchestrator = FakeOrchestrator(
        _build_success_response()
    )

    service = RebalanceApplicationService(
        portfolio_repository=FakePortfolioRepository(
            _build_portfolio()
        ),
        orchestrator=orchestrator,
        persistence_service=FakePersistenceService(
            _build_persisted_run()
        ),
    )

    service.execute_rebalance(
        portfolio_id="P-STORED-001",
        portfolio_value=Decimal("750000.00"),
        transaction_cost_rate=Decimal("0.001"),
    )

    assert orchestrator.received_request is not None
    assert (
        orchestrator.received_request.number_of_clients
        == 1
    )
    assert (
        orchestrator.received_request.portfolio_value
        == 750_000.0
    )
    assert (
        orchestrator.received_request
        .transaction_cost_rate
        == 0.001
    )


def test_service_persists_only_requested_portfolio() -> None:
    persistence_service = FakePersistenceService(
        _build_persisted_run()
    )

    service = RebalanceApplicationService(
        portfolio_repository=FakePortfolioRepository(
            _build_portfolio()
        ),
        orchestrator=FakeOrchestrator(
            _build_success_response()
        ),
        persistence_service=persistence_service,
    )

    service.execute_rebalance(
        portfolio_id="P-STORED-001",
        portfolio_value=Decimal("1000000.00"),
        transaction_cost_rate=Decimal("0.002"),
    )

    assert (
        persistence_service.received_trade_results
        is not None
    )

    assert set(
        persistence_service
        .received_trade_results["portfolio_id"]
    ) == {"P-STORED-001"}


def test_service_does_not_mutate_orchestrator_result() -> None:
    trade_results = _build_trade_results()
    original = trade_results.copy(deep=True)

    service = RebalanceApplicationService(
        portfolio_repository=FakePortfolioRepository(
            _build_portfolio()
        ),
        orchestrator=FakeOrchestrator(
            _build_success_response(
                trade_results
            )
        ),
        persistence_service=FakePersistenceService(
            _build_persisted_run()
        ),
    )

    service.execute_rebalance(
        portfolio_id="P-STORED-001",
        portfolio_value=Decimal("1000000.00"),
        transaction_cost_rate=Decimal("0.002"),
    )

    pd.testing.assert_frame_equal(
        trade_results,
        original,
    )


def test_missing_portfolio_error_is_preserved() -> None:
    service = RebalanceApplicationService(
        portfolio_repository=FakePortfolioRepository(
            None
        ),
        orchestrator=FakeOrchestrator(
            _build_success_response()
        ),
        persistence_service=FakePersistenceService(
            _build_persisted_run()
        ),
    )

    with pytest.raises(
        RecordNotFoundError,
        match="was not found",
    ):
        service.execute_rebalance(
            portfolio_id="UNKNOWN",
            portfolio_value=Decimal("1000000.00"),
            transaction_cost_rate=Decimal("0.002"),
        )


def test_failed_orchestrator_raises_execution_error() -> None:
    failed_response = OrchestratorResponse(
        status=AgentExecutionStatus.FAILED,
        workflow_name="portfolio_rebalancing",
        message="Optimizer failed.",
        result=None,
    )

    service = RebalanceApplicationService(
        portfolio_repository=FakePortfolioRepository(
            _build_portfolio()
        ),
        orchestrator=FakeOrchestrator(
            failed_response
        ),
        persistence_service=FakePersistenceService(
            _build_persisted_run()
        ),
    )

    with pytest.raises(
        RebalanceExecutionError,
        match="Optimizer failed",
    ):
        service.execute_rebalance(
            portfolio_id="P-STORED-001",
            portfolio_value=Decimal("1000000.00"),
            transaction_cost_rate=Decimal("0.002"),
        )


def test_empty_result_raises_execution_error() -> None:
    service = RebalanceApplicationService(
        portfolio_repository=FakePortfolioRepository(
            _build_portfolio()
        ),
        orchestrator=FakeOrchestrator(
            _build_success_response(
                pd.DataFrame()
            )
        ),
        persistence_service=FakePersistenceService(
            _build_persisted_run()
        ),
    )

    with pytest.raises(
        RebalanceExecutionError,
        match="no trade rows",
    ):
        service.execute_rebalance(
            portfolio_id="P-STORED-001",
            portfolio_value=Decimal("1000000.00"),
            transaction_cost_rate=Decimal("0.002"),
        )


def test_persistence_failure_is_wrapped() -> None:
    service = RebalanceApplicationService(
        portfolio_repository=FakePortfolioRepository(
            _build_portfolio()
        ),
        orchestrator=FakeOrchestrator(
            _build_success_response()
        ),
        persistence_service=FakePersistenceService(
            _build_persisted_run(),
            error=RuntimeError("Database failed."),
        ),
    )

    with pytest.raises(
        RebalancePersistenceError,
        match="could not be persisted",
    ):
        service.execute_rebalance(
            portfolio_id="P-STORED-001",
            portfolio_value=Decimal("1000000.00"),
            transaction_cost_rate=Decimal("0.002"),
        )


@pytest.mark.parametrize(
    "portfolio_value",
    [
        Decimal("0"),
        Decimal("-1"),
        Decimal("NaN"),
        Decimal("Infinity"),
    ],
)
def test_invalid_portfolio_value_is_rejected(
    portfolio_value: Decimal,
) -> None:
    service = RebalanceApplicationService(
        portfolio_repository=FakePortfolioRepository(
            _build_portfolio()
        ),
        orchestrator=FakeOrchestrator(
            _build_success_response()
        ),
        persistence_service=FakePersistenceService(
            _build_persisted_run()
        ),
    )

    with pytest.raises(ValueError):
        service.execute_rebalance(
            portfolio_id="P-STORED-001",
            portfolio_value=portfolio_value,
            transaction_cost_rate=Decimal("0.002"),
        )


@pytest.mark.parametrize(
    "transaction_cost_rate",
    [
        Decimal("-0.01"),
        Decimal("1.01"),
        Decimal("NaN"),
        Decimal("Infinity"),
    ],
)
def test_invalid_transaction_cost_rate_is_rejected(
    transaction_cost_rate: Decimal,
) -> None:
    service = RebalanceApplicationService(
        portfolio_repository=FakePortfolioRepository(
            _build_portfolio()
        ),
        orchestrator=FakeOrchestrator(
            _build_success_response()
        ),
        persistence_service=FakePersistenceService(
            _build_persisted_run()
        ),
    )

    with pytest.raises(ValueError):
        service.execute_rebalance(
            portfolio_id="P-STORED-001",
            portfolio_value=Decimal("1000000.00"),
            transaction_cost_rate=(
                transaction_cost_rate
            ),
        )


def test_empty_portfolio_id_is_rejected() -> None:
    service = RebalanceApplicationService(
        portfolio_repository=FakePortfolioRepository(
            _build_portfolio()
        ),
        orchestrator=FakeOrchestrator(
            _build_success_response()
        ),
        persistence_service=FakePersistenceService(
            _build_persisted_run()
        ),
    )

    with pytest.raises(
        ValueError,
        match="portfolio_id must not be empty",
    ):
        service.execute_rebalance(
            portfolio_id="   ",
            portfolio_value=Decimal("1000000.00"),
            transaction_cost_rate=Decimal("0.002"),
        )