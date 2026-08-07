from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pandas as pd
import pytest

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
from src.services.portfolio_input_adapter import (
    DeterministicPortfolioInput,
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


class FakePortfolioInputAdapter:
    """Return predefined deterministic pipeline input."""

    def __init__(
        self,
        deterministic_input: (
            DeterministicPortfolioInput | None
        ) = None,
    ) -> None:
        self.deterministic_input = (
            deterministic_input
            if deterministic_input is not None
            else _build_deterministic_input()
        )
        self.received_portfolio: (
            PortfolioModel | None
        ) = None

    def build_input(
        self,
        portfolio: PortfolioModel,
    ) -> DeterministicPortfolioInput:
        self.received_portfolio = portfolio

        return self.deterministic_input


class FakePortfolioEngine:
    """Return predefined deterministic trade results."""

    def __init__(
        self,
        result: pd.DataFrame | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = (
            _build_trade_results()
            if result is None
            else result
        )
        self.error = error
        self.received_client_profiles: (
            pd.DataFrame | None
        ) = None
        self.received_portfolios: pd.DataFrame | None = None
        self.received_portfolio_value: float | None = None
        self.received_transaction_cost_rate: (
            float | None
        ) = None

    def __call__(
        self,
        *,
        client_profiles: pd.DataFrame,
        portfolios: pd.DataFrame,
        portfolio_value: float = 1_000_000.0,
        transaction_cost_rate: float = 0.002,
    ) -> pd.DataFrame:
        self.received_client_profiles = (
            client_profiles.copy(deep=True)
        )
        self.received_portfolios = portfolios.copy(
            deep=True
        )
        self.received_portfolio_value = portfolio_value
        self.received_transaction_cost_rate = (
            transaction_cost_rate
        )

        if self.error is not None:
            raise self.error

        return self.result


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
        self.received_started_at: datetime | None = None
        self.received_completed_at: datetime | None = None
        self.received_run_id: str | None = None
        self.received_status: str | None = None

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
        self.received_started_at = started_at
        self.received_completed_at = completed_at
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
                "portfolio_id": "P-STORED-001",
                "asset": "domestic_equity",
                "action": "SELL",
                "trade_value": -20_000.0,
            },
            {
                "portfolio_id": "P-STORED-001",
                "asset": "fixed_income",
                "action": "BUY",
                "trade_value": 20_000.0,
            },
        ]
    )


def _build_deterministic_input() -> DeterministicPortfolioInput:
    """Build deterministic pipeline input."""

    return DeterministicPortfolioInput(
        client_profiles=pd.DataFrame(
            [
                {
                    "client_id": "C00001",
                    "portfolio_id": "P-STORED-001",
                    "risk_category": "balanced",
                    "tax_bracket": 0.20,
                    "prior_approval_required": False,
                }
            ]
        ),
        portfolios=pd.DataFrame(
            [
                {
                    "portfolio_id": "P-STORED-001",
                    "risk_category": "balanced",
                    "drift_band": 0.05,
                    "current_cash": 1.0,
                    "target_cash": 1.0,
                }
            ]
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


def _build_service(
    *,
    portfolio_repository: FakePortfolioRepository | None = None,
    persistence_service: FakePersistenceService | None = None,
    portfolio_input_adapter: (
        FakePortfolioInputAdapter | None
    ) = None,
    portfolio_engine: FakePortfolioEngine | None = None,
) -> RebalanceApplicationService:
    """Build the application service with fakes."""

    return RebalanceApplicationService(
        portfolio_repository=(
            portfolio_repository
            if portfolio_repository is not None
            else FakePortfolioRepository(
                _build_portfolio()
            )
        ),
        persistence_service=(
            persistence_service
            if persistence_service is not None
            else FakePersistenceService(
                _build_persisted_run()
            )
        ),
        portfolio_input_adapter=(
            portfolio_input_adapter
            if portfolio_input_adapter is not None
            else FakePortfolioInputAdapter()
        ),
        portfolio_engine=(
            portfolio_engine
            if portfolio_engine is not None
            else FakePortfolioEngine()
        ),
    )


def test_execute_rebalance_returns_persisted_result() -> None:
    portfolio = _build_portfolio()

    portfolio_repository = FakePortfolioRepository(
        portfolio
    )
    persistence_service = FakePersistenceService(
        _build_persisted_run()
    )

    service = _build_service(
        portfolio_repository=portfolio_repository,
        persistence_service=persistence_service,
    )

    result = service.execute_rebalance(
        portfolio_id="P-STORED-001",
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
    portfolio_input_adapter = FakePortfolioInputAdapter()

    service = _build_service(
        portfolio_repository=portfolio_repository,
        portfolio_input_adapter=portfolio_input_adapter,
    )

    service.execute_rebalance(
        portfolio_id=" P-STORED-001 ",
        transaction_cost_rate=Decimal("0.002"),
    )

    assert (
        portfolio_repository.received_portfolio_id
        == "P-STORED-001"
    )
    assert (
        portfolio_input_adapter.received_portfolio
        is portfolio_repository.portfolio
    )


def test_service_passes_values_to_portfolio_engine() -> None:
    portfolio = _build_portfolio()
    portfolio.portfolio_value = Decimal("750000.00")
    portfolio_repository = FakePortfolioRepository(
        portfolio
    )
    portfolio_engine = FakePortfolioEngine()
    persistence_service = FakePersistenceService(
        _build_persisted_run()
    )

    service = _build_service(
        portfolio_repository=portfolio_repository,
        portfolio_engine=portfolio_engine,
        persistence_service=persistence_service,
    )

    service.execute_rebalance(
        portfolio_id="P-STORED-001",
        transaction_cost_rate=Decimal("0.001"),
    )

    assert portfolio_engine.received_client_profiles is not None
    assert portfolio_engine.received_portfolios is not None
    assert (
        portfolio_engine.received_portfolio_value
        == 750000.0
    )
    assert (
        portfolio_engine.received_transaction_cost_rate
        == 0.001
    )
    assert (
        persistence_service.received_portfolio_value
        == Decimal("750000.00")
    )


def test_service_persists_successful_completion_timestamps() -> None:
    persistence_service = FakePersistenceService(
        _build_persisted_run()
    )

    service = _build_service(
        persistence_service=persistence_service,
    )

    service.execute_rebalance(
        portfolio_id="P-STORED-001",
        transaction_cost_rate=Decimal("0.002"),
    )

    assert persistence_service.received_started_at is not None
    assert persistence_service.received_completed_at is not None
    assert (
        persistence_service.received_completed_at
        >= persistence_service.received_started_at
    )


def test_service_persists_only_requested_portfolio() -> None:
    persistence_service = FakePersistenceService(
        _build_persisted_run()
    )

    service = _build_service(
        persistence_service=persistence_service,
    )

    service.execute_rebalance(
        portfolio_id="P-STORED-001",
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


def test_service_rejects_generated_portfolio_results() -> None:
    generated_trade_results = _build_trade_results()
    generated_trade_results["portfolio_id"] = "P00001"

    service = _build_service(
        portfolio_engine=FakePortfolioEngine(
            generated_trade_results
        ),
    )

    with pytest.raises(
        RebalanceExecutionError,
        match="requested portfolio",
    ):
        service.execute_rebalance(
            portfolio_id="P-STORED-001",
            transaction_cost_rate=Decimal("0.002"),
        )


def test_service_does_not_mutate_engine_result() -> None:
    trade_results = _build_trade_results()
    original = trade_results.copy(deep=True)

    service = _build_service(
        portfolio_engine=FakePortfolioEngine(
            trade_results
        ),
    )

    service.execute_rebalance(
        portfolio_id="P-STORED-001",
        transaction_cost_rate=Decimal("0.002"),
    )

    pd.testing.assert_frame_equal(
        trade_results,
        original,
    )


def test_missing_portfolio_error_is_preserved() -> None:
    service = _build_service(
        portfolio_repository=FakePortfolioRepository(
            None
        ),
    )

    with pytest.raises(
        RecordNotFoundError,
        match="was not found",
    ):
        service.execute_rebalance(
            portfolio_id="UNKNOWN",
            transaction_cost_rate=Decimal("0.002"),
        )


def test_failed_portfolio_engine_raises_execution_error() -> None:
    service = _build_service(
        portfolio_engine=FakePortfolioEngine(
            error=RuntimeError("Optimizer failed.")
        ),
    )

    with pytest.raises(
        RebalanceExecutionError,
        match="could not be completed",
    ):
        service.execute_rebalance(
            portfolio_id="P-STORED-001",
            transaction_cost_rate=Decimal("0.002"),
        )


def test_empty_result_raises_execution_error() -> None:
    service = _build_service(
        portfolio_engine=FakePortfolioEngine(
            pd.DataFrame()
        ),
    )

    with pytest.raises(
        RebalanceExecutionError,
        match="no trade rows",
    ):
        service.execute_rebalance(
            portfolio_id="P-STORED-001",
            transaction_cost_rate=Decimal("0.002"),
        )


def test_persistence_failure_is_wrapped() -> None:
    service = _build_service(
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
    portfolio = _build_portfolio()
    portfolio.portfolio_value = portfolio_value
    service = _build_service(
        portfolio_repository=FakePortfolioRepository(
            portfolio
        )
    )

    with pytest.raises(ValueError):
        service.execute_rebalance(
            portfolio_id="P-STORED-001",
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
    service = _build_service()

    with pytest.raises(ValueError):
        service.execute_rebalance(
            portfolio_id="P-STORED-001",
            transaction_cost_rate=(
                transaction_cost_rate
            ),
        )


def test_empty_portfolio_id_is_rejected() -> None:
    service = _build_service()

    with pytest.raises(
        ValueError,
        match="portfolio_id must not be empty",
    ):
        service.execute_rebalance(
            portfolio_id="   ",
            transaction_cost_rate=Decimal("0.002"),
        )
