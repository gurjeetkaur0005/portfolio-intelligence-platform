from __future__ import annotations

from decimal import Decimal

from fastapi.testclient import TestClient

from src.api.dependencies import (
    get_rebalance_application_service,
)
from src.api.main import app
from src.api.schemas import DatabaseRebalanceResponse
from src.database.repositories import RecordNotFoundError
from src.services.rebalance_application_service import (
    PersistedRebalanceResult,
    RebalanceExecutionError,
    RebalancePersistenceError,
)


client = TestClient(app)


class FakeRebalanceApplicationService:
    """Fake database-backed rebalance application service."""

    def __init__(
        self,
        *,
        error: Exception | None = None,
    ) -> None:
        self.error = error
        self.called = False
        self.portfolio_id: str | None = None
        self.portfolio_value: Decimal | None = None
        self.transaction_cost_rate: Decimal | None = None

    def execute_rebalance(
        self,
        *,
        portfolio_id: str,
        portfolio_value: Decimal,
        transaction_cost_rate: Decimal,
        run_id: str | None = None,
    ) -> PersistedRebalanceResult:
        self.called = True
        self.portfolio_id = portfolio_id
        self.portfolio_value = portfolio_value
        self.transaction_cost_rate = transaction_cost_rate

        if self.error is not None:
            raise self.error

        return PersistedRebalanceResult(
            portfolio_id=portfolio_id,
            run_id=run_id or "rebalance-run-1",
            workflow_status="success",
            workflow_name="portfolio_rebalancing",
            workflow_message="Workflow completed.",
            trade_count=2,
            database_run_id=42,
        )


def _override_service(
    service: FakeRebalanceApplicationService,
) -> None:
    app.dependency_overrides[
        get_rebalance_application_service
    ] = lambda: service


def test_database_rebalance_endpoint_returns_success() -> None:
    service = FakeRebalanceApplicationService()
    _override_service(service)

    response = client.post(
        "/portfolios/P00001/rebalance",
        json={
            "portfolio_value": 500_000.0,
            "transaction_cost_rate": 0.001,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "portfolio_id": "P00001",
        "run_id": "rebalance-run-1",
        "trade_count": 2,
        "database_run_id": 42,
        "message": "Workflow completed.",
    }
    assert service.called is True
    assert service.portfolio_value == Decimal("500000.0")
    assert service.transaction_cost_rate == Decimal("0.001")


def test_database_rebalance_endpoint_returns_404_when_missing(
) -> None:
    service = FakeRebalanceApplicationService(
        error=RecordNotFoundError("Portfolio 'P404' was not found.")
    )
    _override_service(service)

    response = client.post(
        "/portfolios/P404/rebalance",
        json={
            "portfolio_value": 500_000.0,
            "transaction_cost_rate": 0.001,
        },
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Portfolio 'P404' was not found.",
    }


def test_database_rebalance_endpoint_rejects_invalid_request(
) -> None:
    service = FakeRebalanceApplicationService()
    _override_service(service)

    response = client.post(
        "/portfolios/P00001/rebalance",
        json={
            "portfolio_value": 500_000.0,
            "transaction_cost_rate": 0.001,
            "unexpected": "field",
        },
    )

    assert response.status_code == 422
    assert service.called is False


def test_database_rebalance_endpoint_rejects_invalid_portfolio_value(
) -> None:
    service = FakeRebalanceApplicationService()
    _override_service(service)

    response = client.post(
        "/portfolios/P00001/rebalance",
        json={
            "portfolio_value": 0,
            "transaction_cost_rate": 0.001,
        },
    )

    assert response.status_code == 422
    assert service.called is False


def test_database_rebalance_endpoint_rejects_invalid_transaction_cost(
) -> None:
    service = FakeRebalanceApplicationService()
    _override_service(service)

    response = client.post(
        "/portfolios/P00001/rebalance",
        json={
            "portfolio_value": 500_000.0,
            "transaction_cost_rate": 1.01,
        },
    )

    assert response.status_code == 422
    assert service.called is False


def test_database_rebalance_endpoint_handles_execution_failure(
) -> None:
    service = FakeRebalanceApplicationService(
        error=RebalanceExecutionError("Internal workflow failure.")
    )
    _override_service(service)

    response = client.post(
        "/portfolios/P00001/rebalance",
        json={
            "portfolio_value": 500_000.0,
            "transaction_cost_rate": 0.001,
        },
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": (
            "The rebalance workflow could not be completed."
        ),
    }


def test_database_rebalance_endpoint_handles_persistence_failure(
) -> None:
    service = FakeRebalanceApplicationService(
        error=RebalancePersistenceError("Persistence failed.")
    )
    _override_service(service)

    response = client.post(
        "/portfolios/P00001/rebalance",
        json={
            "portfolio_value": 500_000.0,
            "transaction_cost_rate": 0.001,
        },
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": (
            "The rebalance workflow could not be completed."
        ),
    }


def test_database_rebalance_response_schema() -> None:
    response = DatabaseRebalanceResponse(
        status="success",
        portfolio_id="P00001",
        run_id="rebalance-run-1",
        trade_count=2,
        database_run_id=42,
        message="Workflow completed.",
    )

    assert response.model_dump() == {
        "status": "success",
        "portfolio_id": "P00001",
        "run_id": "rebalance-run-1",
        "trade_count": 2,
        "database_run_id": 42,
        "message": "Workflow completed.",
    }


def test_database_rebalance_endpoint_uses_dependency_override(
) -> None:
    service = FakeRebalanceApplicationService()
    _override_service(service)

    response = client.post(
        "/portfolios/OVERRIDE/rebalance",
        json={
            "portfolio_value": 250_000.0,
            "transaction_cost_rate": 0.002,
        },
    )

    assert response.status_code == 200
    assert service.called is True
    assert service.portfolio_id == "OVERRIDE"
