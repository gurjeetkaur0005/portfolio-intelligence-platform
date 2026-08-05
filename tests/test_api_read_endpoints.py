from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

from src.api.dependencies import (
    get_portfolio_read_application_service,
)
from src.api.main import app
from src.api.schemas import PortfolioSummaryResponse
from src.database.repositories import RecordNotFoundError
from src.services.portfolio_read_application_service import (
    PortfolioDetail,
    PortfolioHolding,
    PortfolioSummary,
    RebalanceAuditEntry,
    RebalanceRunDetail,
    RebalanceRunSummary,
    RebalanceTradeDetail,
)


client = TestClient(app)

TIMESTAMP = datetime(
    2026,
    8,
    5,
    12,
    30,
    tzinfo=timezone.utc,
)


class FakePortfolioReadService:
    """Fake read service for API dependency overrides."""

    def __init__(
        self,
        *,
        missing: bool = False,
        empty: bool = False,
    ) -> None:
        self.missing = missing
        self.empty = empty
        self.called = False

    def list_portfolios(self) -> list[PortfolioSummary]:
        self.called = True

        if self.empty:
            return []

        return [
            PortfolioSummary(
                portfolio_id="P00001",
                client_id="C00001",
                portfolio_value=Decimal("1000000.00"),
                currency="USD",
            )
        ]

    def get_portfolio(
        self,
        portfolio_id: str,
    ) -> PortfolioDetail:
        self.called = True

        if self.missing:
            raise RecordNotFoundError(
                f"Portfolio {portfolio_id!r} was not found."
            )

        return PortfolioDetail(
            portfolio_id=portfolio_id,
            client_id="C00001",
            portfolio_value=Decimal("1000000.00"),
            currency="USD",
            holdings=(
                PortfolioHolding(
                    asset="cash",
                    current_weight=Decimal("0.0500000000"),
                    current_value=Decimal("50000.00"),
                    cost_basis=Decimal("50000.00"),
                ),
            ),
        )

    def list_portfolio_rebalances(
        self,
        portfolio_id: str,
    ) -> list[RebalanceRunSummary]:
        self.called = True

        if self.missing:
            raise RecordNotFoundError(
                f"Portfolio {portfolio_id!r} was not found."
            )

        if self.empty:
            return []

        return [
            RebalanceRunSummary(
                run_id="RUN000001",
                status="success",
                created_at=TIMESTAMP,
                transaction_cost=Decimal("40.00"),
                portfolio_value=Decimal("1000000.00"),
            )
        ]

    def get_rebalance(
        self,
        run_id: str,
    ) -> RebalanceRunDetail:
        self.called = True

        if self.missing:
            raise RecordNotFoundError(
                f"Rebalance run {run_id!r} was not found."
            )

        return RebalanceRunDetail(
            run_id=run_id,
            portfolio_id="P00001",
            status="success",
            created_at=TIMESTAMP,
            completed_at=TIMESTAMP,
            portfolio_value=Decimal("1000000.00"),
            transaction_cost_rate=Decimal("0.0020000000"),
            trade_count=1,
            transaction_cost=Decimal("40.00"),
            estimated_tax_liability=Decimal("400.00"),
        )

    def list_rebalance_trades(
        self,
        run_id: str,
    ) -> list[RebalanceTradeDetail]:
        self.called = True

        if self.missing:
            raise RecordNotFoundError(
                f"Rebalance run {run_id!r} was not found."
            )

        if self.empty:
            return []

        return [
            RebalanceTradeDetail(
                asset="domestic_equity",
                action="SELL",
                trade_weight=Decimal("-0.0200000000"),
                trade_value=Decimal("-20000.00"),
                estimated_tax=Decimal("400.00"),
                estimated_transaction_cost=Decimal("40.00"),
            )
        ]

    def list_rebalance_audit(
        self,
        run_id: str,
    ) -> list[RebalanceAuditEntry]:
        self.called = True

        if self.missing:
            raise RecordNotFoundError(
                f"Rebalance run {run_id!r} was not found."
            )

        if self.empty:
            return []

        return [
            RebalanceAuditEntry(
                approval_status="NOT_REQUIRED",
                timestamp=TIMESTAMP,
                audit_message="Trade recorded.",
            )
        ]


def _override_service(
    service: FakePortfolioReadService,
) -> None:
    app.dependency_overrides[
        get_portfolio_read_application_service
    ] = lambda: service


def test_list_portfolios_returns_success() -> None:
    service = FakePortfolioReadService()
    _override_service(service)

    response = client.get("/portfolios")

    assert response.status_code == 200
    assert response.json() == {
        "portfolios": [
            {
                "portfolio_id": "P00001",
                "client_id": "C00001",
                "portfolio_value": 1000000.0,
                "currency": "USD",
            }
        ],
        "portfolio_count": 1,
    }


def test_list_portfolios_returns_empty_results() -> None:
    service = FakePortfolioReadService(empty=True)
    _override_service(service)

    response = client.get("/portfolios")

    assert response.status_code == 200
    assert response.json() == {
        "portfolios": [],
        "portfolio_count": 0,
    }


def test_get_portfolio_returns_success() -> None:
    service = FakePortfolioReadService()
    _override_service(service)

    response = client.get("/portfolios/P00001")

    assert response.status_code == 200
    assert response.json()["holdings"] == [
        {
            "asset": "cash",
            "current_weight": 0.05,
            "current_value": 50000.0,
            "cost_basis": 50000.0,
        }
    ]


def test_get_portfolio_returns_404() -> None:
    service = FakePortfolioReadService(missing=True)
    _override_service(service)

    response = client.get("/portfolios/MISSING")

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Portfolio 'MISSING' was not found.",
    }


def test_list_portfolio_rebalances_returns_success() -> None:
    service = FakePortfolioReadService()
    _override_service(service)

    response = client.get(
        "/portfolios/P00001/rebalances"
    )

    assert response.status_code == 200
    assert response.json()["rebalances"] == [
        {
            "run_id": "RUN000001",
            "status": "success",
            "created_at": "2026-08-05T12:30:00Z",
            "transaction_cost": 40.0,
            "portfolio_value": 1000000.0,
        }
    ]


def test_list_portfolio_rebalances_returns_empty_results() -> None:
    service = FakePortfolioReadService(empty=True)
    _override_service(service)

    response = client.get(
        "/portfolios/P00001/rebalances"
    )

    assert response.status_code == 200
    assert response.json()["rebalances"] == []
    assert response.json()["rebalance_count"] == 0


def test_get_rebalance_returns_success() -> None:
    service = FakePortfolioReadService()
    _override_service(service)

    response = client.get("/rebalances/RUN000001")

    assert response.status_code == 200
    assert response.json()["trade_count"] == 1
    assert response.json()["transaction_cost"] == 40.0
    assert (
        response.json()["estimated_tax_liability"]
        == 400.0
    )


def test_get_rebalance_returns_404() -> None:
    service = FakePortfolioReadService(missing=True)
    _override_service(service)

    response = client.get("/rebalances/MISSING")

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Rebalance run 'MISSING' was not found.",
    }


def test_list_rebalance_trades_returns_success() -> None:
    service = FakePortfolioReadService()
    _override_service(service)

    response = client.get(
        "/rebalances/RUN000001/trades"
    )

    assert response.status_code == 200
    assert response.json()["trades"] == [
        {
            "asset": "domestic_equity",
            "action": "SELL",
            "trade_weight": -0.02,
            "trade_value": -20000.0,
            "estimated_tax": 400.0,
            "estimated_transaction_cost": 40.0,
        }
    ]


def test_list_rebalance_trades_returns_empty_results() -> None:
    service = FakePortfolioReadService(empty=True)
    _override_service(service)

    response = client.get(
        "/rebalances/RUN000001/trades"
    )

    assert response.status_code == 200
    assert response.json()["trades"] == []
    assert response.json()["trade_count"] == 0


def test_list_rebalance_audit_returns_success() -> None:
    service = FakePortfolioReadService()
    _override_service(service)

    response = client.get(
        "/rebalances/RUN000001/audit"
    )

    assert response.status_code == 200
    assert response.json()["audit_entries"] == [
        {
            "approval_status": "NOT_REQUIRED",
            "timestamp": "2026-08-05T12:30:00Z",
            "audit_message": "Trade recorded.",
        }
    ]


def test_read_endpoint_uses_dependency_override() -> None:
    service = FakePortfolioReadService()
    _override_service(service)

    response = client.get("/portfolios")

    assert response.status_code == 200
    assert service.called is True


def test_read_response_schema_is_frozen() -> None:
    response = PortfolioSummaryResponse(
        portfolio_id="P00001",
        client_id="C00001",
        portfolio_value=1000000.0,
        currency="USD",
    )

    with pytest.raises(ValidationError):
        response.currency = "CAD"


def test_read_response_schema_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        PortfolioSummaryResponse(
            portfolio_id="P00001",
            client_id="C00001",
            portfolio_value=1000000.0,
            currency="USD",
            extra_field="invalid",  # type: ignore[call-arg]
        )


def test_openapi_contains_read_routes() -> None:
    schema = app.openapi()
    paths = schema["paths"]

    assert "/portfolios" in paths
    assert "/portfolios/{portfolio_id}" in paths
    assert "/portfolios/{portfolio_id}/rebalances" in paths
    assert "/rebalances/{run_id}" in paths
    assert "/rebalances/{run_id}/trades" in paths
    assert "/rebalances/{run_id}/audit" in paths
