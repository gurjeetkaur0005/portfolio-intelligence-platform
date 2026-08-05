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
    PaginatedResult,
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
        self.limit: int | None = None
        self.offset: int | None = None

    def list_portfolios(
        self,
        *,
        limit: int,
        offset: int,
    ) -> PaginatedResult[PortfolioSummary]:
        self.called = True
        self.limit = limit
        self.offset = offset

        if self.empty:
            return PaginatedResult(
                items=[],
                limit=limit,
                offset=offset,
                count=0,
            )

        items = [
            PortfolioSummary(
                portfolio_id=f"P{index:05d}",
                client_id=f"C{index:05d}",
                portfolio_value=Decimal("1000000.00"),
                currency="USD",
            )
            for index in range(offset + 1, offset + limit + 1)
        ]

        return PaginatedResult(
            items=items,
            limit=limit,
            offset=offset,
            count=len(items),
        )

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
        *,
        limit: int,
        offset: int,
    ) -> PaginatedResult[RebalanceRunSummary]:
        self.called = True
        self.limit = limit
        self.offset = offset

        if self.missing:
            raise RecordNotFoundError(
                f"Portfolio {portfolio_id!r} was not found."
            )

        if self.empty:
            return PaginatedResult(
                items=[],
                limit=limit,
                offset=offset,
                count=0,
            )

        items = [
            RebalanceRunSummary(
                run_id="RUN000001",
                status="success",
                created_at=TIMESTAMP,
                transaction_cost=Decimal("40.00"),
                portfolio_value=Decimal("1000000.00"),
            )
        ]

        return PaginatedResult(
            items=items,
            limit=limit,
            offset=offset,
            count=len(items),
        )

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
        *,
        limit: int,
        offset: int,
    ) -> PaginatedResult[RebalanceTradeDetail]:
        self.called = True
        self.limit = limit
        self.offset = offset

        if self.missing:
            raise RecordNotFoundError(
                f"Rebalance run {run_id!r} was not found."
            )

        if self.empty:
            return PaginatedResult(
                items=[],
                limit=limit,
                offset=offset,
                count=0,
            )

        items = [
            RebalanceTradeDetail(
                asset="domestic_equity",
                action="SELL",
                trade_weight=Decimal("-0.0200000000"),
                trade_value=Decimal("-20000.00"),
                estimated_tax=Decimal("400.00"),
                estimated_transaction_cost=Decimal("40.00"),
            )
        ]

        return PaginatedResult(
            items=items,
            limit=limit,
            offset=offset,
            count=len(items),
        )

    def list_rebalance_audit(
        self,
        run_id: str,
        *,
        limit: int,
        offset: int,
    ) -> PaginatedResult[RebalanceAuditEntry]:
        self.called = True
        self.limit = limit
        self.offset = offset

        if self.missing:
            raise RecordNotFoundError(
                f"Rebalance run {run_id!r} was not found."
            )

        if self.empty:
            return PaginatedResult(
                items=[],
                limit=limit,
                offset=offset,
                count=0,
            )

        items = [
            RebalanceAuditEntry(
                approval_status="NOT_REQUIRED",
                timestamp=TIMESTAMP,
                audit_message="Trade recorded.",
            )
        ]

        return PaginatedResult(
            items=items,
            limit=limit,
            offset=offset,
            count=len(items),
        )


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
    body = response.json()

    assert body["limit"] == 20
    assert body["offset"] == 0
    assert body["count"] == 20
    assert body["items"][0] == {
        "portfolio_id": "P00001",
        "client_id": "C00001",
        "portfolio_value": 1000000.0,
        "currency": "USD",
    }
    assert body["items"][1]["portfolio_id"] == "P00002"
    assert service.limit == 20
    assert service.offset == 0


def test_list_portfolios_accepts_custom_limit() -> None:
    service = FakePortfolioReadService()
    _override_service(service)

    response = client.get("/portfolios?limit=2")

    assert response.status_code == 200
    assert response.json()["limit"] == 2
    assert response.json()["offset"] == 0
    assert response.json()["count"] == 2
    assert service.limit == 2


def test_list_portfolios_accepts_custom_offset() -> None:
    service = FakePortfolioReadService()
    _override_service(service)

    response = client.get("/portfolios?limit=2&offset=5")

    assert response.status_code == 200
    assert response.json()["offset"] == 5
    assert response.json()["items"][0]["portfolio_id"] == "P00006"
    assert service.offset == 5


def test_list_portfolios_returns_empty_results() -> None:
    service = FakePortfolioReadService(empty=True)
    _override_service(service)

    response = client.get("/portfolios")

    assert response.status_code == 200
    assert response.json() == {
        "items": [],
        "limit": 20,
        "offset": 0,
        "count": 0,
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
    assert response.json()["items"] == [
        {
            "run_id": "RUN000001",
            "status": "success",
            "created_at": "2026-08-05T12:30:00Z",
            "transaction_cost": 40.0,
            "portfolio_value": 1000000.0,
        }
    ]
    assert response.json()["limit"] == 20
    assert response.json()["offset"] == 0
    assert response.json()["count"] == 1


def test_list_portfolio_rebalances_returns_empty_results() -> None:
    service = FakePortfolioReadService(empty=True)
    _override_service(service)

    response = client.get(
        "/portfolios/P00001/rebalances"
    )

    assert response.status_code == 200
    assert response.json()["items"] == []
    assert response.json()["count"] == 0


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
    assert response.json()["items"] == [
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
    assert response.json()["items"] == []
    assert response.json()["count"] == 0


def test_list_rebalance_audit_returns_success() -> None:
    service = FakePortfolioReadService()
    _override_service(service)

    response = client.get(
        "/rebalances/RUN000001/audit"
    )

    assert response.status_code == 200
    assert response.json()["items"] == [
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


@pytest.mark.parametrize(
    "path",
    [
        "/portfolios?limit=0",
        "/portfolios?limit=51",
        "/portfolios?offset=-1",
        "/portfolios/P00001/rebalances?limit=0",
        "/rebalances/RUN000001/trades?limit=51",
        "/rebalances/RUN000001/audit?offset=-1",
    ],
)
def test_paginated_read_endpoints_reject_invalid_pagination(
    path: str,
) -> None:
    service = FakePortfolioReadService()
    _override_service(service)

    response = client.get(path)

    assert response.status_code == 422


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


def test_openapi_contains_pagination_parameters() -> None:
    schema = app.openapi()

    for path, operation in [
        ("/portfolios", "get"),
        ("/portfolios/{portfolio_id}/rebalances", "get"),
        ("/rebalances/{run_id}/trades", "get"),
        ("/rebalances/{run_id}/audit", "get"),
    ]:
        parameters = schema["paths"][path][operation][
            "parameters"
        ]
        parameter_names = {
            parameter["name"]
            for parameter in parameters
        }

        assert {"limit", "offset"}.issubset(
            parameter_names
        )
