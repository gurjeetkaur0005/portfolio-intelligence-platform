from __future__ import annotations

from datetime import date

import pandas as pd
from fastapi.testclient import TestClient

from src.agents.orchestrator_agent import (
    AgentExecutionStatus,
    OrchestratorRequest,
    OrchestratorResponse,
)
from src.api.dependencies import get_orchestrator_agent
from src.api.main import app


client = TestClient(app)


class FakeSuccessfulOrchestrator:
    """Return a deterministic successful workflow response."""

    def execute_rebalance(
        self,
        request: OrchestratorRequest,
    ) -> OrchestratorResponse:
        assert request.number_of_clients == 2
        assert request.evaluation_date == date(2026, 8, 2)
        assert request.portfolio_value == 500_000.0
        assert request.transaction_cost_rate == 0.001

        result = pd.DataFrame(
            [
                {
                    "portfolio_id": "portfolio_1",
                    "asset": "domestic_equity",
                    "action": "SELL",
                    "trade_value": -10_000.0,
                    "transaction_cost": 10.0,
                },
                {
                    "portfolio_id": "portfolio_1",
                    "asset": "fixed_income",
                    "action": "BUY",
                    "trade_value": 10_000.0,
                    "transaction_cost": 10.0,
                },
            ]
        )

        return OrchestratorResponse(
            status=AgentExecutionStatus.SUCCESS,
            workflow_name="portfolio_rebalancing",
            message=(
                "The portfolio rebalancing workflow completed "
                "successfully."
            ),
            result=result,
        )


class FakeFailedOrchestrator:
    """Return a deterministic failed workflow response."""

    def execute_rebalance(
        self,
        request: OrchestratorRequest,
    ) -> OrchestratorResponse:
        return OrchestratorResponse(
            status=AgentExecutionStatus.FAILED,
            workflow_name="portfolio_rebalancing",
            message="The rebalance workflow failed.",
            result=None,
        )


def test_rebalance_endpoint_returns_serialized_records() -> None:
    app.dependency_overrides[
        get_orchestrator_agent
    ] = FakeSuccessfulOrchestrator

    try:
        response = client.post(
            "/rebalance",
            json={
                "number_of_clients": 2,
                "evaluation_date": "2026-08-02",
                "portfolio_value": 500_000.0,
                "transaction_cost_rate": 0.001,
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200

    response_body = response.json()

    assert response_body["status"] == "success"
    assert (
        response_body["workflow_name"]
        == "portfolio_rebalancing"
    )
    assert response_body["record_count"] == 2

    assert response_body["records"][0] == {
        "portfolio_id": "portfolio_1",
        "asset": "domestic_equity",
        "action": "SELL",
        "trade_value": -10_000.0,
        "transaction_cost": 10.0,
    }


def test_rebalance_endpoint_uses_request_defaults() -> None:
    captured_request: OrchestratorRequest | None = None

    class DefaultCheckingOrchestrator:
        def execute_rebalance(
            self,
            request: OrchestratorRequest,
        ) -> OrchestratorResponse:
            nonlocal captured_request
            captured_request = request

            return OrchestratorResponse(
                status=AgentExecutionStatus.SUCCESS,
                workflow_name="portfolio_rebalancing",
                message="Workflow completed.",
                result=pd.DataFrame(),
            )

    app.dependency_overrides[
        get_orchestrator_agent
    ] = DefaultCheckingOrchestrator

    try:
        response = client.post(
            "/rebalance",
            json={},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert captured_request is not None
    assert captured_request.number_of_clients == 1
    assert captured_request.evaluation_date is None
    assert captured_request.portfolio_value == 1_000_000.0
    assert captured_request.transaction_cost_rate == 0.002


def test_rebalance_endpoint_rejects_invalid_request() -> None:
    response = client.post(
        "/rebalance",
        json={
            "number_of_clients": 0,
            "portfolio_value": -100.0,
        },
    )

    assert response.status_code == 422


def test_rebalance_endpoint_returns_error_for_failed_workflow() -> None:
    app.dependency_overrides[
        get_orchestrator_agent
    ] = FakeFailedOrchestrator

    try:
        response = client.post(
            "/rebalance",
            json={},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    assert response.json() == {
        "detail": "The rebalance workflow failed.",
    }


def test_rebalance_endpoint_converts_missing_values_to_null() -> None:
    class MissingValueOrchestrator:
        def execute_rebalance(
            self,
            request: OrchestratorRequest,
        ) -> OrchestratorResponse:
            return OrchestratorResponse(
                status=AgentExecutionStatus.SUCCESS,
                workflow_name="portfolio_rebalancing",
                message="Workflow completed.",
                result=pd.DataFrame(
                    [
                        {
                            "portfolio_id": "portfolio_1",
                            "asset": "cash",
                            "action": "HOLD",
                            "optional_value": float("nan"),
                        }
                    ]
                ),
            )

    app.dependency_overrides[
        get_orchestrator_agent
    ] = MissingValueOrchestrator

    try:
        response = client.post(
            "/rebalance",
            json={},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["records"][0][
        "optional_value"
    ] is None