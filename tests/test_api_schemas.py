from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    HealthResponse,
    RebalanceRequest,
)


def test_health_response_stores_expected_values() -> None:
    response = HealthResponse(
        status="healthy",
        service="portfolio-intelligence-platform",
    )

    assert response.status == "healthy"
    assert (
        response.service
        == "portfolio-intelligence-platform"
    )


def test_rebalance_request_uses_defaults() -> None:
    request = RebalanceRequest()

    assert request.number_of_clients == 1
    assert request.evaluation_date is None
    assert request.portfolio_value == pytest.approx(
        1_000_000.0
    )
    assert request.transaction_cost_rate == pytest.approx(
        0.002
    )


def test_rebalance_request_accepts_valid_values() -> None:
    request = RebalanceRequest(
        number_of_clients=3,
        evaluation_date=date(2026, 8, 2),
        portfolio_value=500_000.0,
        transaction_cost_rate=0.001,
    )

    assert request.number_of_clients == 3
    assert request.evaluation_date == date(2026, 8, 2)
    assert request.portfolio_value == pytest.approx(
        500_000.0
    )
    assert request.transaction_cost_rate == pytest.approx(
        0.001
    )


def test_zero_clients_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RebalanceRequest(
            number_of_clients=0,
        )


def test_negative_portfolio_value_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RebalanceRequest(
            portfolio_value=-1.0,
        )


def test_negative_transaction_cost_rate_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RebalanceRequest(
            transaction_cost_rate=-0.01,
        )


def test_transaction_cost_rate_above_one_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RebalanceRequest(
            transaction_cost_rate=1.01,
        )


def test_unknown_request_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RebalanceRequest(
            unknown_field="invalid",  # type: ignore[call-arg]
        )


def test_request_is_immutable() -> None:
    request = RebalanceRequest()

    with pytest.raises(ValidationError):
        request.portfolio_value = 10.0