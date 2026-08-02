from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Represent the API health-check response."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    status: str
    service: str


class RebalanceRequest(BaseModel):
    """
    Represent an API request to run portfolio rebalancing.

    Validation happens at the API boundary before the request reaches
    the Orchestrator Agent.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    number_of_clients: int = Field(
        default=1,
        gt=0,
        description=(
            "Number of synthetic client portfolios to process."
        ),
    )

    evaluation_date: date | None = Field(
        default=None,
        description=(
            "Date used for calendar and event-trigger evaluation."
        ),
    )

    portfolio_value: float = Field(
        default=1_000_000.0,
        gt=0.0,
        description=(
            "Starting monetary value assigned to each portfolio."
        ),
    )

    transaction_cost_rate: float = Field(
        default=0.002,
        ge=0.0,
        le=1.0,
        description=(
            "Transaction-cost rate expressed as a decimal."
        ),
    )


class TradeRecordResponse(BaseModel):
    """
    Represent one JSON-safe row returned by the rebalance pipeline.

    The pipeline currently returns a pandas DataFrame. The API layer
    converts every DataFrame row into this JSON-compatible structure.
    """

    model_config = ConfigDict(
        extra="allow",
        frozen=True,
    )

    portfolio_id: Any
    asset: str
    action: str


class RebalanceResponse(BaseModel):
    """Represent a successful rebalance API response."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    status: str
    workflow_name: str
    message: str
    records: list[dict[str, Any]]
    record_count: int


class ErrorResponse(BaseModel):
    """Represent a safe API error response."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    error: str
    detail: str