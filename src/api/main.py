from __future__ import annotations

from fastapi import FastAPI

from src.api.schemas import HealthResponse


app = FastAPI(
    title="Portfolio Intelligence Platform",
    version="1.0.0",
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
def health() -> HealthResponse:
    """Return the API health status."""

    return HealthResponse(
        status="healthy",
        service="portfolio-intelligence-platform",
    )