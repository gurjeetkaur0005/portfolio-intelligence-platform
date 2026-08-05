from __future__ import annotations

from time import perf_counter
from typing import Awaitable, Callable

from fastapi import FastAPI, Request, Response

from src.api.exceptions import register_exception_handlers
from src.api.routes import router
from src.api.schemas import HealthResponse
from src.utils.logger import get_logger


logger = get_logger(__name__)

app = FastAPI(
    title="Portfolio Intelligence Platform",
    version="1.0.0",
    description=(
        "API for deterministic portfolio rebalancing, "
        "portfolio analysis, and AI-assisted explanations."
    ),
)

register_exception_handlers(app)
app.include_router(router)


@app.middleware("http")
async def log_requests(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Log FastAPI request start and completion."""

    start_time = perf_counter()
    logger.info(
        "request_start method=%s path=%s",
        request.method,
        request.url.path,
    )

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (perf_counter() - start_time) * 1000
        logger.exception(
            "request_failed method=%s path=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            duration_ms,
        )
        raise

    duration_ms = (perf_counter() - start_time) * 1000
    logger.info(
        "request_complete method=%s path=%s status_code=%s "
        "duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )

    return response


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
