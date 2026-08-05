from __future__ import annotations

from collections.abc import Awaitable, Callable
from time import perf_counter

from fastapi import Depends, FastAPI, Request, Response, status
from sqlalchemy import text

from src.api.dependencies import (
    get_database_url_reader,
    get_readiness_session_factory_reader,
)
from src.api.exceptions import register_exception_handlers
from src.api.routes import router
from src.api.schemas import HealthResponse, ReadinessResponse
from src.database.session import DatabaseSessionFactory
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


@app.get(
    "/ready",
    response_model=ReadinessResponse,
    tags=["System"],
)
def ready(
    response: Response,
    database_url_reader: Callable[[], str] = Depends(
        get_database_url_reader
    ),
    session_factory_reader: Callable[
        [],
        DatabaseSessionFactory,
    ] = Depends(get_readiness_session_factory_reader),
) -> ReadinessResponse:
    """Return whether the API is ready to serve database-backed traffic."""

    configuration_status = "valid"

    try:
        database_url_reader()
    except (TypeError, ValueError):
        logger.warning("readiness_failed configuration=invalid")
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(
            status="not_ready",
            database="unavailable",
            configuration="invalid",
        )

    try:
        session_factory = session_factory_reader()
        with session_factory() as session:
            session.execute(text("SELECT 1")).scalar_one()
    except Exception:
        logger.exception(
            "readiness_failed database=unavailable "
            "configuration=%s",
            configuration_status,
        )
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(
            status="not_ready",
            database="unavailable",
            configuration=configuration_status,
        )

    logger.info("readiness_check status=ready")
    return ReadinessResponse(
        status="ready",
        database="connected",
        configuration=configuration_status,
    )
