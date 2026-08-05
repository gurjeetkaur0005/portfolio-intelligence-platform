from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from src.api.schemas import ErrorResponse
from src.database.repositories import RecordNotFoundError
from src.llm.gemini_language_model import GeminiLanguageModelError
from src.utils.logger import get_logger


logger = get_logger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Register safe global exception handlers for the API."""

    app.add_exception_handler(
        RecordNotFoundError,
        _record_not_found_handler,
    )
    app.add_exception_handler(
        ValueError,
        _value_error_handler,
    )
    app.add_exception_handler(
        SQLAlchemyError,
        _sqlalchemy_error_handler,
    )
    app.add_exception_handler(
        GeminiLanguageModelError,
        _gemini_error_handler,
    )
    app.add_exception_handler(
        ValidationError,
        _pydantic_validation_error_handler,
    )
    app.add_exception_handler(
        RequestValidationError,
        _request_validation_error_handler,
    )
    app.add_exception_handler(
        Exception,
        _generic_exception_handler,
    )


async def _record_not_found_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Return a safe 404 response for missing database records."""

    logger.warning(
        "record_not_found path=%s error=%s",
        request.url.path,
        exc,
    )
    return _error_response(
        code="record_not_found",
        message=str(exc),
        status_code=status.HTTP_404_NOT_FOUND,
    )


async def _value_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Return a safe 400 response for invalid domain input."""

    logger.warning(
        "value_error path=%s error=%s",
        request.url.path,
        exc,
    )
    return _error_response(
        code="invalid_request",
        message="Invalid request.",
        status_code=status.HTTP_400_BAD_REQUEST,
    )


async def _sqlalchemy_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Return a safe 500 response for database failures."""

    logger.error(
        "database_error path=%s",
        request.url.path,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return _error_response(
        code="database_error",
        message="A database error occurred.",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


async def _gemini_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Return a safe 502 response for language-model failures."""

    logger.error(
        "llm_error path=%s",
        request.url.path,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return _error_response(
        code="llm_error",
        message="The language model provider is unavailable.",
        status_code=status.HTTP_502_BAD_GATEWAY,
    )


async def _pydantic_validation_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Return a safe 422 response for Pydantic validation errors."""

    logger.warning(
        "pydantic_validation_error path=%s error=%s",
        request.url.path,
        exc,
    )
    return _validation_error_response()


async def _request_validation_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Return a safe 422 response for FastAPI request validation errors."""

    logger.warning(
        "request_validation_error path=%s error=%s",
        request.url.path,
        exc,
    )
    return _validation_error_response()


async def _generic_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Return a safe 500 response for unexpected failures."""

    logger.error(
        "unexpected_error path=%s",
        request.url.path,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return _error_response(
        code="internal_error",
        message="An internal server error occurred.",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


def _validation_error_response() -> JSONResponse:
    """Build the standard validation error response."""

    return _error_response(
        code="validation_error",
        message="Request validation failed.",
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
    )


def _error_response(
    *,
    code: str,
    message: str,
    status_code: int,
) -> JSONResponse:
    """Build the standard API error response."""

    response = ErrorResponse(
        code=code,
        message=message,
        status=status_code,
    )
    return JSONResponse(
        status_code=status_code,
        content=response.model_dump(),
    )
