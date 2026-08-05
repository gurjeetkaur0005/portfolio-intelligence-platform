from __future__ import annotations

from fastapi import Body, FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError

from src.api.exceptions import register_exception_handlers
from src.database.repositories import RecordNotFoundError
from src.llm.gemini_language_model import GeminiLanguageModelError


class SampleRequest(BaseModel):
    """Request model used to trigger FastAPI validation."""

    value: int


def _client_for_exception(
    exception: Exception,
) -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/raise")
    def raise_exception() -> dict[str, str]:
        raise exception

    return TestClient(
        app,
        raise_server_exceptions=False,
    )


def test_record_not_found_handler_returns_safe_404() -> None:
    client = _client_for_exception(
        RecordNotFoundError("Portfolio 'P404' was not found.")
    )

    response = client.get("/raise")

    assert response.status_code == 404
    assert response.json() == {
        "code": "record_not_found",
        "message": "Portfolio 'P404' was not found.",
        "status": 404,
    }


def test_value_error_handler_returns_safe_400() -> None:
    client = _client_for_exception(
        ValueError("sensitive invalid value detail")
    )

    response = client.get("/raise")

    assert response.status_code == 400
    assert response.json() == {
        "code": "invalid_request",
        "message": "Invalid request.",
        "status": 400,
    }


def test_sqlalchemy_error_handler_returns_safe_500() -> None:
    client = _client_for_exception(
        SQLAlchemyError("database connection details")
    )

    response = client.get("/raise")

    assert response.status_code == 500
    assert response.json() == {
        "code": "database_error",
        "message": "A database error occurred.",
        "status": 500,
    }


def test_gemini_error_handler_returns_safe_502() -> None:
    client = _client_for_exception(
        GeminiLanguageModelError("provider internals")
    )

    response = client.get("/raise")

    assert response.status_code == 502
    assert response.json() == {
        "code": "llm_error",
        "message": "The language model provider is unavailable.",
        "status": 502,
    }


def test_request_validation_handler_returns_safe_422() -> None:
    app = FastAPI()
    register_exception_handlers(app)

    @app.post("/validate")
    def validate_request(
        request: SampleRequest = Body(...),
    ) -> SampleRequest:
        return request

    client = TestClient(
        app,
        raise_server_exceptions=False,
    )

    response = client.post(
        "/validate",
        json={"value": "not-an-int"},
    )

    assert response.status_code == 422
    assert response.json() == {
        "code": "validation_error",
        "message": "Request validation failed.",
        "status": 422,
    }


def test_pydantic_validation_handler_returns_safe_422() -> None:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/validate-model")
    def validate_model() -> dict[str, str]:
        SampleRequest(value="not-an-int")  # type: ignore[arg-type]
        return {"status": "unreachable"}

    client = TestClient(
        app,
        raise_server_exceptions=False,
    )

    response = client.get("/validate-model")

    assert response.status_code == 422
    assert response.json() == {
        "code": "validation_error",
        "message": "Request validation failed.",
        "status": 422,
    }


def test_generic_exception_handler_returns_safe_500() -> None:
    client = _client_for_exception(
        RuntimeError("internal implementation detail")
    )

    response = client.get("/raise")

    assert response.status_code == 500
    assert response.json() == {
        "code": "internal_error",
        "message": "An internal server error occurred.",
        "status": 500,
    }
