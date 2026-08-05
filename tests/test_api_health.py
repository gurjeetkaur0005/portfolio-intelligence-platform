from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.dependencies import (
    get_database_url_reader,
    get_readiness_session_factory_reader,
)
from src.api.main import app


client = TestClient(app)


class FakeScalarResult:
    """Fake SQLAlchemy result for readiness checks."""

    def scalar_one(self) -> int:
        """Return one successful SELECT 1 value."""

        return 1


class FakeSession:
    """Fake SQLAlchemy session context manager."""

    def __init__(
        self,
        *,
        execute_error: Exception | None = None,
    ) -> None:
        self.execute_error = execute_error
        self.executed_statement: object | None = None

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        return None

    def execute(
        self,
        statement: object,
    ) -> FakeScalarResult:
        """Record the statement and return a fake scalar result."""

        self.executed_statement = statement

        if self.execute_error is not None:
            raise self.execute_error

        return FakeScalarResult()


class FakeSessionFactory:
    """Fake SQLAlchemy session factory for readiness checks."""

    def __init__(
        self,
        *,
        session: FakeSession | None = None,
        open_error: Exception | None = None,
    ) -> None:
        self.session = (
            session
            if session is not None
            else FakeSession()
        )
        self.open_error = open_error
        self.called = False

    def __call__(self) -> FakeSession:
        self.called = True

        if self.open_error is not None:
            raise self.open_error

        return self.session


def _override_readiness_dependencies(
    *,
    database_url_reader,
    session_factory_reader,
) -> None:
    app.dependency_overrides[get_database_url_reader] = (
        lambda: database_url_reader
    )
    app.dependency_overrides[
        get_readiness_session_factory_reader
    ] = lambda: session_factory_reader


def test_health_endpoint_returns_success() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "portfolio-intelligence-platform",
    }


def test_ready_endpoint_returns_success() -> None:
    session = FakeSession()
    session_factory = FakeSessionFactory(session=session)
    _override_readiness_dependencies(
        database_url_reader=lambda: "postgresql://example/db",
        session_factory_reader=lambda: session_factory,
    )

    try:
        response = client.get("/ready")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "database": "connected",
        "configuration": "valid",
    }
    assert session_factory.called is True
    assert session.executed_statement is not None


def test_ready_endpoint_returns_503_for_missing_configuration() -> None:
    session_factory = FakeSessionFactory()

    def missing_database_url() -> str:
        raise ValueError("DATABASE_URL is missing.")

    _override_readiness_dependencies(
        database_url_reader=missing_database_url,
        session_factory_reader=lambda: session_factory,
    )

    try:
        response = client.get("/ready")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "database": "unavailable",
        "configuration": "invalid",
    }
    assert session_factory.called is False


def test_ready_endpoint_returns_503_for_connection_failure() -> None:
    session_factory = FakeSessionFactory(
        open_error=RuntimeError("connection failed")
    )
    _override_readiness_dependencies(
        database_url_reader=lambda: "postgresql://example/db",
        session_factory_reader=lambda: session_factory,
    )

    try:
        response = client.get("/ready")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "database": "unavailable",
        "configuration": "valid",
    }


def test_ready_endpoint_returns_503_for_select_failure() -> None:
    session_factory = FakeSessionFactory(
        session=FakeSession(
            execute_error=RuntimeError("SELECT 1 failed")
        )
    )
    _override_readiness_dependencies(
        database_url_reader=lambda: "postgresql://example/db",
        session_factory_reader=lambda: session_factory,
    )

    try:
        response = client.get("/ready")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "database": "unavailable",
        "configuration": "valid",
    }


def test_health_endpoint_is_independent_of_postgresql() -> None:
    def missing_database_url() -> str:
        raise ValueError("DATABASE_URL is missing.")

    _override_readiness_dependencies(
        database_url_reader=missing_database_url,
        session_factory_reader=lambda: FakeSessionFactory(
            open_error=RuntimeError("connection failed")
        ),
    )

    try:
        response = client.get("/health")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "portfolio-intelligence-platform",
    }
