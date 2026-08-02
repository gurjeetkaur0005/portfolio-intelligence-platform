from __future__ import annotations

from collections.abc import Iterator

import pytest

from src.api.main import app


_ORIGINAL_ROUTES = list(app.router.routes)
_ORIGINAL_OPENAPI_SCHEMA = app.openapi_schema


@pytest.fixture(autouse=True)
def restore_fastapi_app_state() -> Iterator[None]:
    """Keep tests from leaking mutations to the shared FastAPI app."""

    app.router.routes[:] = list(_ORIGINAL_ROUTES)
    app.openapi_schema = _ORIGINAL_OPENAPI_SCHEMA
    app.dependency_overrides.clear()

    yield

    app.router.routes[:] = list(_ORIGINAL_ROUTES)
    app.openapi_schema = _ORIGINAL_OPENAPI_SCHEMA
    app.dependency_overrides.clear()
