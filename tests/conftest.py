from __future__ import annotations

from collections.abc import Iterator

import pytest

from src.api.main import app


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> Iterator[None]:
    """Prevent dependency overrides from leaking between tests."""

    app.dependency_overrides.clear()

    try:
        yield
    finally:
        app.dependency_overrides.clear()
