from __future__ import annotations

import pytest

from src.database.config import (
    DATABASE_URL_ENV,
    get_database_url,
)


def test_get_database_url_returns_environment_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        DATABASE_URL_ENV,
        (
            "postgresql+psycopg://"
            "user:password@localhost:5432/portfolio"
        ),
    )

    assert get_database_url() == (
        "postgresql+psycopg://"
        "user:password@localhost:5432/portfolio"
    )


def test_get_database_url_strips_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        DATABASE_URL_ENV,
        "  sqlite+pysqlite:///:memory:  ",
    )

    assert (
        get_database_url()
        == "sqlite+pysqlite:///:memory:"
    )


def test_missing_database_url_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        DATABASE_URL_ENV,
        raising=False,
    )

    with pytest.raises(
        ValueError,
        match="DATABASE_URL environment variable",
    ):
        get_database_url()


def test_empty_database_url_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        DATABASE_URL_ENV,
        "   ",
    )

    with pytest.raises(
        ValueError,
        match="DATABASE_URL environment variable",
    ):
        get_database_url()