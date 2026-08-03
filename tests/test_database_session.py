from __future__ import annotations

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

from src.database.session import (
    create_database_engine,
    create_database_session_factory,
)


def test_create_database_engine_returns_engine() -> None:
    engine = create_database_engine(
        "sqlite+pysqlite:///:memory:"
    )

    assert isinstance(engine, Engine)

    engine.dispose()


def test_engine_can_execute_query() -> None:
    engine = create_database_engine(
        "sqlite+pysqlite:///:memory:"
    )

    try:
        with engine.connect() as connection:
            result = connection.execute(
                text("SELECT 1")
            )

            assert result.scalar_one() == 1
    finally:
        engine.dispose()


def test_session_factory_creates_session() -> None:
    engine = create_database_engine(
        "sqlite+pysqlite:///:memory:"
    )

    try:
        session_factory = (
            create_database_session_factory(engine)
        )

        with session_factory() as session:
            assert isinstance(session, Session)
    finally:
        engine.dispose()


def test_session_can_execute_query() -> None:
    engine = create_database_engine(
        "sqlite+pysqlite:///:memory:"
    )

    try:
        session_factory = (
            create_database_session_factory(engine)
        )

        with session_factory() as session:
            result = session.execute(
                text("SELECT 1")
            )

            assert result.scalar_one() == 1
    finally:
        engine.dispose()


def test_session_factory_disables_autoflush() -> None:
    engine = create_database_engine(
        "sqlite+pysqlite:///:memory:"
    )

    try:
        session_factory = (
            create_database_session_factory(engine)
        )

        with session_factory() as session:
            assert session.autoflush is False
    finally:
        engine.dispose()


def test_session_factory_disables_expiration_on_commit() -> None:
    engine = create_database_engine(
        "sqlite+pysqlite:///:memory:"
    )

    try:
        session_factory = (
            create_database_session_factory(engine)
        )

        with session_factory() as session:
            assert session.expire_on_commit is False
    finally:
        engine.dispose()


def test_non_string_database_url_raises_type_error() -> None:
    with pytest.raises(
        TypeError,
        match="database_url must be a string",
    ):
        create_database_engine(  # type: ignore[arg-type]
            123
        )


def test_empty_database_url_raises_value_error() -> None:
    with pytest.raises(
        ValueError,
        match="database_url must not be empty",
    ):
        create_database_engine("   ")


def test_invalid_engine_raises_type_error() -> None:
    with pytest.raises(
        TypeError,
        match="engine must be a SQLAlchemy Engine",
    ):
        create_database_session_factory(  # type: ignore[arg-type]
            object()
        )