from __future__ import annotations

from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.database.config import get_database_url


DatabaseSessionFactory = sessionmaker[Session]


def create_database_engine(
    database_url: str,
) -> Engine:
    """
    Create a SQLAlchemy engine for the supplied database URL.

    Args:
        database_url:
            SQLAlchemy-compatible database connection URL.

    Returns:
        Configured SQLAlchemy engine.

    Raises:
        TypeError:
            If database_url is not a string.
        ValueError:
            If database_url is empty.
    """

    normalized_url = _validate_database_url(
        database_url
    )

    return create_engine(
        normalized_url,
        pool_pre_ping=True,
    )


def create_database_session_factory(
    engine: Engine,
) -> DatabaseSessionFactory:
    """
    Create sessions bound to the supplied SQLAlchemy engine.

    Args:
        engine:
            SQLAlchemy engine used for database connectivity.

    Returns:
        Configured SQLAlchemy session factory.

    Raises:
        TypeError:
            If engine is not a SQLAlchemy Engine.
    """

    if not isinstance(engine, Engine):
        raise TypeError(
            "engine must be a SQLAlchemy Engine."
        )

    return sessionmaker(
        bind=engine,
        class_=Session,
        autoflush=False,
        expire_on_commit=False,
    )


@lru_cache(maxsize=1)
def get_database_engine() -> Engine:
    """
    Return the shared production database engine.

    The engine is created lazily and cached for application reuse.
    """

    return create_database_engine(
        get_database_url()
    )


@lru_cache(maxsize=1)
def get_database_session_factory(
) -> DatabaseSessionFactory:
    """
    Return the shared production session factory.
    """

    return create_database_session_factory(
        get_database_engine()
    )


def get_database_session() -> Generator[Session, None, None]:
    """
    Yield one short-lived database session.

    This generator is suitable for FastAPI dependency injection.
    The session is always closed after use.
    """

    session_factory = get_database_session_factory()

    with session_factory() as session:
        yield session


def _validate_database_url(
    database_url: str,
) -> str:
    """Validate and normalize a database URL."""

    if not isinstance(database_url, str):
        raise TypeError(
            "database_url must be a string."
        )

    normalized_url = database_url.strip()

    if not normalized_url:
        raise ValueError(
            "database_url must not be empty."
        )

    return normalized_url