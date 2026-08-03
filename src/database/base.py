from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class DatabaseBase(DeclarativeBase):
    """
    Base class for all SQLAlchemy database models.

    Every ORM model must inherit from this class so SQLAlchemy can
    discover its table definition and collect it in shared metadata.
    """