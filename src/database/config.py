from __future__ import annotations

import os


DATABASE_URL_ENV = "DATABASE_URL"


def get_database_url() -> str:
    """
    Return the configured database URL.

    Raises:
        ValueError:
            If DATABASE_URL is missing or empty.
    """

    database_url = os.getenv(DATABASE_URL_ENV)

    if database_url is None or not database_url.strip():
        raise ValueError(
            "DATABASE_URL environment variable must be configured."
        )

    return database_url.strip()