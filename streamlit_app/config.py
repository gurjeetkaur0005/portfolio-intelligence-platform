from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True, slots=True)
class FrontendSettings:
    """Store configuration used by the Streamlit frontend."""

    api_base_url: str
    api_timeout_seconds: float
    app_title: str


@lru_cache(maxsize=1)
def get_settings() -> FrontendSettings:
    """Load and validate frontend configuration."""

    api_base_url = os.getenv(
        "FASTAPI_BASE_URL",
        "http://localhost:8000",
    ).strip().rstrip("/")

    timeout_text = os.getenv(
        "FASTAPI_TIMEOUT_SECONDS",
        "10",
    ).strip()

    app_title = os.getenv(
        "STREAMLIT_APP_TITLE",
        "Portfolio Intelligence Platform",
    ).strip()

    if not api_base_url:
        raise ValueError("FASTAPI_BASE_URL must not be empty.")

    try:
        api_timeout_seconds = float(timeout_text)
    except ValueError as exc:
        raise ValueError(
            "FASTAPI_TIMEOUT_SECONDS must be a number."
        ) from exc

    if api_timeout_seconds <= 0:
        raise ValueError(
            "FASTAPI_TIMEOUT_SECONDS must be greater than zero."
        )

    if not app_title:
        raise ValueError("STREAMLIT_APP_TITLE must not be empty.")

    return FrontendSettings(
        api_base_url=api_base_url,
        api_timeout_seconds=api_timeout_seconds,
        app_title=app_title,
    )