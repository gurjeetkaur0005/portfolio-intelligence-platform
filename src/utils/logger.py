from __future__ import annotations

import logging
import os
import sys


LOG_LEVEL_ENV = "LOG_LEVEL"
DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT = (
    "%(asctime)s %(levelname)s %(name)s %(message)s"
)


def configure_logging() -> None:
    """Configure application logging once."""

    logging.basicConfig(
        level=_log_level(),
        format=LOG_FORMAT,
        stream=sys.stdout,
        force=False,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for one module."""

    configure_logging()

    return logging.getLogger(name)


def _log_level() -> int:
    """Return the configured log level."""

    level_name = os.getenv(
        LOG_LEVEL_ENV,
        DEFAULT_LOG_LEVEL,
    ).upper()

    configured_level = getattr(
        logging,
        level_name,
        logging.INFO,
    )

    if not isinstance(configured_level, int):
        return logging.INFO

    return configured_level


logger = get_logger("portfolio_intelligence")
