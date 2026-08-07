from __future__ import annotations


def display_timestamp(
    value: object,
) -> str:
    """Return a safe timestamp display value."""

    if value is None:
        return "Not recorded"

    if isinstance(value, str) and value.strip():
        return value

    return "Unavailable"
