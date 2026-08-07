from __future__ import annotations

from decimal import Decimal, InvalidOperation


def display_timestamp(
    value: object,
) -> str:
    """Return a safe timestamp display value."""

    if value is None:
        return "Not recorded"

    if isinstance(value, str) and value.strip():
        return value

    return "Unavailable"


def display_label(
    value: str,
) -> str:
    """Return a human-readable label for snake-case identifiers."""

    normalized_value = value.strip()

    if not normalized_value:
        return "Unavailable"

    return normalized_value.replace("_", " ").title()


def display_currency(
    value: object,
    *,
    currency_symbol: str = "$",
) -> str:
    """Format a numeric value as currency."""

    decimal_value = _decimal_value(value)

    if decimal_value is None:
        return "Not available"

    return f"{currency_symbol}{decimal_value:,.2f}"


def display_percentage(
    value: object,
) -> str:
    """Format a decimal value as a percentage."""

    decimal_value = _decimal_value(value)

    if decimal_value is None:
        return "Not available"

    return f"{decimal_value * 100:.2f}%"


def display_number(
    value: object,
) -> str:
    """Format a numeric value."""

    decimal_value = _decimal_value(value)

    if decimal_value is None:
        return "Not available"

    return f"{decimal_value:,.2f}"


def _decimal_value(
    value: object,
) -> Decimal | None:
    """Return a decimal for valid numeric display values."""

    if value is None or isinstance(value, bool):
        return None

    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
