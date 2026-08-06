from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

import streamlit as st


def render_metric_card(
    *,
    label: str,
    value: str,
    help_text: str | None = None,
) -> None:
    """Render one reusable Streamlit metric card."""

    st.metric(
        label=label,
        value=value,
        help=help_text,
    )


def format_currency(
    value: Any,
    *,
    currency_symbol: str = "$",
) -> str:
    """Format a numeric backend value as currency."""

    if value is None:
        return "Not available"

    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return "Not available"

    return f"{currency_symbol}{decimal_value:,.2f}"