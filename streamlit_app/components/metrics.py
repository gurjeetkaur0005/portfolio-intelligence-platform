from __future__ import annotations

from typing import Any

import streamlit as st

from streamlit_app.services.display import (
    display_currency,
    display_number,
    display_percentage,
)


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

    return display_currency(
        value,
        currency_symbol=currency_symbol,
    )


def format_percentage(
    value: Any,
) -> str:
    """Format a decimal backend value as a percentage."""

    return display_percentage(value)


def format_number(
    value: Any,
) -> str:
    """Format a numeric backend value."""

    return display_number(value)
