from __future__ import annotations


def render_kpi_card(
    *,
    title: str,
    value: str,
    subtitle: str | None = None,
    status: str | None = None,
) -> None:
    """Render a reusable KPI card."""

    import streamlit as st

    st.metric(
        label=title,
        value=value,
        help=subtitle,
    )

    if status is not None:
        st.caption(status)
