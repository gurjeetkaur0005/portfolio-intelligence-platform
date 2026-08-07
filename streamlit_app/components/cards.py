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


def render_key_value(
    *,
    label: str,
    value: str,
) -> None:
    """Render one compact key-value metadata row."""

    import streamlit as st

    st.markdown(
        (
            "<div class='pm-key-value'>"
            f"<span>{label}</span>"
            f"<strong>{value}</strong>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
