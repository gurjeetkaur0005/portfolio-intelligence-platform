from __future__ import annotations

from html import escape


def render_page_header(
    *,
    title: str,
    description: str,
    context: str | None = None,
) -> None:
    """Render a consistent page header."""

    import streamlit as st

    context_html = (
        f"<div class='pm-page-context'>{context}</div>"
        if context is not None and context.strip()
        else ""
    )

    st.markdown(
        (
            "<section class='pm-page-header'>"
            f"<h1>{title}</h1>"
            f"<p>{description}</p>"
            f"{context_html}"
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def render_kpi_card(
    *,
    title: str,
    value: str,
    subtitle: str | None = None,
    status: str | None = None,
) -> None:
    """Render a reusable KPI card."""

    import streamlit as st

    subtitle_html = (
        f"<div class='pm-kpi-subtitle'>{escape(subtitle)}</div>"
        if subtitle is not None and subtitle.strip()
        else ""
    )
    status_html = (
        f"<div class='pm-kpi-status'>{escape(status)}</div>"
        if status is not None and status.strip()
        else ""
    )

    st.markdown(
        (
            "<div class='pm-kpi-card'>"
            f"<div class='pm-kpi-label'>{escape(title)}</div>"
            f"<div class='pm-kpi-value'>{escape(value)}</div>"
            f"{subtitle_html}"
            f"{status_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_feature_card(
    *,
    title: str,
    description: str,
) -> None:
    """Render one product capability card."""

    import streamlit as st

    st.markdown(
        (
            "<div class='pm-feature-card'>"
            f"<strong>{title}</strong>"
            f"<p>{description}</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


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


def render_workflow_steps(
    steps: list[str],
) -> None:
    """Render a compact product workflow."""

    import streamlit as st

    step_markup = "".join(
        f"<li><span>{index}</span><strong>{step}</strong></li>"
        for index, step in enumerate(steps, start=1)
    )

    st.markdown(
        f"<ol class='pm-workflow'>{step_markup}</ol>",
        unsafe_allow_html=True,
    )
