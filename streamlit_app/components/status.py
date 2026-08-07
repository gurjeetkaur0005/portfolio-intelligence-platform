from __future__ import annotations

POSITIVE_STATES = {
    "approved",
    "buy",
    "healthy",
    "hold",
    "ok",
    "ready",
    "success",
}
WARNING_STATES = {
    "not_configured",
    "pending",
    "sell",
}
NEGATIVE_STATES = {
    "failed",
    "rejected",
    "unhealthy",
}


def status_label(status: str | None) -> str:
    """Return a readable status label."""

    if status is None or not status.strip():
        return "Unavailable"

    normalized_status = status.strip()

    return normalized_status.replace("_", " ").title()


def status_tone(status: str | None) -> str:
    """Return the presentation tone for a status."""

    if status is None:
        return "neutral"

    normalized_status = status.strip().lower()

    if normalized_status in POSITIVE_STATES:
        return "positive"

    if normalized_status in WARNING_STATES:
        return "warning"

    if normalized_status in NEGATIVE_STATES:
        return "negative"

    return "neutral"


def approval_status_label(
    *,
    required: bool | None,
    status: str | None,
) -> str:
    """Return a display label for persisted approval metadata."""

    if status is not None and status.strip():
        return status

    if required is False:
        return "Not Required"

    return "Unavailable"


def render_status_badge(
    status: str | None,
) -> None:
    """Render a semantic status badge."""

    import streamlit as st

    tone = status_tone(status)
    label = status_label(status)

    st.markdown(
        (
            f"<span class='pm-status-badge pm-status-{tone}'>"
            f"{label}</span>"
        ),
        unsafe_allow_html=True,
    )
