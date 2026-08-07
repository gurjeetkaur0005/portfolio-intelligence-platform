from __future__ import annotations

import streamlit as st

from streamlit_app.components.navigation import render_sidebar
from streamlit_app.config import get_settings
from streamlit_app.services.api_client import (
    ApiClientError,
    FastApiClient,
    JsonObject,
)


def _build_client() -> FastApiClient:
    """Create the reusable FastAPI client."""

    settings = get_settings()

    return FastApiClient(
        base_url=settings.api_base_url,
        timeout_seconds=settings.api_timeout_seconds,
    )


def _render_status_metric(
    *,
    label: str,
    payload: JsonObject | None,
) -> None:
    """Render one health status metric."""

    st.metric(
        label=label,
        value=_status_label(payload),
    )


def _status_label(
    payload: JsonObject | None,
) -> str:
    """Return a user-facing status label from a health payload."""

    if payload is None:
        return "Unavailable"

    status = payload.get("status")

    if isinstance(status, str):
        normalized_status = status.strip().lower()

        if normalized_status in {
            "healthy",
            "ready",
            "ok",
            "success",
        }:
            return "Healthy"

        if normalized_status == "not_configured":
            return "Not Configured"

        if normalized_status == "unhealthy":
            return "Unhealthy"

        if normalized_status == "failed":
            return "Failed"

        if normalized_status:
            return normalized_status.replace("_", " ").title()

    return "Unavailable"


def main() -> None:
    """Render system health information."""

    settings = get_settings()

    st.set_page_config(
        page_title=f"System Health | {settings.app_title}",
        page_icon="🩺",
        layout="wide",
    )

    render_sidebar(settings)

    st.title("System Health")

    st.caption(
        "Monitor FastAPI, PostgreSQL readiness, and language-model health."
    )

    client = _build_client()

    health: JsonObject | None = None
    readiness: JsonObject | None = None
    llm_health: JsonObject | None = None

    try:
        health = client.get_health()
    except ApiClientError as exc:
        st.warning(
            f"FastAPI health check failed: {exc}"
        )

    try:
        readiness = client.get_readiness()
    except ApiClientError as exc:
        st.warning(
            f"Readiness check failed: {exc}"
        )

    try:
        llm_health = client.get_llm_health()
    except ApiClientError as exc:
        st.warning(
            f"LLM health check failed: {exc}"
        )

    api_column, database_column, llm_column = st.columns(3)

    with api_column:
        _render_status_metric(
            label="FastAPI",
            payload=health,
        )

    with database_column:
        _render_status_metric(
            label="PostgreSQL",
            payload=readiness,
        )

    with llm_column:
        _render_status_metric(
            label="LLM",
            payload=llm_health,
        )

    st.divider()

    st.subheader("Raw Health Responses")

    with st.expander("FastAPI Health"):
        st.json(
            health
            if health is not None
            else {"status": "unavailable"}
        )

    with st.expander("Readiness"):
        st.json(
            readiness
            if readiness is not None
            else {"status": "unavailable"}
        )

    with st.expander("LLM Health"):
        st.json(
            llm_health
            if llm_health is not None
            else {"status": "unavailable"}
        )


if __name__ == "__main__":
    main()
