from __future__ import annotations

import streamlit as st

from streamlit_app.components.cards import (
    render_key_value,
    render_kpi_card,
)
from streamlit_app.components.navigation import render_sidebar
from streamlit_app.components.status import (
    payload_status_label,
    render_status_badge,
)
from streamlit_app.config import get_settings
from streamlit_app.services.api_client import (
    ApiClientError,
    FastApiClient,
    JsonObject,
    JsonValue,
)
from streamlit_app.services.display import display_label
from streamlit_app.services.styles import load_global_styles


def _build_client() -> FastApiClient:
    """Create the reusable FastAPI client."""

    settings = get_settings()

    return FastApiClient(
        base_url=settings.api_base_url,
        timeout_seconds=settings.api_timeout_seconds,
    )


def _status_label(
    payload: JsonObject | None,
) -> str:
    """Return a user-facing status label from a health payload."""

    return payload_status_label(payload)


def _render_status_metric(
    *,
    label: str,
    payload: JsonObject | None,
) -> None:
    """Render one health status card."""

    render_kpi_card(
        title=label,
        value=_status_label(payload),
    )
    render_status_badge(_status_label(payload))


def _display_value(
    value: JsonValue,
) -> str:
    """Return a safe display value for health detail fields."""

    if value is None:
        return "Unavailable"

    if isinstance(value, bool):
        return "Yes" if value else "No"

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, str) and value.strip():
        return value

    return "Unavailable"


def _render_service_details(
    *,
    title: str,
    payload: JsonObject | None,
    fields: tuple[str, ...],
) -> None:
    """Render useful backend-provided health fields."""

    with st.container(border=True):
        st.markdown(f"#### {title}")

        if payload is None:
            render_key_value(
                label="Status",
                value="Unavailable",
            )
            return

        rendered = False

        for field_name in fields:
            if field_name not in payload:
                continue

            render_key_value(
                label=display_label(field_name),
                value=_display_value(payload.get(field_name)),
            )
            rendered = True

        if not rendered:
            render_key_value(
                label="Status",
                value=_status_label(payload),
            )


def main() -> None:
    """Render system health information."""

    settings = get_settings()

    st.set_page_config(
        page_title=f"System Health | {settings.app_title}",
        page_icon="🩺",
        layout="wide",
    )
    load_global_styles()

    render_sidebar(settings)

    st.title("System Health")
    st.markdown(
        (
            "<div class='pm-page-caption'>"
            "Monitor FastAPI, PostgreSQL readiness, and "
            "language-model health."
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    client = _build_client()

    health: JsonObject | None = None
    readiness: JsonObject | None = None
    llm_health: JsonObject | None = None

    try:
        health = client.get_health()
    except ApiClientError as exc:
        st.warning(f"FastAPI health check failed: {exc}")

    try:
        readiness = client.get_readiness()
    except ApiClientError as exc:
        st.warning(f"Readiness check failed: {exc}")

    try:
        llm_health = client.get_llm_health()
    except ApiClientError as exc:
        st.warning(f"LLM health check failed: {exc}")

    st.subheader("Service Status")

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
            label="Gemini / LLM",
            payload=llm_health,
        )

    st.subheader("Service Details")

    detail_api, detail_database, detail_llm = st.columns(3)

    with detail_api:
        _render_service_details(
            title="FastAPI",
            payload=health,
            fields=(
                "status",
                "message",
            ),
        )

    with detail_database:
        _render_service_details(
            title="PostgreSQL",
            payload=readiness,
            fields=(
                "status",
                "database",
                "configuration",
            ),
        )

    with detail_llm:
        _render_service_details(
            title="Gemini / LLM",
            payload=llm_health,
            fields=(
                "status",
                "provider",
                "model_name",
                "configured",
                "live_check_performed",
            ),
        )

    st.subheader("Raw Responses")

    with st.expander("FastAPI Health", expanded=False):
        st.json(
            health
            if health is not None
            else {"status": "unavailable"}
        )

    with st.expander("Readiness", expanded=False):
        st.json(
            readiness
            if readiness is not None
            else {"status": "unavailable"}
        )

    with st.expander("LLM Health", expanded=False):
        st.json(
            llm_health
            if llm_health is not None
            else {"status": "unavailable"}
        )


if __name__ == "__main__":
    main()
