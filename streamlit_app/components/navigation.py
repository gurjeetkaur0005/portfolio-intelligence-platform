from __future__ import annotations

import streamlit as st

from streamlit_app.config import FrontendSettings


def render_sidebar(settings: FrontendSettings) -> None:
    """Render shared sidebar information."""

    with st.sidebar:
        st.title("PortfolioMind")
        st.caption("AI-Powered Portfolio Intelligence Platform")
        st.divider()

        st.markdown("### Architecture")
        st.caption(
            "Streamlit → FastAPI → Application Services → PostgreSQL"
        )

        st.divider()

        st.markdown("### Backend API")
        st.code(settings.api_base_url, language=None)

        st.divider()

        st.caption(
            "The frontend does not connect directly to PostgreSQL "
            "or duplicate portfolio calculations."
        )