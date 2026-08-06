from __future__ import annotations

import streamlit as st

from streamlit_app.components.navigation import render_sidebar
from streamlit_app.config import get_settings


def main() -> None:
    """Render the main Streamlit application shell."""

    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    render_sidebar(settings)

    st.title(settings.app_title)
    st.subheader(
        "AI-assisted portfolio monitoring and rebalancing"
    )

    st.info(
        "Use the sidebar to open the dashboard and other pages."
    )

    st.markdown(
        """
        This Streamlit application is a frontend client only.

        It communicates with the existing FastAPI backend through HTTP.
        It does not directly access PostgreSQL or import backend business
        logic.
        """
    )


if __name__ == "__main__":
    main()