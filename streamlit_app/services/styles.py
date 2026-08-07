from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _stylesheet() -> str:
    """Return the shared Streamlit stylesheet."""

    return (
        Path(__file__)
        .resolve()
        .parents[1]
        .joinpath("styles.css")
        .read_text(encoding="utf-8")
    )


def load_global_styles() -> None:
    """Load shared Streamlit visual styles."""

    import streamlit as st

    st.markdown(
        f"<style>{_stylesheet()}</style>",
        unsafe_allow_html=True,
    )
