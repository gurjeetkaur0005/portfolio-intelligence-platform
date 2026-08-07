from __future__ import annotations

from streamlit_app.services.display import display_timestamp


def test_null_completed_timestamp_displays_not_recorded() -> None:
    """Legacy null completion timestamps render clearly."""

    assert display_timestamp(None) == "Not recorded"


def test_string_timestamp_displays_as_received() -> None:
    """Persisted timestamp strings are not altered."""

    assert (
        display_timestamp("2026-08-07T10:47:25Z")
        == "2026-08-07T10:47:25Z"
    )
