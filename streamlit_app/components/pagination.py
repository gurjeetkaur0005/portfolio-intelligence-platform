from __future__ import annotations

from dataclasses import dataclass

from streamlit_app.services.api_client import PaginatedResponse


DEFAULT_PAGE_LIMIT = 20


@dataclass(frozen=True, slots=True)
class PaginationSummary:
    """Represent display metadata for one backend page."""

    label: str
    page_number: int
    previous_disabled: bool
    next_disabled: bool


def page_offset(
    *,
    page_index: int,
    limit: int = DEFAULT_PAGE_LIMIT,
) -> int:
    """Return the SQL offset for a zero-based page index."""

    normalized_page_index = max(page_index, 0)

    return normalized_page_index * limit


def page_summary(
    *,
    page: PaginatedResponse,
    page_index: int,
) -> PaginationSummary:
    """Return display metadata for a paginated backend response."""

    normalized_page_index = max(page_index, 0)
    page_number = normalized_page_index + 1

    if page.count <= 0:
        label = "Showing 0 records"
    else:
        first_record = page.offset + 1
        last_record = page.offset + page.count
        label = f"Showing {first_record}-{last_record}"

    return PaginationSummary(
        label=label,
        page_number=page_number,
        previous_disabled=normalized_page_index == 0,
        next_disabled=page.count < page.limit,
    )


def reset_page_on_selection_change(
    *,
    selection_key: str,
    selected_value: str,
    page_keys: tuple[str, ...],
) -> None:
    """Reset pages when a parent selection changes."""

    import streamlit as st

    previous_value = st.session_state.get(selection_key)

    if previous_value == selected_value:
        return

    st.session_state[selection_key] = selected_value

    for page_key in page_keys:
        st.session_state[page_key] = 0


def current_page_index(
    page_key: str,
) -> int:
    """Return the current zero-based page index."""

    import streamlit as st

    value = st.session_state.get(page_key, 0)

    if not isinstance(value, int):
        return 0

    return max(value, 0)


def render_pagination_controls(
    *,
    page: PaginatedResponse,
    page_key: str,
) -> None:
    """Render Previous and Next controls for one backend page."""

    import streamlit as st

    page_index = current_page_index(page_key)
    summary = page_summary(
        page=page,
        page_index=page_index,
    )

    left, middle, right = st.columns([1, 2, 1])

    with left:
        if st.button(
            "Previous",
            key=f"{page_key}_previous",
            disabled=summary.previous_disabled,
        ):
            st.session_state[page_key] = max(page_index - 1, 0)
            st.rerun()

    with middle:
        st.caption(f"{summary.label} | Page {summary.page_number}")

    with right:
        if st.button(
            "Next",
            key=f"{page_key}_next",
            disabled=summary.next_disabled,
        ):
            st.session_state[page_key] = page_index + 1
            st.rerun()
