from __future__ import annotations

from streamlit_app.components.cards import (
    render_key_value,
    render_kpi_card,
    render_page_header,
)
from streamlit_app.components.charts import (
    _format_asset_label,
    render_rebalance_allocation_comparison,
)
from streamlit_app.components.metrics import format_currency
from streamlit_app.components.navigation import render_sidebar
from streamlit_app.components.status import (
    approval_status_label,
    render_status_badge,
    status_label,
)
from streamlit_app.components.tables import (
    render_rebalance_audit_table,
    render_rebalance_trade_table,
)
from streamlit_app.config import get_settings
from streamlit_app.services.api_client import (
    ApiClientError,
    FastApiClient,
    JsonObject,
    JsonValue,
    PaginatedResponse,
)
from streamlit_app.services.display import display_timestamp
from streamlit_app.services.styles import load_global_styles


def _build_client() -> FastApiClient:
    """Create the reusable FastAPI client."""

    settings = get_settings()

    return FastApiClient(
        base_url=settings.api_base_url,
        timeout_seconds=settings.api_timeout_seconds,
    )


def _extract_portfolio_ids(
    portfolio_page: PaginatedResponse,
) -> list[str]:
    """Extract valid portfolio identifiers from a portfolio page."""

    portfolio_ids: list[str] = []

    for portfolio in portfolio_page.items:
        portfolio_id = portfolio.get("portfolio_id")

        if isinstance(portfolio_id, str) and portfolio_id.strip():
            portfolio_ids.append(portfolio_id)

    return portfolio_ids


def _text_value(
    value: JsonValue,
    *,
    fallback: str = "Unavailable",
) -> str:
    """Return a safe compact text value from a JSON field."""

    if value is None:
        return fallback

    if isinstance(value, bool):
        return "Yes" if value else "No"

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, str) and value.strip():
        return value

    return fallback


def _percentage_value(
    value: JsonValue,
) -> str:
    """Format a decimal backend value as a percentage string."""

    if isinstance(value, bool) or value is None:
        return "Unavailable"

    try:
        numeric_value = float(str(value))
    except ValueError:
        return "Unavailable"

    return f"{numeric_value * 100:.2f}%"


def _string_list_value(
    value: JsonValue,
) -> str:
    """Return a readable list value without changing backend data."""

    if isinstance(value, list):
        labels: list[str] = []

        for item in value:
            if isinstance(item, str) and item.strip():
                labels.append(item)

        return ", ".join(labels) if labels else "Not available"

    return _text_value(
        value,
        fallback="Not available",
    )


def _extract_approval(
    trade: JsonObject,
) -> JsonObject | None:
    """Return nested approval metadata when available."""

    approval = trade.get("approval")

    if not isinstance(approval, dict):
        return None

    return approval


def _approval_status(
    trade: JsonObject,
) -> str:
    """Return the persisted approval status for display."""

    approval = _extract_approval(trade)

    if approval is None:
        return "Not Required"

    required = approval.get("required")
    status = approval.get("status")

    normalized_status = status if isinstance(status, str) else None
    normalized_required = required if isinstance(required, bool) else None

    return approval_status_label(
        required=normalized_required,
        status=normalized_status,
    )


def _rebalance_request_active() -> bool:
    """Return whether the current Streamlit session is submitting."""

    import streamlit as st

    return bool(st.session_state.get("rebalance_in_progress", False))


def _set_rebalance_request_active(
    active: bool,
) -> None:
    """Store the current rebalance submission state."""

    import streamlit as st

    st.session_state["rebalance_in_progress"] = active


def _render_command_result(
    result: JsonObject,
) -> None:
    """Render the lightweight POST rebalance response."""

    import streamlit as st

    message = result.get("message")
    run_id = result.get("run_id")

    with st.container(border=True):
        if isinstance(message, str) and message.strip():
            st.success(message)
        else:
            st.success("Rebalance completed successfully.")

        if isinstance(run_id, str) and run_id.strip():
            render_key_value(
                label="Run ID",
                value=run_id,
            )


def _render_summary_kpis(
    summary: JsonObject,
) -> None:
    """Render persisted rebalance summary KPI cards."""

    import streamlit as st

    first, second, third, fourth = st.columns(4)

    with first:
        render_kpi_card(
            title="Status",
            value=status_label(
                _text_value(summary.get("status"))
            ),
        )

    with second:
        render_kpi_card(
            title="Portfolio Value",
            value=format_currency(
                summary.get("portfolio_value")
            ),
        )

    with third:
        render_kpi_card(
            title="Trade Count",
            value=_text_value(
                summary.get("trade_count")
            ),
        )

    with fourth:
        render_kpi_card(
            title="Transaction Cost",
            value=format_currency(
                summary.get("transaction_cost")
            ),
        )

    fifth, sixth, seventh = st.columns(3)

    with fifth:
        render_kpi_card(
            title="Estimated Tax",
            value=format_currency(
                summary.get("estimated_tax_liability")
            ),
        )

    with sixth:
        render_kpi_card(
            title="Approvals Required",
            value=_text_value(
                summary.get("approval_required_count")
            ),
        )

    with seventh:
        render_kpi_card(
            title="Pending Approvals",
            value=_text_value(
                summary.get("pending_approval_count")
            ),
        )


def _render_run_information(
    summary: JsonObject,
) -> None:
    """Render compact persisted run metadata."""

    import streamlit as st

    with st.container(border=True):
        st.markdown("#### Run Information")

        left, right = st.columns(2)

        with left:
            render_key_value(
                label="Run ID",
                value=_text_value(summary.get("run_id")),
            )
            render_key_value(
                label="Portfolio ID",
                value=_text_value(summary.get("portfolio_id")),
            )
            render_key_value(
                label="Transaction Cost Rate",
                value=_percentage_value(
                    summary.get("transaction_cost_rate")
                ),
            )

        with right:
            render_key_value(
                label="Created",
                value=display_timestamp(summary.get("created_at")),
            )
            render_key_value(
                label="Completed",
                value=display_timestamp(summary.get("completed_at")),
            )


def _render_rebalance_summary(
    summary: JsonObject,
) -> None:
    """Render persisted rebalance run information."""

    import streamlit as st

    st.subheader("Run Summary")
    _render_summary_kpis(summary)
    _render_run_information(summary)


def _render_trigger_details(
    trade: JsonObject,
) -> None:
    """Render trigger details for one trade."""

    import streamlit as st

    st.markdown("##### Trigger Context")

    left, right = st.columns(2)

    with left:
        render_key_value(
            label="Threshold Breached",
            value=_text_value(trade.get("threshold_breached")),
        )
        render_key_value(
            label="Threshold Severity",
            value=_text_value(
                trade.get("threshold_severity"),
                fallback="Not available",
            ),
        )
        render_key_value(
            label="Breach Ratio",
            value=_text_value(
                trade.get("breach_ratio"),
                fallback="Not available",
            ),
        )

    with right:
        render_key_value(
            label="Final Trigger",
            value=_text_value(
                trade.get("final_trigger_type"),
                fallback="Not available",
            ),
        )
        render_key_value(
            label="Final Priority",
            value=_text_value(
                trade.get("final_priority"),
                fallback="Not available",
            ),
        )
        render_key_value(
            label="Contributing Triggers",
            value=_string_list_value(
                trade.get("contributing_triggers")
            ),
        )


def _render_explanation_text(
    *,
    title: str,
    value: JsonValue,
) -> None:
    """Render one explanation field."""

    import streamlit as st

    st.markdown(f"##### {title}")

    if isinstance(value, str) and value.strip():
        st.write(value)
        return

    st.caption("Not available")


def _render_approval_details(
    trade: JsonObject,
) -> None:
    """Render approval details for one trade."""

    import streamlit as st

    st.markdown("##### Approval")

    approval = _extract_approval(trade)

    if approval is None:
        render_status_badge("Not Required")
        return

    left, right = st.columns(2)

    with left:
        render_key_value(
            label="Required",
            value=_text_value(approval.get("required")),
        )
        st.write("Status")
        render_status_badge(_approval_status(trade))
        render_key_value(
            label="Reason",
            value=_text_value(
                approval.get("reason"),
                fallback="Not available",
            ),
        )

    with right:
        render_key_value(
            label="Reviewed By",
            value=_text_value(
                approval.get("reviewed_by"),
                fallback="Not recorded",
            ),
        )
        render_key_value(
            label="Reviewed At",
            value=display_timestamp(approval.get("reviewed_at")),
        )


def _render_trade_explanations(
    trade_page: PaginatedResponse,
) -> None:
    """Render explanations and approval metadata per trade."""

    import streamlit as st

    st.subheader("Trade Details")

    if not trade_page.items:
        st.info("No trade details are available for this run.")
        return

    for trade in trade_page.items:
        asset = trade.get("asset")
        action = trade.get("action")

        if not isinstance(asset, str) or not asset.strip():
            continue

        action_label = (
            action
            if isinstance(action, str) and action.strip()
            else "Unknown"
        )

        with st.expander(
            f"{_format_asset_label(asset)} - {action_label}"
        ):
            st.markdown("##### Recommendation")
            render_status_badge(action_label)

            _render_trigger_details(trade)
            st.divider()

            _render_explanation_text(
                title="Client Explanation",
                value=trade.get("client_explanation"),
            )
            _render_explanation_text(
                title="Advisor Explanation",
                value=trade.get("advisor_explanation"),
            )
            _render_explanation_text(
                title="Compliance Explanation",
                value=trade.get("compliance_explanation"),
            )
            st.divider()

            _render_approval_details(trade)


def _render_trades(
    trade_page: PaginatedResponse,
) -> None:
    """Render persisted trades for the current rebalance run."""

    import streamlit as st

    st.subheader("Allocation Change")

    with st.container(border=True):
        render_rebalance_allocation_comparison(trade_page.items)

    st.subheader("Recommended Trades")
    with st.container(border=True):
        render_rebalance_trade_table(trade_page.items)

    st.caption(
        f"Showing {trade_page.count} trade(s), "
        f"limit={trade_page.limit}, "
        f"offset={trade_page.offset}."
    )

    _render_trade_explanations(trade_page)


def _render_audit(
    audit_page: PaginatedResponse,
) -> None:
    """Render persisted audit records."""

    import streamlit as st

    st.subheader("Audit Trail")
    render_rebalance_audit_table(audit_page.items)

    st.caption(
        f"Showing {audit_page.count} audit record(s), "
        f"limit={audit_page.limit}, "
        f"offset={audit_page.offset}."
    )


def _load_run_results(
    *,
    client: FastApiClient,
    run_id: str,
) -> None:
    """Load and render persisted details for one rebalance run."""

    import streamlit as st

    try:
        summary = client.get_rebalance(run_id)
    except ApiClientError as exc:
        st.warning(
            "The rebalance completed, but its summary "
            f"could not be loaded: {exc}"
        )
        return

    _render_rebalance_summary(summary)

    try:
        trade_page = client.list_rebalance_trades(
            run_id=run_id,
            limit=20,
            offset=0,
        )
    except ApiClientError as exc:
        st.warning(f"Could not load rebalance trades: {exc}")
    else:
        _render_trades(trade_page)

    try:
        audit_page = client.list_rebalance_audit(
            run_id=run_id,
            limit=20,
            offset=0,
        )
    except ApiClientError as exc:
        st.warning(f"Could not load the audit trail: {exc}")
    else:
        _render_audit(audit_page)


def main() -> None:
    """Render the database-backed portfolio rebalancing page."""

    import streamlit as st

    settings = get_settings()

    st.set_page_config(
        page_title=f"Rebalancing | {settings.app_title}",
        page_icon="⚖️",
        layout="wide",
    )
    load_global_styles()

    render_sidebar(settings)

    render_page_header(
        title="Rebalancing",
        description=(
            "Review a portfolio and generate its latest rebalance "
            "recommendations."
        ),
    )

    client = _build_client()

    try:
        portfolio_page = client.list_portfolios(
            limit=20,
            offset=0,
        )
    except ApiClientError as exc:
        st.error(f"Could not load portfolios: {exc}")
        return

    portfolio_ids = _extract_portfolio_ids(portfolio_page)

    if not portfolio_ids:
        st.info("No stored portfolios are available for rebalancing.")
        return

    st.subheader("Rebalance Setup")

    with st.container(border=True):
        st.caption(
            "Portfolio value and holdings are loaded from the platform API. "
            "The frontend only submits the recommendation request."
        )

        selected_portfolio = st.selectbox(
            label="Portfolio",
            options=portfolio_ids,
        )

        transaction_cost_rate = st.number_input(
            label="Transaction Cost Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.002,
            step=0.001,
            format="%.4f",
            help="0.002 represents a transaction cost rate of 0.2%.",
        )

        run_button = st.button(
            label="Generate Recommendations",
            type="primary",
            disabled=_rebalance_request_active(),
        )

    if run_button:
        _set_rebalance_request_active(True)
        try:
            with st.spinner("Generating recommendations..."):
                result = client.run_portfolio_rebalance(
                    portfolio_id=selected_portfolio,
                    transaction_cost_rate=float(transaction_cost_rate),
                )
        except (ApiClientError, ValueError) as exc:
            st.error(f"Rebalance failed: {exc}")
            return
        finally:
            _set_rebalance_request_active(False)

        run_id_value = result.get("run_id")

        if not isinstance(run_id_value, str) or not run_id_value.strip():
            st.error(
                "The backend completed the request but did not "
                "return a valid run ID."
            )
            return

        st.session_state["rebalance_run_id"] = run_id_value
        _render_command_result(result)

    stored_run_id = st.session_state.get("rebalance_run_id")

    if isinstance(stored_run_id, str) and stored_run_id.strip():
        st.divider()
        _load_run_results(
            client=client,
            run_id=stored_run_id,
        )


if __name__ == "__main__":
    main()
