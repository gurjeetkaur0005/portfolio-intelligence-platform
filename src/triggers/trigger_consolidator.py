import pandas as pd


PRIORITY_ORDER = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


def consolidate_triggers(
    threshold_results: pd.DataFrame,
    calendar_results: pd.DataFrame,
    event_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine threshold, calendar, and event triggers.

    For each portfolio:
    - preserve all trigger information
    - identify whether any trigger fired
    - choose the highest-priority trigger
    """

    threshold_columns = [
        "portfolio_id",
        "threshold_breached",
        "trigger_severity",
        "breach_ratio",
    ]

    calendar_columns = [
        "portfolio_id",
        "calendar_triggered",
        "calendar_trigger_type",
    ]

    event_columns = [
        "portfolio_id",
        "event_triggered",
        "event_trigger_type",
        "event_priority",
    ]

    result = threshold_results[threshold_columns].copy()

    result = result.merge(
        calendar_results[calendar_columns],
        on="portfolio_id",
        how="left",
    )

    result = result.merge(
        event_results[event_columns],
        on="portfolio_id",
        how="left",
    )

    result["any_triggered"] = (
        result["threshold_breached"]
        | result["calendar_triggered"]
        | result["event_triggered"]
    )

    result["final_trigger_type"] = "none"
    result["final_priority"] = "none"

    # Calendar triggers start with low priority.
    result.loc[
        result["calendar_triggered"],
        ["final_trigger_type", "final_priority"],
    ] = ["calendar", "low"]

    # Threshold triggers use their calculated severity.
    threshold_mask = result["threshold_breached"]

    result.loc[
        threshold_mask,
        "final_trigger_type",
    ] = "threshold"

    result.loc[
        threshold_mask,
        "final_priority",
    ] = result.loc[
        threshold_mask,
        "trigger_severity",
    ]

    # Event triggers may override lower-priority triggers.
    for index, row in result.iterrows():
        if not row["event_triggered"]:
            continue

        current_priority_value = PRIORITY_ORDER[
            row["final_priority"]
        ]

        event_priority_value = PRIORITY_ORDER[
            row["event_priority"]
        ]

        if event_priority_value >= current_priority_value:
            result.at[index, "final_trigger_type"] = "event"
            result.at[index, "final_priority"] = row["event_priority"]

    result["contributing_triggers"] = result.apply(
        build_contributing_trigger_list,
        axis=1,
    )

    return result


def build_contributing_trigger_list(
    row: pd.Series,
) -> str:
    """
    Return all triggers that contributed to the final decision.
    """

    triggers = []

    if row["threshold_breached"]:
        triggers.append("threshold")

    if row["calendar_triggered"]:
        triggers.append("calendar")

    if row["event_triggered"]:
        triggers.append(
            f"event:{row['event_trigger_type']}"
        )

    if not triggers:
        return "none"

    return ", ".join(triggers)


if __name__ == "__main__":
    from datetime import date

    from src.data.client_profile_generator import (
        generate_client_profiles,
    )
    from src.data.portfolio_generator import generate_portfolios
    from src.monitoring.drift_calculator import calculate_drift
    from src.triggers.calendar_trigger import (
        evaluate_calendar_triggers,
    )
    from src.triggers.event_trigger import evaluate_event_triggers
    from src.triggers.threshold_trigger import (
        evaluate_threshold_triggers,
    )

    clients = generate_client_profiles()
    portfolios = generate_portfolios(clients)

    drift_results = calculate_drift(portfolios)

    threshold_results = evaluate_threshold_triggers(
        drift_results
    )

    calendar_results = evaluate_calendar_triggers(
        portfolios,
        evaluation_date=date(2026, 7, 1),
    )

    event_results = evaluate_event_triggers(
        portfolios=portfolios,
        evaluation_date=date(2026, 7, 1),
        market_drop=-0.12,
    )

    consolidated_results = consolidate_triggers(
        threshold_results=threshold_results,
        calendar_results=calendar_results,
        event_results=event_results,
    )

    columns_to_display = [
        "portfolio_id",
        "threshold_breached",
        "calendar_triggered",
        "event_triggered",
        "any_triggered",
        "final_trigger_type",
        "final_priority",
        "contributing_triggers",
    ]

    print(
        consolidated_results[
            columns_to_display
        ].head(10)
    )

    print("\nFinal priority distribution:")
    print(
        consolidated_results[
            "final_priority"
        ].value_counts()
    )