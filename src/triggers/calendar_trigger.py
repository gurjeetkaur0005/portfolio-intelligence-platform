from datetime import date

import pandas as pd


def evaluate_calendar_triggers(
    portfolios: pd.DataFrame,
    evaluation_date: date,
) -> pd.DataFrame:
    """
    Evaluate calendar-based portfolio review triggers.

    Trigger rules:
    - Monthly review: first business day of the month
    - Quarterly review: January, April, July, or October
    - Annual review: January
    """

    result = portfolios.copy()

    evaluation_timestamp = pd.Timestamp(evaluation_date)

    result["monthly_review_due"] = False
    result["quarterly_review_due"] = False
    result["annual_review_due"] = False

    is_weekday = evaluation_timestamp.weekday() < 5

    is_first_week_of_month = evaluation_timestamp.day <= 7

    if is_weekday and is_first_week_of_month:
        result["monthly_review_due"] = True

    if (
        is_weekday
        and is_first_week_of_month
        and evaluation_timestamp.month in [1, 4, 7, 10]
    ):
        result["quarterly_review_due"] = True

    if (
        is_weekday
        and is_first_week_of_month
        and evaluation_timestamp.month == 1
    ):
        result["annual_review_due"] = True

    result["calendar_triggered"] = (
        result["monthly_review_due"]
        | result["quarterly_review_due"]
        | result["annual_review_due"]
    )

    result["calendar_trigger_type"] = "none"

    result.loc[
        result["monthly_review_due"],
        "calendar_trigger_type",
    ] = "monthly"

    result.loc[
        result["quarterly_review_due"],
        "calendar_trigger_type",
    ] = "quarterly"

    result.loc[
        result["annual_review_due"],
        "calendar_trigger_type",
    ] = "annual"

    result["trigger_type"] = "calendar"

    return result


if __name__ == "__main__":
    from src.data.client_profile_generator import generate_client_profiles
    from src.data.portfolio_generator import generate_portfolios

    clients = generate_client_profiles()
    portfolios = generate_portfolios(clients)

    evaluation_date = date(2026, 7, 1)

    calendar_results = evaluate_calendar_triggers(
        portfolios,
        evaluation_date,
    )

    columns_to_display = [
        "portfolio_id",
        "risk_category",
        "monthly_review_due",
        "quarterly_review_due",
        "annual_review_due",
        "calendar_triggered",
        "calendar_trigger_type",
    ]

    print(calendar_results[columns_to_display].head())

    print("\nCalendar trigger distribution:")
    print(calendar_results["calendar_trigger_type"].value_counts())