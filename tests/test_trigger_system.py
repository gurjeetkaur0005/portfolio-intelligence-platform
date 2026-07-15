from datetime import date

import pandas as pd

from src.triggers.calendar_trigger import evaluate_calendar_triggers
from src.triggers.event_trigger import evaluate_event_triggers
from src.triggers.threshold_trigger import evaluate_threshold_triggers
from src.triggers.trigger_consolidator import consolidate_triggers


def create_sample_drift_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "portfolio_id": [
                "P00001",
                "P00002",
                "P00003",
                "P00004",
            ],
            "drift_band": [
                0.03,
                0.03,
                0.03,
                0.03,
            ],
            "max_absolute_drift": [
                0.02,
                0.035,
                0.045,
                0.065,
            ],
        }
    )


def create_sample_portfolios() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "portfolio_id": [
                "P00001",
                "P00002",
                "P00003",
                "P00004",
            ],
            "risk_category": [
                "balanced",
                "balanced",
                "balanced",
                "balanced",
            ],
        }
    )


def test_threshold_severity_classification() -> None:
    drift_results = create_sample_drift_results()

    result = evaluate_threshold_triggers(drift_results)

    assert result.loc[0, "trigger_severity"] == "none"
    assert result.loc[1, "trigger_severity"] == "medium"
    assert result.loc[2, "trigger_severity"] == "high"
    assert result.loc[3, "trigger_severity"] == "critical"


def test_threshold_boundary_is_not_breached() -> None:
    drift_results = pd.DataFrame(
        {
            "portfolio_id": ["P00001"],
            "drift_band": [0.03],
            "max_absolute_drift": [0.03],
        }
    )

    result = evaluate_threshold_triggers(drift_results)

    assert not bool(result.loc[0, "threshold_breached"])
    assert result.loc[0, "trigger_severity"] == "none"


def test_quarterly_calendar_trigger() -> None:
    portfolios = create_sample_portfolios()

    result = evaluate_calendar_triggers(
        portfolios,
        evaluation_date=date(2026, 7, 1),
    )

    assert result["monthly_review_due"].all()
    assert result["quarterly_review_due"].all()
    assert not result["annual_review_due"].any()
    assert (result["calendar_trigger_type"] == "quarterly").all()


def test_annual_calendar_trigger() -> None:
    portfolios = create_sample_portfolios()

    result = evaluate_calendar_triggers(
        portfolios,
        evaluation_date=date(2026, 1, 2),
    )

    assert result["annual_review_due"].all()
    assert (result["calendar_trigger_type"] == "annual").all()


def test_no_calendar_trigger_outside_first_week() -> None:
    portfolios = create_sample_portfolios()

    result = evaluate_calendar_triggers(
        portfolios,
        evaluation_date=date(2026, 7, 15),
    )

    assert not result["calendar_triggered"].any()
    assert (result["calendar_trigger_type"] == "none").all()


def test_market_crash_event_trigger() -> None:
    portfolios = create_sample_portfolios()

    result = evaluate_event_triggers(
        portfolios=portfolios,
        evaluation_date=date(2026, 7, 15),
        market_drop=-0.12,
    )

    assert result["market_crash_trigger"].all()
    assert result["event_triggered"].all()
    assert (result["event_trigger_type"] == "market_crash").all()
    assert (result["event_priority"] == "critical").all()


def test_tax_harvesting_event_in_march() -> None:
    portfolios = create_sample_portfolios()

    result = evaluate_event_triggers(
        portfolios=portfolios,
        evaluation_date=date(2026, 3, 10),
    )

    assert result["tax_harvesting_trigger"].all()
    assert (result["event_trigger_type"] == "tax_harvesting").all()
    assert (result["event_priority"] == "medium").all()


def test_no_event_trigger() -> None:
    portfolios = create_sample_portfolios()

    result = evaluate_event_triggers(
        portfolios=portfolios,
        evaluation_date=date(2026, 7, 15),
    )

    assert not result["event_triggered"].any()
    assert (result["event_trigger_type"] == "none").all()


def test_event_overrides_lower_priority_triggers() -> None:
    portfolios = create_sample_portfolios()
    drift_results = create_sample_drift_results()

    threshold_results = evaluate_threshold_triggers(drift_results)

    calendar_results = evaluate_calendar_triggers(
        portfolios,
        evaluation_date=date(2026, 7, 1),
    )

    event_results = evaluate_event_triggers(
        portfolios=portfolios,
        evaluation_date=date(2026, 7, 1),
        market_drop=-0.12,
    )

    result = consolidate_triggers(
        threshold_results,
        calendar_results,
        event_results,
    )

    assert (result["final_trigger_type"] == "event").all()
    assert (result["final_priority"] == "critical").all()


def test_contributing_triggers_are_preserved() -> None:
    portfolios = create_sample_portfolios()
    drift_results = create_sample_drift_results()

    threshold_results = evaluate_threshold_triggers(drift_results)

    calendar_results = evaluate_calendar_triggers(
        portfolios,
        evaluation_date=date(2026, 7, 1),
    )

    event_results = evaluate_event_triggers(
        portfolios=portfolios,
        evaluation_date=date(2026, 7, 1),
        market_drop=-0.12,
    )

    result = consolidate_triggers(
        threshold_results,
        calendar_results,
        event_results,
    )

    assert result.loc[0, "contributing_triggers"] == (
        "calendar, event:market_crash"
    )

    assert result.loc[3, "contributing_triggers"] == (
        "threshold, calendar, event:market_crash"
    )