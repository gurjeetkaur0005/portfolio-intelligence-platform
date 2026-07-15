from datetime import date

import pandas as pd


def evaluate_event_triggers(
    portfolios: pd.DataFrame,
    evaluation_date: date,
    market_drop: float = 0.0,
    regulatory_change: bool = False,
    client_life_event: bool = False,
    corporate_action: bool = False,
    large_cash_flow: bool = False,
) -> pd.DataFrame:
    """
    Evaluate event-driven portfolio rebalancing triggers.

    Supported events:
    - Market crash
    - Regulatory change
    - Client life event
    - Corporate action
    - Tax harvesting window
    - Large cash flow
    """

    result = portfolios.copy()

    evaluation_timestamp = pd.Timestamp(evaluation_date)

    result["market_crash_trigger"] = market_drop <= -0.10
    result["regulatory_change_trigger"] = regulatory_change
    result["client_life_event_trigger"] = client_life_event
    result["corporate_action_trigger"] = corporate_action
    result["large_cash_flow_trigger"] = large_cash_flow

    result["tax_harvesting_trigger"] = (
        evaluation_timestamp.month == 3
    )

    event_columns = [
        "market_crash_trigger",
        "regulatory_change_trigger",
        "client_life_event_trigger",
        "corporate_action_trigger",
        "large_cash_flow_trigger",
        "tax_harvesting_trigger",
    ]

    result["event_triggered"] = result[event_columns].any(axis=1)

    result["event_trigger_type"] = "none"
    result["event_priority"] = "none"

    result.loc[
        result["large_cash_flow_trigger"],
        ["event_trigger_type", "event_priority"],
    ] = ["cash_flow", "low"]

    result.loc[
        result["tax_harvesting_trigger"],
        ["event_trigger_type", "event_priority"],
    ] = ["tax_harvesting", "medium"]

    result.loc[
        result["corporate_action_trigger"],
        ["event_trigger_type", "event_priority"],
    ] = ["corporate_action", "high"]

    result.loc[
        result["client_life_event_trigger"],
        ["event_trigger_type", "event_priority"],
    ] = ["client_life_event", "high"]

    result.loc[
        result["regulatory_change_trigger"],
        ["event_trigger_type", "event_priority"],
    ] = ["regulatory_change", "critical"]

    result.loc[
        result["market_crash_trigger"],
        ["event_trigger_type", "event_priority"],
    ] = ["market_crash", "critical"]

    result["trigger_type"] = "event"

    return result


if __name__ == "__main__":
    from src.data.client_profile_generator import generate_client_profiles
    from src.data.portfolio_generator import generate_portfolios

    clients = generate_client_profiles()
    portfolios = generate_portfolios(clients)

    event_results = evaluate_event_triggers(
        portfolios=portfolios,
        evaluation_date=date(2026, 3, 20),
        market_drop=-0.12,
        regulatory_change=False,
        client_life_event=False,
        corporate_action=False,
        large_cash_flow=False,
    )

    columns_to_display = [
        "portfolio_id",
        "risk_category",
        "market_crash_trigger",
        "tax_harvesting_trigger",
        "event_triggered",
        "event_trigger_type",
        "event_priority",
    ]

    print(event_results[columns_to_display].head())

    print("\nEvent trigger distribution:")
    print(event_results["event_trigger_type"].value_counts())