import pandas as pd


def evaluate_threshold_triggers(
    drift_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate threshold-based rebalancing triggers.

    A portfolio is triggered when its maximum absolute drift exceeds
    its assigned drift band.

    Severity levels:
    - none: drift is within the allowed band
    - medium: drift is above the band but less than 1.5 times the band
    - high: drift is at least 1.5 times the band
    - critical: drift is at least 2 times the band
    """

    result = drift_results.copy()

    result["threshold_breached"] = (
        result["max_absolute_drift"] > result["drift_band"]
    )

    result["breach_ratio"] = (
        result["max_absolute_drift"] / result["drift_band"]
    )

    result["trigger_severity"] = "none"

    result.loc[
        result["breach_ratio"] > 1.0,
        "trigger_severity",
    ] = "medium"

    result.loc[
        result["breach_ratio"] >= 1.5,
        "trigger_severity",
    ] = "high"

    result.loc[
        result["breach_ratio"] >= 2.0,
        "trigger_severity",
    ] = "critical"

    result["trigger_type"] = "threshold"

    return result


if __name__ == "__main__":
    from src.data.client_profile_generator import generate_client_profiles
    from src.data.portfolio_generator import generate_portfolios
    from src.monitoring.drift_calculator import calculate_drift

    clients = generate_client_profiles()
    portfolios = generate_portfolios(clients)
    drift_results = calculate_drift(portfolios)

    trigger_results = evaluate_threshold_triggers(drift_results)

    columns_to_display = [
        "portfolio_id",
        "risk_category",
        "drift_band",
        "max_absolute_drift",
        "breach_ratio",
        "threshold_breached",
        "trigger_severity",
    ]

    print(trigger_results[columns_to_display].head(10))

    print("\nTrigger severity distribution:")
    print(trigger_results["trigger_severity"].value_counts())
    