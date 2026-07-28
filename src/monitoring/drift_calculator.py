import numpy as np
import pandas as pd

from config.asset_classes import ASSET_CLASSES


def calculate_drift(portfolios: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate portfolio drift metrics.

    Parameters
    ----------
    portfolios:
        DataFrame containing target and current allocations.

    Returns
    -------
    pd.DataFrame
        Original portfolio data with additional drift metrics.
    """

    result = portfolios.copy()

    drift_columns = []

    for asset in ASSET_CLASSES:
        target_column = f"target_{asset}"
        current_column = f"current_{asset}"
        drift_column = f"drift_{asset}"

        result[drift_column] = (
            result[current_column] - result[target_column]
        )

        drift_columns.append(drift_column)

    result["max_absolute_drift"] = (
        result[drift_columns]
        .abs()
        .max(axis=1)
    )

    result["sum_absolute_drift"] = (
        result[drift_columns]
        .abs()
        .sum(axis=1)
    )

    result["rms_drift"] = np.sqrt(
        result[drift_columns]
        .pow(2)
        .mean(axis=1)
    )

    result["requires_rebalancing"] = (
        result["max_absolute_drift"] > result["drift_band"]
    )

    return result


if __name__ == "__main__":
    from src.data.client_profile_generator import generate_client_profiles
    from src.data.portfolio_generator import generate_portfolios

    client_profiles = generate_client_profiles()

    portfolios = generate_portfolios(client_profiles)

    drift_results = calculate_drift(portfolios)

    columns_to_display = [
        "portfolio_id",
        "risk_category",
        "drift_band",
        "max_absolute_drift",
        "sum_absolute_drift",
        "rms_drift",
        "requires_rebalancing",
    ]

    print(drift_results[columns_to_display].head(10))

    print("\nPortfolios requiring rebalancing:")
    print(drift_results["requires_rebalancing"].value_counts())
