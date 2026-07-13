import numpy as np
import pandas as pd

from config.asset_classes import ASSET_CLASSES
from config.risk_categories import RISK_CATEGORIES
from config.settings import RANDOM_SEED
from src.data.client_profile_generator import generate_client_profiles


def generate_portfolios(
    client_profiles: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate synthetic investment portfolios for the supplied clients.

    Each portfolio contains:
    - Portfolio ID
    - Risk category
    - Target asset allocation
    - Current asset allocation after simulated market movement
    - Drift band
    """

    random_generator = np.random.default_rng(seed)

    rows = []

    for client in client_profiles.itertuples(index=False):
        category = client.risk_category
        portfolio_id = client.portfolio_id

        category_config = RISK_CATEGORIES[category]

        target = np.array(
            category_config["target"],
            dtype=float,
        )

        drift_band = category_config["drift_band"]

        noise = random_generator.normal(
            loc=0,
            scale=drift_band * 0.8,
            size=len(ASSET_CLASSES),
        )

        current = target + noise

        # Asset allocations cannot be negative.
        current = np.clip(current, 0, None)

        # Ensure that all current allocations add up to 100%.
        current = current / current.sum()

        row = {
            "portfolio_id": portfolio_id,
            "risk_category": category,
            "drift_band": drift_band,
        }

        for index, asset in enumerate(ASSET_CLASSES):
            row[f"target_{asset}"] = target[index]
            row[f"current_{asset}"] = current[index]

        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    client_profiles = generate_client_profiles()

    portfolios = generate_portfolios(client_profiles)

    print(portfolios.head())

    print("\nRisk category distribution:")
    print(portfolios["risk_category"].value_counts())

    current_columns = [
        f"current_{asset}"
        for asset in ASSET_CLASSES
    ]

    allocation_sums = portfolios[current_columns].sum(axis=1)

    all_valid = allocation_sums.round(6).eq(1.0).all()

    print("\nDo all current allocations sum to 100%?")
    print(all_valid)

    profiles_to_compare = client_profiles[
        ["portfolio_id", "risk_category"]
    ].rename(
        columns={"risk_category": "client_risk_category"}
    )

    portfolios_to_compare = portfolios[
        ["portfolio_id", "risk_category"]
    ].rename(
        columns={"risk_category": "portfolio_risk_category"}
    )

    comparison = profiles_to_compare.merge(
        portfolios_to_compare,
        on="portfolio_id",
        how="inner",
    )

    categories_match = (
        comparison["client_risk_category"]
        == comparison["portfolio_risk_category"]
    ).all()

    print("\nDo client and portfolio risk categories match?")
    print(categories_match)