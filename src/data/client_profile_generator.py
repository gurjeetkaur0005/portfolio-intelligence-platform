import numpy as np
import pandas as pd

from config.risk_categories import RISK_CATEGORIES
from config.settings import (
    NUMBER_OF_PORTFOLIOS,
    RANDOM_SEED,
)


TAX_BRACKETS = [0.00, 0.20, 0.30]

TAX_BRACKET_PROBABILITIES = [0.15, 0.35, 0.50]


def generate_client_profiles(
    number_of_clients: int = NUMBER_OF_PORTFOLIOS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate synthetic investor profiles.

    Each profile contains client-specific information that may affect
    portfolio rebalancing decisions.
    """

    random_generator = np.random.default_rng(seed)

    risk_category_names = list(RISK_CATEGORIES.keys())

    risk_category_probabilities = [
        RISK_CATEGORIES[category]["weight"]
        for category in risk_category_names
    ]

    rows = []

    for client_number in range(1, number_of_clients + 1):
        risk_category = random_generator.choice(
            risk_category_names,
            p=risk_category_probabilities,
        )

        tax_bracket = random_generator.choice(
            TAX_BRACKETS,
            p=TAX_BRACKET_PROBABILITIES,
        )

        age = int(
            random_generator.integers(
                low=21,
                high=76,
            )
        )

        investment_horizon_years = int(
            random_generator.integers(
                low=1,
                high=31,
            )
        )

        annual_income = float(
            random_generator.integers(
                low=400_000,
                high=5_000_001,
            )
        )

        monthly_sip = float(
            random_generator.choice(
                [0, 5_000, 10_000, 15_000, 25_000, 50_000],
                p=[0.10, 0.20, 0.25, 0.20, 0.15, 0.10],
            )
        )

        planned_withdrawal = float(
            random_generator.choice(
                [0, 50_000, 100_000, 250_000, 500_000],
                p=[0.80, 0.08, 0.06, 0.04, 0.02],
            )
        )

        restricted_security_count = int(
            random_generator.choice(
                [0, 1, 2, 3, 4, 5],
                p=[0.60, 0.18, 0.10, 0.06, 0.04, 0.02],
            )
        )

        esg_preference = bool(
            random_generator.choice(
                [True, False],
                p=[0.25, 0.75],
            )
        )

        prior_approval_required = bool(
            random_generator.choice(
                [True, False],
                p=[0.10, 0.90],
            )
        )

        row = {
            "client_id": f"C{client_number:05d}",
            "portfolio_id": f"P{client_number:05d}",
            "risk_category": risk_category,
            "age": age,
            "investment_horizon_years": investment_horizon_years,
            "annual_income": annual_income,
            "tax_bracket": tax_bracket,
            "monthly_sip": monthly_sip,
            "planned_withdrawal": planned_withdrawal,
            "restricted_security_count": restricted_security_count,
            "esg_preference": esg_preference,
            "prior_approval_required": prior_approval_required,
        }

        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    client_profiles = generate_client_profiles()

    print(client_profiles.head())
    print()
    print("Shape:", client_profiles.shape)
    print()
    print("Risk category distribution:")
    print(client_profiles["risk_category"].value_counts())
    print()
    print("Tax bracket distribution:")
    print(client_profiles["tax_bracket"].value_counts())
    print()
    print("Missing values:")
    print(client_profiles.isna().sum())