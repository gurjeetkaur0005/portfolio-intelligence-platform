import numpy as np
import pandas as pd
from config.asset_classes import ASSET_CLASSES
from config.risk_categories import RISK_CATEGORIES
from config.settings import (
    NUMBER_OF_PORTFOLIOS,
    RANDOM_SEED,
)

# Portfolio templates for each investor risk profile.
# - target: ideal asset allocation
# - drift_band: expected deviation from the target due to market movement
# - weight: probability of generating this type of portfolio



def generate_portfolios(
    number_of_portfolios: int=  NUMBER_OF_PORTFOLIOS,
    seed: int= RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate synthetic investment portfolios.

    Each portfolio contains:
    - Portfolio ID
    - Risk category
    - Target asset allocation
    - Current asset allocation after simulated market movement
    - Drift band
    """

    # Create a reproducible random number generator.
    # Using the same seed always produces the same portfolios.
    random_generator = np.random.default_rng(seed)

    # Extract all available risk category names.
    category_names = list(RISK_CATEGORIES.keys())

    # Probability of selecting each risk category.
    category_probabilities = [
        RISK_CATEGORIES[category]["weight"]
        for category in category_names
    ]

    # Store every generated portfolio here.
    rows = []

    # Generate portfolios one by one.
    for portfolio_number in range(1, number_of_portfolios + 1):

        # Randomly choose a risk category using the specified probabilities.
        category = random_generator.choice(
            category_names,
            p=category_probabilities,
        )

        # Retrieve configuration for the selected category.
        category_config = RISK_CATEGORIES[category]

        # Convert target allocation into a NumPy array for mathematical operations.
        target = np.array(
            category_config["target"],
            dtype=float,
        )

        # Maximum expected drift for this portfolio type.
        drift_band = category_config["drift_band"]

        # Simulate market movement by adding normally distributed random noise.
        noise = random_generator.normal(
            loc=0,
            scale=drift_band * 0.8,
            size=len(ASSET_CLASSES),
        )

        # Current allocation after market movement.
        current = target + noise

        # Remove negative allocations (asset weights cannot be negative).
        current = np.clip(current, 0, None)

        # Normalize allocations so the portfolio always sums to 100%.
        current = current / current.sum()

        # Store basic portfolio information.
        row = {
            "portfolio_id": f"P{portfolio_number:05d}",
            "risk_category": category,
            "drift_band": drift_band,
        }

        # Store target and current allocation for every asset class.
        for index, asset in enumerate(ASSET_CLASSES):
            row[f"target_{asset}"] = target[index]
            row[f"current_{asset}"] = current[index]

        # Save this portfolio.
        rows.append(row)

    # Convert all portfolio dictionaries into a DataFrame.
    return pd.DataFrame(rows)


if __name__ == "__main__":

    # Generate the synthetic dataset.
    portfolios = generate_portfolios()

    # Display the first five portfolios.
    print(portfolios.head())

    # Verify that the generated portfolio distribution roughly follows
    # the specified category probabilities.
    print("\nRisk category distribution:")
    print(portfolios["risk_category"].value_counts())

    # List of columns containing current asset allocations.
    current_columns = [
        f"current_{asset}"
        for asset in ASSET_CLASSES
    ]

    # Calculate total allocation for each portfolio.
    allocation_sums = portfolios[current_columns].sum(axis=1)

    # Check that every portfolio sums to exactly 100%.
    all_valid = allocation_sums.round(6).eq(1.0).all()

    print("\nDo all current allocations sum to 100%?")
    print(all_valid)