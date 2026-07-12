import numpy as np
import pandas as pd

from config.asset_classes import ASSET_CLASSES
from config.settings import (
    NUMBER_OF_TRADING_DAYS,
    RANDOM_SEED,
)


# Expected yearly return and yearly volatility for each asset class.
MARKET_ASSUMPTIONS = {
    "domestic_equity": {
        "annual_return": 0.12,
        "annual_volatility": 0.20,
    },
    "international_equity": {
        "annual_return": 0.10,
        "annual_volatility": 0.18,
    },
    "fixed_income": {
        "annual_return": 0.06,
        "annual_volatility": 0.06,
    },
    "real_estate": {
        "annual_return": 0.08,
        "annual_volatility": 0.14,
    },
    "commodities": {
        "annual_return": 0.07,
        "annual_volatility": 0.16,
    },
    "cash": {
        "annual_return": 0.04,
        "annual_volatility": 0.01,
    },
}


def simulate_market_data(
    number_of_days: int = NUMBER_OF_TRADING_DAYS,
    seed: int = RANDOM_SEED,
    starting_price: float = 100.0,
) -> pd.DataFrame:
    """
    Generate synthetic daily prices for each asset class.

    Parameters
    ----------
    number_of_days:
        Number of trading days to simulate.

    seed:
        Random seed used to make the output reproducible.

    starting_price:
        Initial price assigned to every asset class.

    Returns
    -------
    pd.DataFrame
        Table containing dates and simulated asset-class prices.
    """

    random_generator = np.random.default_rng(seed)

    dates = pd.bdate_range(
        start="2025-01-01",
        periods=number_of_days,
    )

    market_data = pd.DataFrame({"date": dates})

    for asset in ASSET_CLASSES:
        assumptions = MARKET_ASSUMPTIONS[asset]

        annual_return = assumptions["annual_return"]
        annual_volatility = assumptions["annual_volatility"]

        daily_return = annual_return / NUMBER_OF_TRADING_DAYS

        daily_volatility = (
            annual_volatility / np.sqrt(NUMBER_OF_TRADING_DAYS)
        )

        simulated_returns = random_generator.normal(
            loc=daily_return,
            scale=daily_volatility,
            size=number_of_days,
        )

        prices = starting_price * np.cumprod(
            1 + simulated_returns
        )

        market_data[asset] = prices

    return market_data


if __name__ == "__main__":
    market_data = simulate_market_data()

    print(market_data.head())
    print()
    print(market_data.tail())
    print()
    print("Shape:", market_data.shape)
    print()
    print("Missing values:")
    print(market_data.isna().sum())