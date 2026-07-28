import numpy as np
import pandas as pd

from config.asset_classes import ASSET_CLASSES
from src.data.portfolio_generator import generate_portfolios


def _sample_client_profiles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "client_id": ["C00001", "C00002"],
            "portfolio_id": ["P00001", "P00002"],
            "risk_category": ["balanced", "conservative"],
        }
    )


def test_generate_portfolios_returns_expected_columns() -> None:
    """
    Portfolio generation should emit target and current weights.
    """

    result = generate_portfolios(
        client_profiles=_sample_client_profiles(),
    )

    expected_columns = {
        "portfolio_id",
        "risk_category",
        "drift_band",
    }

    for asset in ASSET_CLASSES:
        expected_columns.add(f"target_{asset}")
        expected_columns.add(f"current_{asset}")

    assert expected_columns.issubset(result.columns)


def test_generated_current_weights_sum_to_one() -> None:
    """
    Current allocations should remain normalized.
    """

    result = generate_portfolios(
        client_profiles=_sample_client_profiles(),
    )

    current_columns = [
        f"current_{asset}"
        for asset in ASSET_CLASSES
    ]

    assert np.allclose(
        result[current_columns].sum(axis=1),
        1.0,
    )


def test_generate_portfolios_preserves_client_mapping() -> None:
    """
    Portfolio IDs and risk categories should follow client profiles.
    """

    client_profiles = _sample_client_profiles()

    result = generate_portfolios(
        client_profiles=client_profiles,
    )

    assert list(result["portfolio_id"]) == list(
        client_profiles["portfolio_id"]
    )
    assert list(result["risk_category"]) == list(
        client_profiles["risk_category"]
    )
