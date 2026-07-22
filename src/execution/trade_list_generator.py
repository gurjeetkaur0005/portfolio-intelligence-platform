from typing import Sequence
import numpy as np
import pandas as pd


def generate_trade_list(
    asset_names: Sequence[str],
    current_weights: np.ndarray,
    trade_weights: np.ndarray,
    post_trade_weights: np.ndarray,
    minimum_trade_threshold: float = 1e-6,
) -> pd.DataFrame:
    """
    Convert optimized portfolio weights into structured trade instructions.
    """
    if not (
        len(asset_names)
        == len(current_weights)
        == len(trade_weights)
        == len(post_trade_weights)
    ):
        raise ValueError(
            "All input sequences must have the same length."
        )

    trade_records = []

    for (
        asset_name,
        current_weight,
        trade_weight,
        post_trade_weight,
    ) in zip(
        asset_names,
        current_weights,
        trade_weights,
        post_trade_weights,
    ):
        if trade_weight > minimum_trade_threshold:
            action = "BUY"
        elif trade_weight < -minimum_trade_threshold:
            action = "SELL"
        else:
            action = "HOLD"
            trade_weight = 0.0
            post_trade_weight = current_weight

        trade_records.append(
            {
                "asset": asset_name,
                "action": action,
                "current_weight": float(current_weight),
                "trade_weight": float(trade_weight),
                "post_trade_weight": float(post_trade_weight),
            }
        )

    return pd.DataFrame(
        trade_records
    )
