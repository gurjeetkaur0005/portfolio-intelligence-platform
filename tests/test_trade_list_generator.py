import numpy as np
import pandas as pd
import pytest

from src.execution.trade_list_generator import (
    generate_trade_list,
)
@pytest.fixture
def trade_inputs():
    return {
        "asset_names": [
            "domestic_equity",
            "fixed_income",
            "cash",
        ],
        "current_weights": np.array(
            [0.50, 0.30, 0.20]
        ),
        "trade_weights": np.array(
            [-0.05, 0.03, 0.02]
        ),
        "post_trade_weights": np.array(
            [0.45, 0.33, 0.22]
        ),
    }

def test_trade_list_returns_dataframe(
    trade_inputs,
):
    result = generate_trade_list(
        **trade_inputs
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )

def test_trade_list_has_expected_columns(
    trade_inputs,
):
    result = generate_trade_list(
        **trade_inputs
    )

    expected_columns = [
        "asset",
        "action",
        "current_weight",
        "trade_weight",
        "post_trade_weight",
    ]

    assert list(result.columns) == expected_columns


def test_trade_list_has_correct_number_of_rows(
    trade_inputs,
):
    result = generate_trade_list(
        **trade_inputs
    )

    assert len(result) == len(
        trade_inputs["asset_names"]
    )


def test_trade_actions_are_classified_correctly(
    trade_inputs,
):
    inputs = trade_inputs.copy()

    inputs["trade_weights"] = np.array(
        [-0.05, 0.03, 0.00]
    )
    inputs["post_trade_weights"] = np.array(
        [0.45, 0.33, 0.20]
    )

    result = generate_trade_list(
        **inputs
    )

    assert list(result["action"]) == [
        "SELL",
        "BUY",
        "HOLD",
    ]


def test_tiny_trades_are_classified_as_hold(
    trade_inputs,
):
    inputs = trade_inputs.copy()

    inputs["trade_weights"] = np.array(
        [
            0.0000001,
            -0.0000001,
            0.00,
        ]
    )

    inputs["post_trade_weights"] = np.array(
        [
            0.5000001,
            0.2999999,
            0.20,
        ]
    )

    result = generate_trade_list(
        **inputs
    )

    assert list(result["action"]) == [
        "HOLD",
        "HOLD",
        "HOLD",
    ]

def test_custom_trade_threshold(
    trade_inputs,
):
    inputs = trade_inputs.copy()

    inputs["trade_weights"] = np.array(
        [
            0.005,
            -0.005,
            0.02,
        ]
    )

    inputs["post_trade_weights"] = np.array(
        [
            0.505,
            0.295,
            0.22,
        ]
    )

    result = generate_trade_list(
        **inputs,
        minimum_trade_threshold=0.01,
    )

    assert list(result["action"]) == [
        "HOLD",
        "HOLD",
        "BUY",
    ]


def test_mismatched_input_lengths_raise_value_error(
    trade_inputs,
):
    inputs = trade_inputs.copy()

    inputs["trade_weights"] = np.array(
        [-0.05, 0.03]
    )

    with pytest.raises(
        ValueError,
        match="All input sequences must have the same length.",
    ):
        generate_trade_list(
            **inputs
        )

