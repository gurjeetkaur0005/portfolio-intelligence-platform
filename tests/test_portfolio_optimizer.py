import numpy as np
import pytest

from src.optimization.portfolio_optimizer import PortfolioOptimizer


@pytest.fixture
def optimizer_inputs():
    """
    Common inputs shared by all optimizer tests.
    """
    return {
        "current_weights": np.array([0.50, 0.30, 0.20]),
        "target_weights": np.array([0.40, 0.35, 0.25]),
        "covariance_matrix": np.array(
            [
                [0.040, 0.010, 0.000],
                [0.010, 0.020, 0.000],
                [0.000, 0.000, 0.001],
            ],
        ),
    }


def test_optimizer_returns_valid_solution(optimizer_inputs) -> None:
    """
    Test that the optimizer successfully produces a valid solution.
    """

    optimizer = PortfolioOptimizer(turnover_budget=0.20)

    result = optimizer.optimize(**optimizer_inputs)

    assert result.status in {
        "optimal",
        "optimal_inaccurate",
    }

    assert result.trade_weights is not None
    assert result.post_trade_weights is not None
    assert result.turnover is not None


def test_trades_are_cash_neutral(optimizer_inputs) -> None:
    """
    Test that buys equal sells.
    """

    optimizer = PortfolioOptimizer(turnover_budget=0.20)

    result = optimizer.optimize(**optimizer_inputs)

    assert result.trade_weights is not None

    assert np.isclose(
        np.sum(result.trade_weights),
        0.0,
        atol=1e-6,
    )


def test_turnover_does_not_exceed_budget(optimizer_inputs) -> None:
    """
    Test that turnover remains within the specified budget.
    """

    turnover_budget = 0.10

    optimizer = PortfolioOptimizer(
        turnover_budget=turnover_budget,
    )

    result = optimizer.optimize(**optimizer_inputs)

    assert result.turnover is not None

    assert result.turnover <= turnover_budget + 1e-6


def test_post_trade_weights_are_non_negative(
    optimizer_inputs,
) -> None:
    """
    Test that the optimizer does not create short positions.
    """

    optimizer = PortfolioOptimizer(turnover_budget=0.20)

    result = optimizer.optimize(**optimizer_inputs)

    assert result.post_trade_weights is not None

    assert np.all(
        result.post_trade_weights >= -1e-6,
    )


def test_optimizer_reduces_tracking_error(
    optimizer_inputs,
) -> None:
    """
    Test that optimization reduces tracking error.
    """

    optimizer = PortfolioOptimizer(turnover_budget=0.10)

    result = optimizer.optimize(**optimizer_inputs)

    assert result.tracking_error_before is not None
    assert result.tracking_error_after is not None

    assert (
        result.tracking_error_after
        <= result.tracking_error_before + 1e-10
    )


def test_optimizer_rejects_mismatched_weight_lengths() -> None:
    """
    Test that current and target weights must have equal length.
    """

    optimizer = PortfolioOptimizer()

    result = optimizer.optimize(
        current_weights=np.array([0.50, 0.30, 0.20]),
        target_weights=np.array([0.60, 0.40]),
        covariance_matrix=np.eye(3),
    )

    assert result.trade_weights is None
    assert result.post_trade_weights is None


def test_optimizer_rejects_invalid_covariance_shape() -> None:
    """
    Test that covariance matrix dimensions match the asset count.
    """

    optimizer = PortfolioOptimizer()

    result = optimizer.optimize(
        current_weights=np.array([0.50, 0.30, 0.20]),
        target_weights=np.array([0.40, 0.35, 0.25]),
        covariance_matrix=np.eye(2),
    )

    assert result.trade_weights is None
    assert result.post_trade_weights is None