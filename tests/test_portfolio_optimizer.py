import numpy as np

from src.optimization.portfolio_optimizer import PortfolioOptimizer


def test_optimizer_returns_valid_solution() -> None:
    """
    Test that the optimizer successfully produces a valid solution.
    """

    current_weights = np.array(
        [0.50, 0.30, 0.20],
    )

    target_weights = np.array(
        [0.40, 0.35, 0.25],
    )

    covariance_matrix = np.array(
        [
            [0.040, 0.010, 0.000],
            [0.010, 0.020, 0.000],
            [0.000, 0.000, 0.001],
        ],
    )

    optimizer = PortfolioOptimizer(
        turnover_budget=0.20,
    )

    result = optimizer.optimize(
        current_weights=current_weights,
        target_weights=target_weights,
        covariance_matrix=covariance_matrix,
    )

    assert result.status in {
        "optimal",
        "optimal_inaccurate",
    }

    assert result.trade_weights is not None
    assert result.post_trade_weights is not None
    assert result.turnover is not None
def test_trades_are_cash_neutral() -> None:
    """
    Test that total buys equal total sells.
    """

    current_weights = np.array(
        [0.50, 0.30, 0.20],
    )

    target_weights = np.array(
        [0.40, 0.35, 0.25],
    )

    covariance_matrix = np.array(
        [
            [0.040, 0.010, 0.000],
            [0.010, 0.020, 0.000],
            [0.000, 0.000, 0.001],
        ],
    )

    optimizer = PortfolioOptimizer(
        turnover_budget=0.20,
    )

    result = optimizer.optimize(
        current_weights=current_weights,
        target_weights=target_weights,
        covariance_matrix=covariance_matrix,
    )

    assert result.trade_weights is not None

    assert np.isclose(
        np.sum(result.trade_weights),
        0.0,
        atol=1e-6,
    )
def test_turnover_does_not_exceed_budget() -> None:
    """
    Test that the optimizer respects the turnover budget.
    """

    current_weights = np.array(
        [0.50, 0.30, 0.20],
    )

    target_weights = np.array(
        [0.40, 0.35, 0.25],
    )

    covariance_matrix = np.array(
        [
            [0.040, 0.010, 0.000],
            [0.010, 0.020, 0.000],
            [0.000, 0.000, 0.001],
        ],
    )

    turnover_budget = 0.10

    optimizer = PortfolioOptimizer(
        turnover_budget=turnover_budget,
    )

    result = optimizer.optimize(
        current_weights=current_weights,
        target_weights=target_weights,
        covariance_matrix=covariance_matrix,
    )

    assert result.turnover is not None

    assert result.turnover <= turnover_budget + 1e-6
def test_post_trade_weights_are_non_negative() -> None:
    """
    Test that the optimizer does not create short positions.
    """

    current_weights = np.array(
        [0.50, 0.30, 0.20],
    )

    target_weights = np.array(
        [0.40, 0.35, 0.25],
    )

    covariance_matrix = np.array(
        [
            [0.040, 0.010, 0.000],
            [0.010, 0.020, 0.000],
            [0.000, 0.000, 0.001],
        ],
    )

    optimizer = PortfolioOptimizer(
        turnover_budget=0.20,
    )

    result = optimizer.optimize(
        current_weights=current_weights,
        target_weights=target_weights,
        covariance_matrix=covariance_matrix,
    )

    assert result.post_trade_weights is not None

    assert np.all(
        result.post_trade_weights >= -1e-6,
    )
def test_optimizer_reduces_tracking_error() -> None:
    """
    Test that tracking error decreases after optimization.
    """

    current_weights = np.array(
        [0.50, 0.30, 0.20],
    )

    target_weights = np.array(
        [0.40, 0.35, 0.25],
    )

    covariance_matrix = np.array(
        [
            [0.040, 0.010, 0.000],
            [0.010, 0.020, 0.000],
            [0.000, 0.000, 0.001],
        ],
    )

    optimizer = PortfolioOptimizer(
        turnover_budget=0.10,
    )

    result = optimizer.optimize(
        current_weights=current_weights,
        target_weights=target_weights,
        covariance_matrix=covariance_matrix,
    )

    assert result.tracking_error_before is not None
    assert result.tracking_error_after is not None

    assert (
        result.tracking_error_after
        <= result.tracking_error_before + 1e-10
    )
def test_optimizer_rejects_mismatched_weight_lengths() -> None:
    """
    Test that current and target weights must have the same length.
    """

    current_weights = np.array(
        [0.50, 0.30, 0.20],
    )

    target_weights = np.array(
        [0.60, 0.40],
    )

    covariance_matrix = np.eye(3)

    optimizer = PortfolioOptimizer()

    result = optimizer.optimize(
        current_weights=current_weights,
        target_weights=target_weights,
        covariance_matrix=covariance_matrix,
    )

    assert result.trade_weights is None
    assert result.post_trade_weights is None
def test_optimizer_rejects_invalid_covariance_shape() -> None:
    """
    Test that covariance matrix dimensions match the number of assets.
    """

    current_weights = np.array(
        [0.50, 0.30, 0.20],
    )

    target_weights = np.array(
        [0.40, 0.35, 0.25],
    )

    covariance_matrix = np.eye(2)

    optimizer = PortfolioOptimizer()

    result = optimizer.optimize(
        current_weights=current_weights,
        target_weights=target_weights,
        covariance_matrix=covariance_matrix,
    )

    assert result.trade_weights is None
    assert result.post_trade_weights is None