import numpy as np
import pytest
import cvxpy as cp

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


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_optimizer_rejects_non_finite_current_weights(
    optimizer_inputs,
    invalid_value: float,
) -> None:
    """Current weights must not contain NaN or infinity."""

    optimizer_inputs["current_weights"] = np.array(
        [0.50, invalid_value, 0.20]
    )

    result = PortfolioOptimizer().optimize(
        **optimizer_inputs
    )

    assert result.status == "error"
    assert result.trade_weights is None
    assert "Current weights" in result.message


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_optimizer_rejects_non_finite_target_weights(
    optimizer_inputs,
    invalid_value: float,
) -> None:
    """Target weights must not contain NaN or infinity."""

    optimizer_inputs["target_weights"] = np.array(
        [0.40, invalid_value, 0.25]
    )

    result = PortfolioOptimizer().optimize(
        **optimizer_inputs
    )

    assert result.status == "error"
    assert result.trade_weights is None
    assert "Target weights" in result.message


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_optimizer_rejects_non_finite_covariance_matrix(
    optimizer_inputs,
    invalid_value: float,
) -> None:
    """Covariance values must not contain NaN or infinity."""

    covariance_matrix = optimizer_inputs["covariance_matrix"].copy()
    covariance_matrix[1, 1] = invalid_value
    optimizer_inputs["covariance_matrix"] = covariance_matrix

    result = PortfolioOptimizer().optimize(
        **optimizer_inputs
    )

    assert result.status == "error"
    assert result.trade_weights is None
    assert "Covariance matrix" in result.message


@pytest.mark.parametrize(
    "invalid_budget",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_optimizer_rejects_non_finite_turnover_budget(
    invalid_budget: float,
) -> None:
    """Turnover budget must be finite."""

    with pytest.raises(
        ValueError,
        match="turnover_budget must be finite",
    ):
        PortfolioOptimizer(turnover_budget=invalid_budget)


def test_solver_none_output_is_handled_safely(
    optimizer_inputs,
    monkeypatch,
) -> None:
    """Successful solver status with no vector is rejected."""

    _install_fake_solver(
        monkeypatch,
        solver_value=None,
    )

    result = PortfolioOptimizer().optimize(
        **optimizer_inputs
    )

    assert result.status == "error"
    assert result.trade_weights is None
    assert "must not be None" in result.message


@pytest.mark.parametrize(
    "solver_value",
    [
        np.array([np.nan, 0.0, 0.0]),
        np.array([np.inf, 0.0, -np.inf]),
    ],
)
def test_solver_non_finite_output_is_handled_safely(
    optimizer_inputs,
    monkeypatch,
    solver_value: np.ndarray,
) -> None:
    """Solver output vectors must be finite."""

    _install_fake_solver(
        monkeypatch,
        solver_value=solver_value,
    )

    result = PortfolioOptimizer().optimize(
        **optimizer_inputs
    )

    assert result.status == "error"
    assert result.trade_weights is None
    assert "Solver output trade_weights" in result.message


def test_tiny_negative_post_trade_noise_is_allowed(
    monkeypatch,
) -> None:
    """Harmless solver tolerance noise is not treated as a short."""

    _install_fake_solver(
        monkeypatch,
        solver_value=np.array([-1e-12, 0.0, 1e-12]),
    )

    result = PortfolioOptimizer().optimize(
        current_weights=np.array([0.0, 0.50, 0.50]),
        target_weights=np.array([0.0, 0.50, 0.50]),
        covariance_matrix=np.eye(3),
    )

    assert result.status == "optimal"
    assert result.post_trade_weights is not None
    assert result.post_trade_weights[0] == pytest.approx(-1e-12)


def _install_fake_solver(
    monkeypatch: pytest.MonkeyPatch,
    *,
    solver_value: np.ndarray | None,
) -> None:
    """Install a fake CVXPY solve result for output validation tests."""

    def fake_solve(
        problem: cp.Problem,
    ) -> float:
        variables = problem.variables()
        variables[0]._value = solver_value
        problem._status = cp.OPTIMAL
        problem._value = 0.0

        return 0.0

    monkeypatch.setattr(
        cp.Problem,
        "solve",
        fake_solve,
    )
