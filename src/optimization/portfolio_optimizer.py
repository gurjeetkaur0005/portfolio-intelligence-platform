import cvxpy as cp
import numpy as np

from src.optimization.optimization_models import OptimizationResult


NUMERICAL_TOLERANCE = 1e-6


class PortfolioOptimizer:
    """
    Optimize portfolio trades while respecting portfolio constraints.
    """

    def __init__(
        self,
        turnover_budget: float = 0.10,
    ) -> None:
        if not np.isfinite(turnover_budget):
            raise ValueError(
                "turnover_budget must be finite."
            )

        self.turnover_budget = turnover_budget

    def optimize(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> OptimizationResult:
        """
        Optimize portfolio trades.
        """

        input_error = _validate_optimizer_inputs(
            current_weights=current_weights,
            target_weights=target_weights,
            covariance_matrix=covariance_matrix,
            turnover_budget=self.turnover_budget,
        )

        if input_error is not None:
            return _error_result(input_error)

        if len(current_weights) != len(target_weights):
            return _error_result(
                (
                    "Current and target weights must have "
                    "the same length."
                )
            )

        number_of_assets = len(current_weights)

        if covariance_matrix.shape != (
            number_of_assets,
            number_of_assets,
        ):
            return _error_result(
                (
                    "Covariance matrix must have shape "
                    f"({number_of_assets}, {number_of_assets})."
                )
            )

        trade_weights = cp.Variable(number_of_assets)

        post_trade_weights = (
            current_weights + trade_weights
        )

        post_trade_drift = (
            post_trade_weights - target_weights
        )

        tracking_error = cp.quad_form(
            post_trade_drift,
            covariance_matrix,
        )

        objective = cp.Minimize(tracking_error)

        constraints = [
            cp.sum(trade_weights) == 0,
            post_trade_weights >= 0,
            cp.norm1(trade_weights) <= self.turnover_budget,
        ]

        problem = cp.Problem(
            objective,
            constraints,
        )

        problem.solve()

        if problem.status not in {
            cp.OPTIMAL,
            cp.OPTIMAL_INACCURATE,
        }:
            return _error_result(
                "Portfolio optimization failed.",
                status=str(problem.status),
            )

        optimized_trade_weights = _validated_solver_vector(
            value=trade_weights.value,
            expected_length=number_of_assets,
            field_name="trade_weights",
        )

        if isinstance(optimized_trade_weights, str):
            return _error_result(optimized_trade_weights)

        optimized_post_trade_weights = (
            current_weights + optimized_trade_weights
        )

        post_trade_error = _validate_post_trade_solution(
            trade_weights=optimized_trade_weights,
            post_trade_weights=optimized_post_trade_weights,
        )

        if post_trade_error is not None:
            return _error_result(post_trade_error)

        current_drift = (
            current_weights - target_weights
        )

        tracking_error_before = float(
            current_drift.T
            @ covariance_matrix
            @ current_drift
        )

        optimized_drift = (
            optimized_post_trade_weights - target_weights
        )

        tracking_error_after = float(
            optimized_drift.T
            @ covariance_matrix
            @ optimized_drift
        )

        objective_value = float(problem.value)

        if not np.isfinite(objective_value):
            return _error_result(
                "Solver objective value must be finite."
            )

        turnover = float(
            np.sum(np.abs(optimized_trade_weights))
        )

        return OptimizationResult(
            status=problem.status,
            trade_weights=optimized_trade_weights,
            post_trade_weights=optimized_post_trade_weights,
            tracking_error_before=tracking_error_before,
            tracking_error_after=tracking_error_after,
            turnover=turnover,
            objective_value=objective_value,
            message="Portfolio optimization completed successfully.",
        )


def _validate_optimizer_inputs(
    *,
    current_weights: np.ndarray,
    target_weights: np.ndarray,
    covariance_matrix: np.ndarray,
    turnover_budget: float,
) -> str | None:
    """Return an error message when optimizer inputs are invalid."""

    if not np.all(np.isfinite(current_weights)):
        return "Current weights must contain only finite values."

    if not np.all(np.isfinite(target_weights)):
        return "Target weights must contain only finite values."

    if not np.all(np.isfinite(covariance_matrix)):
        return "Covariance matrix must contain only finite values."

    if not np.isfinite(turnover_budget):
        return "turnover_budget must be finite."

    return None


def _validated_solver_vector(
    *,
    value: object,
    expected_length: int,
    field_name: str,
) -> np.ndarray | str:
    """Return a validated finite solver vector or an error message."""

    if value is None:
        return f"Solver output {field_name} must not be None."

    vector = np.asarray(
        value,
        dtype=float,
    )

    if vector.shape != (expected_length,):
        return (
            f"Solver output {field_name} must have shape "
            f"({expected_length},)."
        )

    if not np.all(np.isfinite(vector)):
        return (
            f"Solver output {field_name} must contain only "
            "finite values."
        )

    return vector


def _validate_post_trade_solution(
    *,
    trade_weights: np.ndarray,
    post_trade_weights: np.ndarray,
) -> str | None:
    """Validate optimizer output without rejecting tolerance noise."""

    if not np.all(np.isfinite(post_trade_weights)):
        return "Post-trade weights must contain only finite values."

    if np.any(post_trade_weights < -NUMERICAL_TOLERANCE):
        return "Post-trade weights must not be materially negative."

    if not np.isclose(
        np.sum(trade_weights),
        0.0,
        atol=NUMERICAL_TOLERANCE,
    ):
        return "Trade weights must sum approximately to zero."

    if not np.isclose(
        np.sum(post_trade_weights),
        1.0,
        atol=NUMERICAL_TOLERANCE,
    ):
        return "Post-trade weights must sum approximately to 1.0."

    return None


def _error_result(
    message: str,
    *,
    status: str = "error",
) -> OptimizationResult:
    """Build a failed optimization result."""

    return OptimizationResult(
        status=status,
        trade_weights=None,
        post_trade_weights=None,
        tracking_error_before=None,
        tracking_error_after=None,
        turnover=None,
        objective_value=None,
        message=message,
    )
