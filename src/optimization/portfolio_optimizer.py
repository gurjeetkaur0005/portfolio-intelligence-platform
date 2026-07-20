import cvxpy as cp
import numpy as np

from src.optimization.optimization_models import OptimizationResult


class PortfolioOptimizer:
    """
    Optimize portfolio trades while respecting portfolio constraints.
    """

    def __init__(
        self,
        turnover_budget: float = 0.10,
    ) -> None:
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

        if len(current_weights) != len(target_weights):
            return OptimizationResult(
                status="error",
                trade_weights=None,
                post_trade_weights=None,
                tracking_error_before=None,
                tracking_error_after=None,
                turnover=None,
                objective_value=None,
                message=(
                    "Current and target weights must have "
                    "the same length."
                ),
            )

        number_of_assets = len(current_weights)

        if covariance_matrix.shape != (
            number_of_assets,
            number_of_assets,
        ):
            return OptimizationResult(
                status="error",
                trade_weights=None,
                post_trade_weights=None,
                tracking_error_before=None,
                tracking_error_after=None,
                turnover=None,
                objective_value=None,
                message=(
                    "Covariance matrix must have shape "
                    f"({number_of_assets}, {number_of_assets})."
                ),
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
            return OptimizationResult(
                status=problem.status,
                trade_weights=None,
                post_trade_weights=None,
                tracking_error_before=None,
                tracking_error_after=None,
                turnover=None,
                objective_value=None,
                message="Portfolio optimization failed.",
            )

        optimized_trade_weights = np.asarray(
            trade_weights.value,
            dtype=float,
        )

        optimized_post_trade_weights = np.asarray(
            post_trade_weights.value,
            dtype=float,
        )

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
            objective_value=float(problem.value),
            message="Portfolio optimization completed successfully.",
        )