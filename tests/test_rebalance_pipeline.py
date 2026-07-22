import sys
import types

from src.optimization.optimization_models import OptimizationResult

from src.pipeline.rebalance_pipeline import run_rebalance_pipeline


class FakePortfolioOptimizer:
    """Optimizer stand-in used to keep this test dependency-light."""

    def optimize(
        self,
        current_weights,
        target_weights,
        covariance_matrix,
    ) -> OptimizationResult:
        trade_weights = target_weights - current_weights

        return OptimizationResult(
            status="optimal",
            trade_weights=trade_weights,
            post_trade_weights=target_weights,
            tracking_error_before=0.0,
            tracking_error_after=0.0,
            turnover=float(abs(trade_weights).sum()),
            objective_value=0.0,
            message="Fake optimization completed successfully.",
        )


def test_rebalance_pipeline_produces_final_trade_outputs(
    monkeypatch,
) -> None:
    fake_optimizer_module = types.SimpleNamespace(
        PortfolioOptimizer=FakePortfolioOptimizer
    )
    monkeypatch.setitem(
        sys.modules,
        "src.optimization.portfolio_optimizer",
        fake_optimizer_module,
    )

    result = run_rebalance_pipeline(
        number_of_clients=1,
        portfolio_value=1_000_000,
    )

    expected_columns = {
        "trade_value",
        "transaction_cost",
        "estimated_tax_liability",
        "client_explanation",
        "advisor_explanation",
        "compliance_explanation",
    }

    assert not result.empty
    assert expected_columns.issubset(result.columns)
    assert result["trade_value"].notna().all()
    assert result["transaction_cost"].notna().all()
    assert result["estimated_tax_liability"].notna().all()
    assert result["client_explanation"].str.len().gt(0).all()
    assert result["advisor_explanation"].str.len().gt(0).all()
    assert result["compliance_explanation"].str.len().gt(0).all()
