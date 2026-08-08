import sys
import types

import pandas as pd

from src.data.client_profile_generator import generate_client_profiles
from src.data.portfolio_generator import generate_portfolios
from src.optimization.optimization_models import OptimizationResult

from src.pipeline.rebalance_pipeline import (
    run_rebalance_pipeline,
    run_rebalance_pipeline_for_inputs,
)


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
        "audit_id",
        "audit_timestamp",
        "trade_value",
        "transaction_cost",
        "estimated_tax_liability",
        "client_explanation",
        "advisor_explanation",
        "compliance_explanation",
        "approval_required",
        "approval_status",
        "approval_reason",
        "final_trigger_type",
        "final_priority",
    }

    assert not result.empty
    assert expected_columns.issubset(result.columns)
    assert result["trade_value"].notna().all()
    assert result["transaction_cost"].notna().all()
    assert result["estimated_tax_liability"].notna().all()
    assert result["client_explanation"].str.len().gt(0).all()
    assert result["advisor_explanation"].str.len().gt(0).all()
    assert result["compliance_explanation"].str.len().gt(0).all()
    assert result["approval_status"].notna().all()
    assert result["audit_id"].str.startswith("AUD").all()


def test_rebalance_pipeline_preserves_money_conservation(
    monkeypatch,
) -> None:
    """Pipeline trades remain self-financing before costs and taxes."""

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

    assert not result.empty

    for _, portfolio_trades in result.groupby("portfolio_id"):
        assert abs(portfolio_trades["trade_weight"].sum()) < 1e-9
        assert abs(portfolio_trades["trade_value"].sum()) < 1e-6


def test_input_pipeline_matches_synthetic_pipeline_outputs(
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

    client_profiles = generate_client_profiles(
        number_of_clients=1
    )
    portfolios = generate_portfolios(
        client_profiles=client_profiles
    )

    synthetic_result = run_rebalance_pipeline(
        number_of_clients=1,
        portfolio_value=1_000_000,
    ).drop(columns=["audit_timestamp"])
    input_result = run_rebalance_pipeline_for_inputs(
        client_profiles=client_profiles,
        portfolios=portfolios,
        portfolio_value=1_000_000,
    ).drop(columns=["audit_timestamp"])

    pd.testing.assert_frame_equal(
        input_result,
        synthetic_result,
    )


def test_input_pipeline_does_not_call_synthetic_generators(
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

    client_profiles = generate_client_profiles(
        number_of_clients=1
    )
    portfolios = generate_portfolios(
        client_profiles=client_profiles
    )

    def fail_generator(*args, **kwargs):
        raise AssertionError(
            "Synthetic generator should not be called."
        )

    monkeypatch.setattr(
        "src.pipeline.rebalance_pipeline.generate_client_profiles",
        fail_generator,
    )
    monkeypatch.setattr(
        "src.pipeline.rebalance_pipeline.generate_portfolios",
        fail_generator,
    )

    result = run_rebalance_pipeline_for_inputs(
        client_profiles=client_profiles,
        portfolios=portfolios,
        portfolio_value=1_000_000,
    )

    assert not result.empty
