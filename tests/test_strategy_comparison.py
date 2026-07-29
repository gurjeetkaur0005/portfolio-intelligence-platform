from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.backtesting.strategy_comparison import (
    BUY_AND_HOLD_NAME,
    THRESHOLD_REBALANCING_NAME,
    compare_backtest_results,
)


@dataclass(frozen=True)
class FakeBuyAndHoldResult:
    total_return: float = 0.12
    annualized_return: float = 0.08
    volatility: float = 0.15
    sharpe_ratio: float = 0.53
    maximum_drawdown: float = -0.20


@dataclass(frozen=True)
class FakeThresholdResult:
    total_return: float = 0.14
    annualized_return: float = 0.09
    volatility: float = 0.13
    sharpe_ratio: float = 0.69
    maximum_drawdown: float = -0.15
    total_transaction_costs: float = 450.0
    total_taxes_paid: float = 1_200.0
    number_of_rebalances: int = 4


def test_compare_backtest_results_returns_both_strategies() -> None:
    result = compare_backtest_results(
        buy_and_hold_result=FakeBuyAndHoldResult(),
        threshold_rebalancing_result=FakeThresholdResult(),
    )

    assert result.buy_and_hold.strategy_name == BUY_AND_HOLD_NAME
    assert (
        result.threshold_rebalancing.strategy_name
        == THRESHOLD_REBALANCING_NAME
    )


def test_buy_and_hold_has_no_rebalancing_costs() -> None:
    result = compare_backtest_results(
        buy_and_hold_result=FakeBuyAndHoldResult(),
        threshold_rebalancing_result=FakeThresholdResult(),
    )

    assert result.buy_and_hold.transaction_costs == 0.0
    assert result.buy_and_hold.taxes_paid == 0.0
    assert result.buy_and_hold.number_of_rebalances == 0


def test_threshold_metrics_are_preserved() -> None:
    result = compare_backtest_results(
        buy_and_hold_result=FakeBuyAndHoldResult(),
        threshold_rebalancing_result=FakeThresholdResult(),
    )

    threshold_metrics = result.threshold_rebalancing

    assert threshold_metrics.total_return == pytest.approx(0.14)
    assert threshold_metrics.transaction_costs == pytest.approx(450.0)
    assert threshold_metrics.taxes_paid == pytest.approx(1_200.0)
    assert threshold_metrics.number_of_rebalances == 4


def test_total_implementation_cost_combines_costs_and_taxes() -> None:
    result = compare_backtest_results(
        buy_and_hold_result=FakeBuyAndHoldResult(),
        threshold_rebalancing_result=FakeThresholdResult(),
    )

    assert (
        result.threshold_rebalancing.total_implementation_cost
        == pytest.approx(1_650.0)
    )


def test_summary_identifies_threshold_return_advantage() -> None:
    result = compare_backtest_results(
        buy_and_hold_result=FakeBuyAndHoldResult(),
        threshold_rebalancing_result=FakeThresholdResult(),
    )

    assert (
        "Threshold Rebalancing produced higher total return"
        in result.performance_summary
    )


def test_summary_identifies_threshold_risk_advantage() -> None:
    result = compare_backtest_results(
        buy_and_hold_result=FakeBuyAndHoldResult(),
        threshold_rebalancing_result=FakeThresholdResult(),
    )

    assert (
        "Threshold Rebalancing produced lower volatility"
        in result.performance_summary
    )
    assert (
        "Threshold Rebalancing produced lower maximum drawdown"
        in result.performance_summary
    )


def test_summary_includes_costs_and_rebalances() -> None:
    result = compare_backtest_results(
        buy_and_hold_result=FakeBuyAndHoldResult(),
        threshold_rebalancing_result=FakeThresholdResult(),
    )

    assert "4 rebalances" in result.performance_summary
    assert "$450.00" in result.performance_summary
    assert "$1,200.00" in result.performance_summary


def test_negative_volatility_raises_value_error() -> None:
    buy_and_hold_result = FakeBuyAndHoldResult(
        volatility=-0.15,
    )

    with pytest.raises(
        ValueError,
        match="volatility cannot be negative",
    ):
        compare_backtest_results(
            buy_and_hold_result=buy_and_hold_result,
            threshold_rebalancing_result=FakeThresholdResult(),
        )


def test_negative_transaction_cost_raises_value_error() -> None:
    threshold_result = FakeThresholdResult(
        total_transaction_costs=-100.0,
    )

    with pytest.raises(
        ValueError,
        match="transaction_costs cannot be negative",
    ):
        compare_backtest_results(
            buy_and_hold_result=FakeBuyAndHoldResult(),
            threshold_rebalancing_result=threshold_result,
        )


def test_negative_taxes_raise_value_error() -> None:
    threshold_result = FakeThresholdResult(
        total_taxes_paid=-100.0,
    )

    with pytest.raises(
        ValueError,
        match="taxes_paid cannot be negative",
    ):
        compare_backtest_results(
            buy_and_hold_result=FakeBuyAndHoldResult(),
            threshold_rebalancing_result=threshold_result,
        )


def test_negative_rebalance_count_raises_value_error() -> None:
    threshold_result = FakeThresholdResult(
        number_of_rebalances=-1,
    )

    with pytest.raises(
        ValueError,
        match="number_of_rebalances cannot be negative",
    ):
        compare_backtest_results(
            buy_and_hold_result=FakeBuyAndHoldResult(),
            threshold_rebalancing_result=threshold_result,
        )


def test_non_integer_rebalance_count_raises_value_error() -> None:
    threshold_result = FakeThresholdResult(
        number_of_rebalances=4,  # type: ignore[arg-type]
    )

    object.__setattr__(
        threshold_result,
        "number_of_rebalances",
        4.5,
    )

    with pytest.raises(
        ValueError,
        match="number_of_rebalances must be an integer",
    ):
        compare_backtest_results(
            buy_and_hold_result=FakeBuyAndHoldResult(),
            threshold_rebalancing_result=threshold_result,
        )


def test_equal_metrics_generate_similar_summary() -> None:
    buy_and_hold_result = FakeBuyAndHoldResult(
        total_return=0.14,
        volatility=0.13,
        sharpe_ratio=0.69,
        maximum_drawdown=-0.15,
    )

    result = compare_backtest_results(
        buy_and_hold_result=buy_and_hold_result,
        threshold_rebalancing_result=FakeThresholdResult(),
    )

    assert (
        "Both strategies produced similar total return"
        in result.performance_summary
    )
    assert (
        "Both strategies produced similar volatility"
        in result.performance_summary
    )