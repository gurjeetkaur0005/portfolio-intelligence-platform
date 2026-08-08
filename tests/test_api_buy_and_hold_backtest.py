from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd
from fastapi.testclient import TestClient

from src.api.dependencies import get_buy_and_hold_backtest_runner
from src.api.main import app
from src.backtesting.backtest_engine import BacktestResult


client = TestClient(app)


def _valid_payload() -> dict[str, Any]:
    """Return a valid Buy & Hold backtest request payload."""

    return {
        "asset_names": [
            "domestic_equity",
            "fixed_income",
        ],
        "market_returns": [
            [
                0.01,
                0.002,
            ],
            [
                -0.005,
                0.001,
            ],
        ],
        "initial_weights": [
            0.6,
            0.4,
        ],
    }


def _backtest_result(
    portfolio_history: pd.DataFrame | None = None,
) -> BacktestResult:
    """Return a deterministic fake backtest result."""

    if portfolio_history is None:
        portfolio_history = pd.DataFrame(
            [
                {
                    "date": "initial",
                    "portfolio_value": 100_000.0,
                    "domestic_equity_weight": 0.6,
                    "fixed_income_weight": 0.4,
                },
                {
                    "date": 0,
                    "portfolio_value": 100_680.0,
                    "domestic_equity_weight": 0.604,
                    "fixed_income_weight": 0.396,
                },
            ]
        )

    return BacktestResult(
        portfolio_history=portfolio_history,
        total_return=0.0068,
        annualized_return=0.12,
        volatility=0.08,
        sharpe_ratio=1.5,
        maximum_drawdown=-0.02,
        drawdown_history=pd.DataFrame(
            [
                {
                    "period": 0,
                    "date": "initial",
                    "drawdown": 0.0,
                },
                {
                    "period": 1,
                    "date": 0,
                    "drawdown": -0.02,
                },
            ]
        ),
    )


class CapturingBacktestRunner:
    """Capture API-adapted inputs and return a fixed result."""

    def __init__(
        self,
        result: BacktestResult | None = None,
    ) -> None:
        self.result = (
            result
            if result is not None
            else _backtest_result()
        )
        self.call_count = 0
        self.initial_weights: list[float] | None = None
        self.market_returns: pd.DataFrame | None = None
        self.initial_portfolio_value: float | None = None
        self.risk_free_rate: float | None = None
        self.periods_per_year: int | None = None

    def __call__(
        self,
        initial_weights,
        market_returns,
        initial_portfolio_value=100_000.0,
        risk_free_rate=0.0,
        periods_per_year=252,
    ) -> BacktestResult:
        self.call_count += 1
        self.initial_weights = list(initial_weights)
        self.market_returns = market_returns
        self.initial_portfolio_value = initial_portfolio_value
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

        return self.result


class ValueErrorBacktestRunner:
    """Raise a domain ValueError from the backtest engine."""

    def __call__(self, *args, **kwargs) -> BacktestResult:
        raise ValueError("Domain validation failed.")


class TypeErrorBacktestRunner:
    """Raise a domain TypeError from the backtest engine."""

    def __call__(self, *args, **kwargs) -> BacktestResult:
        raise TypeError("Domain type validation failed.")


def _override_runner(runner: object) -> None:
    app.dependency_overrides[
        get_buy_and_hold_backtest_runner
    ] = lambda: runner


def test_buy_and_hold_backtest_returns_success() -> None:
    runner = CapturingBacktestRunner()
    _override_runner(runner)

    try:
        response = client.post(
            "/backtests/buy-and-hold",
            json=_valid_payload(),
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert runner.call_count == 1


def test_strategy_name_is_buy_and_hold() -> None:
    _override_runner(CapturingBacktestRunner())

    try:
        response = client.post(
            "/backtests/buy-and-hold",
            json=_valid_payload(),
        )
    finally:
        app.dependency_overrides.clear()

    assert response.json()["strategy_name"] == "Buy & Hold"


def test_metrics_are_serialized_correctly() -> None:
    _override_runner(CapturingBacktestRunner())

    try:
        response = client.post(
            "/backtests/buy-and-hold",
            json=_valid_payload(),
        )
    finally:
        app.dependency_overrides.clear()

    assert response.json()["metrics"] == {
        "total_return": 0.0068,
        "annualized_return": 0.12,
        "volatility": 0.08,
        "sharpe_ratio": 1.5,
        "maximum_drawdown": -0.02,
    }


def test_portfolio_history_is_serialized_correctly() -> None:
    _override_runner(CapturingBacktestRunner())

    try:
        response = client.post(
            "/backtests/buy-and-hold",
            json=_valid_payload(),
        )
    finally:
        app.dependency_overrides.clear()

    assert response.json()["portfolio_history"][0] == {
        "date": "initial",
        "portfolio_value": 100_000.0,
        "domestic_equity_weight": 0.6,
        "fixed_income_weight": 0.4,
    }


def test_history_record_count_is_correct() -> None:
    _override_runner(CapturingBacktestRunner())

    try:
        response = client.post(
            "/backtests/buy-and-hold",
            json=_valid_payload(),
        )
    finally:
        app.dependency_overrides.clear()

    assert response.json()["history_record_count"] == 2


def test_drawdown_history_is_serialized() -> None:
    """Buy & Hold response exposes backend-provided drawdown history."""

    _override_runner(CapturingBacktestRunner())

    response = client.post(
        "/backtests/buy-and-hold",
        json=_valid_payload(),
    )

    body = response.json()

    assert response.status_code == 200
    assert body["metrics"]["maximum_drawdown"] == -0.02
    assert body["drawdown_history"] == [
        {
            "period": 0,
            "date": "initial",
            "drawdown": 0.0,
        },
        {
            "period": 1,
            "date": 0,
            "drawdown": -0.02,
        },
    ]


def test_request_defaults_are_passed_to_runner() -> None:
    runner = CapturingBacktestRunner()
    _override_runner(runner)

    try:
        response = client.post(
            "/backtests/buy-and-hold",
            json=_valid_payload(),
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert runner.initial_portfolio_value == 100_000.0
    assert runner.risk_free_rate == 0.0
    assert runner.periods_per_year == 252


def test_custom_values_are_passed_to_runner() -> None:
    runner = CapturingBacktestRunner()
    payload = _valid_payload()
    payload["initial_portfolio_value"] = 250_000.0
    payload["risk_free_rate"] = 0.03
    payload["periods_per_year"] = 12
    _override_runner(runner)

    try:
        response = client.post(
            "/backtests/buy-and-hold",
            json=payload,
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert runner.initial_portfolio_value == 250_000.0
    assert runner.risk_free_rate == 0.03
    assert runner.periods_per_year == 12


def test_asset_names_and_initial_weights_length_mismatch_returns_422(
) -> None:
    payload = _valid_payload()
    payload["initial_weights"] = [
        1.0,
    ]

    response = client.post(
        "/backtests/buy-and-hold",
        json=payload,
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "asset_names and initial_weights must have the same length."
    )


def test_market_return_row_width_mismatch_returns_422() -> None:
    payload = _valid_payload()
    payload["market_returns"] = [
        [
            0.01,
        ],
    ]

    response = client.post(
        "/backtests/buy-and-hold",
        json=payload,
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Each market return row must contain one value for every asset."
    )


def test_empty_asset_names_returns_422() -> None:
    payload = _valid_payload()
    payload["asset_names"] = []

    response = client.post(
        "/backtests/buy-and-hold",
        json=payload,
    )

    assert response.status_code == 422


def test_empty_market_returns_returns_422() -> None:
    payload = _valid_payload()
    payload["market_returns"] = []

    response = client.post(
        "/backtests/buy-and-hold",
        json=payload,
    )

    assert response.status_code == 422


def test_empty_initial_weights_returns_422() -> None:
    payload = _valid_payload()
    payload["initial_weights"] = []

    response = client.post(
        "/backtests/buy-and-hold",
        json=payload,
    )

    assert response.status_code == 422


def test_negative_portfolio_value_returns_422() -> None:
    payload = _valid_payload()
    payload["initial_portfolio_value"] = -1.0

    response = client.post(
        "/backtests/buy-and-hold",
        json=payload,
    )

    assert response.status_code == 422


def test_non_positive_periods_per_year_returns_422() -> None:
    payload = _valid_payload()
    payload["periods_per_year"] = 0

    response = client.post(
        "/backtests/buy-and-hold",
        json=payload,
    )

    assert response.status_code == 422


def test_domain_value_error_becomes_http_422() -> None:
    _override_runner(ValueErrorBacktestRunner())

    try:
        response = client.post(
            "/backtests/buy-and-hold",
            json=_valid_payload(),
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
    assert response.json()["detail"] == "Domain validation failed."


def test_domain_type_error_becomes_http_422() -> None:
    _override_runner(TypeErrorBacktestRunner())

    try:
        response = client.post(
            "/backtests/buy-and-hold",
            json=_valid_payload(),
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
    assert (
        response.json()["detail"]
        == "Domain type validation failed."
    )


def test_missing_pandas_values_in_history_serialize_to_null() -> None:
    history = pd.DataFrame(
        [
            {
                "date": "initial",
                "portfolio_value": 100_000.0,
                "optional_value": float("nan"),
            }
        ]
    )
    _override_runner(
        CapturingBacktestRunner(
            result=_backtest_result(history),
        )
    )

    try:
        response = client.post(
            "/backtests/buy-and-hold",
            json=_valid_payload(),
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["portfolio_history"][0][
        "optional_value"
    ] is None


def test_input_request_is_not_mutated() -> None:
    payload = _valid_payload()
    original_payload = deepcopy(payload)
    _override_runner(CapturingBacktestRunner())

    try:
        response = client.post(
            "/backtests/buy-and-hold",
            json=payload,
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert payload == original_payload


def test_runner_receives_pandas_dataframe() -> None:
    runner = CapturingBacktestRunner()
    _override_runner(runner)

    try:
        response = client.post(
            "/backtests/buy-and-hold",
            json=_valid_payload(),
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert isinstance(
        runner.market_returns,
        pd.DataFrame,
    )


def test_dataframe_columns_match_asset_names_in_order() -> None:
    runner = CapturingBacktestRunner()
    _override_runner(runner)

    try:
        response = client.post(
            "/backtests/buy-and-hold",
            json=_valid_payload(),
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert runner.market_returns is not None
    assert list(runner.market_returns.columns) == [
        "domestic_equity",
        "fixed_income",
    ]
