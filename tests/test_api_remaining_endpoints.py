from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.agents.orchestrator_agent import (
    AgentExecutionStatus,
    OrchestratorRequest,
    OrchestratorResponse,
)
from src.agents.portfolio_analyst_agent import PortfolioAnalysis
from src.api.dependencies import (
    get_llm_health_status,
    get_optional_language_model,
    get_orchestrator_agent,
    get_portfolio_analyst_agent,
    get_prompt_builder,
    get_strategy_comparison_runner,
    get_threshold_rebalancing_backtest_runner,
    is_prompt_preview_enabled,
)
from src.api.main import app
from src.api.schemas import LlmHealthResponse
from src.backtesting.backtest_engine import BacktestResult
from src.backtesting.strategy_comparison import compare_backtest_results
from src.llm.fake_language_model import FakeLanguageModel
from src.llm.language_model import (
    LanguageModelRequest,
    LanguageModelResponse,
)
from src.llm.prompt_builder import PromptBuilder


client = TestClient(app)


def _threshold_payload() -> dict[str, Any]:
    return {
        "asset_names": [
            "domestic_equity",
            "international_equity",
            "fixed_income",
            "real_estate",
            "commodities",
            "cash",
        ],
        "market_returns": [
            [
                0.01,
                0.002,
                0.001,
                0.003,
                -0.001,
                0.0001,
            ]
        ],
        "initial_weights": [
            0.30,
            0.20,
            0.30,
            0.10,
            0.05,
            0.05,
        ],
        "target_weights": [
            0.28,
            0.22,
            0.30,
            0.10,
            0.05,
            0.05,
        ],
    }


def _analysis_payload() -> dict[str, Any]:
    return {
        "portfolio_id": "P00001",
        "rebalance_required": True,
        "highest_threshold_severity": "high",
        "threshold_breached": True,
        "threshold_breach_count": 1,
        "assets_to_buy": [
            "fixed_income",
        ],
        "assets_to_sell": [
            "domestic_equity",
        ],
        "assets_to_hold": [
            "cash",
        ],
        "total_transaction_cost": 12.5,
        "total_estimated_tax_liability": 25.0,
        "client_explanations": [
            "Fixed income will be increased.",
        ],
        "advisor_explanations": [
            "Buy fixed income.",
        ],
        "compliance_explanations": [
            "Deterministic recommendation recorded.",
        ],
    }


def _fake_backtest_result(
    history: pd.DataFrame | None = None,
) -> BacktestResult:
    if history is None:
        history = pd.DataFrame(
            [
                {
                    "date": "initial",
                    "portfolio_value": 100_000.0,
                    "rebalanced": False,
                    "transaction_cost": 0.0,
                    "estimated_tax_liability": 0.0,
                }
            ]
        )

    return BacktestResult(
        portfolio_history=history,
        total_return=0.10,
        annualized_return=0.08,
        volatility=0.12,
        sharpe_ratio=0.66,
        maximum_drawdown=0.05,
    )


class CapturingThresholdRunner:
    def __init__(self) -> None:
        self.market_returns: pd.DataFrame | None = None
        self.covariance_matrix: np.ndarray | None = None

    def __call__(
        self,
        *,
        initial_weights,
        target_weights,
        market_returns,
        initial_portfolio_value=100_000.0,
        drift_band=0.05,
        transaction_cost_rate=0.002,
        tax_rate=0.20,
        turnover_budget=0.10,
        risk_free_rate=0.0,
        periods_per_year=252,
        portfolio_id="BACKTEST",
        covariance_matrix=None,
    ) -> BacktestResult:
        self.market_returns = market_returns
        self.covariance_matrix = covariance_matrix
        return _fake_backtest_result()


class ErrorRunner:
    def __call__(self, *args, **kwargs) -> BacktestResult:
        raise ValueError("Domain validation failed.")


class FakeSuccessfulOrchestrator:
    def execute_rebalance(
        self,
        request: OrchestratorRequest,
    ) -> OrchestratorResponse:
        return OrchestratorResponse(
            status=AgentExecutionStatus.SUCCESS,
            workflow_name="portfolio_rebalancing",
            message="Workflow completed.",
            result=pd.DataFrame(
                [
                    {
                        "portfolio_id": "P00001",
                        "asset": "fixed_income",
                        "action": "BUY",
                        "threshold_breached": True,
                        "threshold_severity": "high",
                        "transaction_cost": 12.5,
                        "estimated_tax_liability": 25.0,
                        "client_explanation": "Client detail.",
                        "advisor_explanation": "Advisor detail.",
                        "compliance_explanation": "Compliance detail.",
                    }
                ]
            ),
        )


class FakeFailedOrchestrator:
    def execute_rebalance(
        self,
        request: OrchestratorRequest,
    ) -> OrchestratorResponse:
        return OrchestratorResponse(
            status=AgentExecutionStatus.FAILED,
            workflow_name="portfolio_rebalancing",
            message="Sensitive internal error should not leak.",
            result=None,
        )


class FakeAnalyst:
    def analyze(
        self,
        trade_list: pd.DataFrame,
    ) -> list[PortfolioAnalysis]:
        return [
            PortfolioAnalysis(
                portfolio_id="P00001",
                rebalance_required=True,
                highest_threshold_severity="high",
                threshold_breached=True,
                threshold_breach_count=1,
                assets_to_buy=("fixed_income",),
                assets_to_sell=("domestic_equity",),
                assets_to_hold=("cash",),
                total_transaction_cost=12.5,
                total_estimated_tax_liability=25.0,
                client_explanations=("Client detail.",),
                advisor_explanations=("Advisor detail.",),
                compliance_explanations=("Compliance detail.",),
            )
        ]


class ExceptionLanguageModel:
    def generate(
        self,
        request: LanguageModelRequest,
    ) -> LanguageModelResponse:
        raise RuntimeError("Provider failure with secret token.")


class CapturingPromptBuilder(PromptBuilder):
    def __init__(self) -> None:
        self.call_count = 0

    def build(self, analysis, audience):
        self.call_count += 1
        return super().build(
            analysis=analysis,
            audience=audience,
        )


def test_threshold_rebalancing_backtest_success() -> None:
    runner = CapturingThresholdRunner()
    app.dependency_overrides[
        get_threshold_rebalancing_backtest_runner
    ] = lambda: runner

    try:
        response = client.post(
            "/backtests/threshold-rebalancing",
            json=_threshold_payload(),
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["strategy_name"] == "Threshold Rebalancing"
    assert body["history_record_count"] == 1
    assert body["metrics"]["total_return"] == pytest.approx(0.10)
    assert isinstance(runner.market_returns, pd.DataFrame)


def test_threshold_rebalancing_validates_row_width() -> None:
    payload = _threshold_payload()
    payload["market_returns"] = [
        [
            0.01,
        ]
    ]

    response = client.post(
        "/backtests/threshold-rebalancing",
        json=payload,
    )

    assert response.status_code == 422


def test_threshold_rebalancing_domain_error_becomes_422() -> None:
    app.dependency_overrides[
        get_threshold_rebalancing_backtest_runner
    ] = lambda: ErrorRunner()

    try:
        response = client.post(
            "/backtests/threshold-rebalancing",
            json=_threshold_payload(),
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
    assert response.json()["detail"] == "Domain validation failed."


def test_strategy_comparison_success_uses_existing_runner() -> None:
    response = client.post(
        "/strategy-comparisons",
        json={
            "buy_and_hold": {
                "total_return": 0.10,
                "annualized_return": 0.08,
                "volatility": 0.12,
                "sharpe_ratio": 0.66,
                "maximum_drawdown": 0.05,
            },
            "threshold_rebalancing": {
                "total_return": 0.11,
                "annualized_return": 0.09,
                "volatility": 0.10,
                "sharpe_ratio": 0.90,
                "maximum_drawdown": 0.03,
                "transaction_costs": 10.0,
                "taxes_paid": 20.0,
                "number_of_rebalances": 2,
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["buy_and_hold"]["strategy_name"] == "Buy & Hold"
    assert body["threshold_rebalancing"]["transaction_costs"] == 10.0
    assert "2 rebalances" in body["performance_summary"]


def test_strategy_comparison_dependency_override() -> None:
    called = False

    def fake_runner(
        buy_and_hold_result,
        threshold_rebalancing_result,
    ):
        nonlocal called
        called = True
        return compare_backtest_results(
            buy_and_hold_result,
            threshold_rebalancing_result,
        )

    app.dependency_overrides[
        get_strategy_comparison_runner
    ] = lambda: fake_runner

    try:
        response = client.post(
            "/strategy-comparisons",
            json={
                "buy_and_hold": {
                    "total_return": 0.10,
                    "annualized_return": 0.08,
                    "volatility": 0.12,
                    "sharpe_ratio": 0.66,
                    "maximum_drawdown": 0.05,
                },
                "threshold_rebalancing": {
                    "total_return": 0.10,
                    "annualized_return": 0.08,
                    "volatility": 0.12,
                    "sharpe_ratio": 0.66,
                    "maximum_drawdown": 0.05,
                },
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert called is True


def test_portfolio_analysis_returns_analysis() -> None:
    app.dependency_overrides[
        get_orchestrator_agent
    ] = FakeSuccessfulOrchestrator
    app.dependency_overrides[
        get_portfolio_analyst_agent
    ] = FakeAnalyst

    try:
        response = client.post(
            "/portfolio-analysis",
            json={
                "number_of_clients": 1,
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["analysis_count"] == 1
    assert body["analyses"][0]["assets_to_buy"] == [
        "fixed_income"
    ]


def test_portfolio_analysis_failure_returns_safe_500() -> None:
    app.dependency_overrides[
        get_orchestrator_agent
    ] = FakeFailedOrchestrator

    try:
        response = client.post(
            "/portfolio-analysis",
            json={},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    assert "Sensitive" not in response.text


def test_portfolio_explanations_complete_fake_llm_response() -> None:
    fake_model = FakeLanguageModel(
        responses=[
            (
                "This complete client summary is generated from the "
                "supplied facts and ends safely."
            )
        ]
    )
    app.dependency_overrides[
        get_optional_language_model
    ] = lambda: fake_model

    try:
        response = client.post(
            "/portfolio-explanations",
            json={
                "analyses": [
                    _analysis_payload(),
                ]
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert (
        response.json()["explanations"][0]["client_summary"]
        == fake_model.responses[0]
    )
    assert fake_model.call_count == 1


def test_portfolio_explanations_provider_failure_falls_back() -> None:
    app.dependency_overrides[
        get_optional_language_model
    ] = lambda: ExceptionLanguageModel()

    try:
        response = client.post(
            "/portfolio-explanations",
            json={
                "analyses": [
                    _analysis_payload(),
                ]
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()["explanations"][0]
    assert "Provider failure" not in body["client_summary"]
    assert body["advisor_summary"].startswith("Portfolio P00001")


def test_portfolio_explanations_input_not_mutated() -> None:
    payload = {
        "analyses": [
            _analysis_payload(),
        ]
    }
    original = deepcopy(payload)

    response = client.post(
        "/portfolio-explanations",
        json=payload,
    )

    assert response.status_code == 200
    assert payload == original


def test_llm_health_is_safe_and_contains_no_secret() -> None:
    app.dependency_overrides[get_llm_health_status] = lambda: {
        "status": "configured",
        "provider": "gemini",
        "model_name": "gemini-3.6-flash",
        "configured": True,
        "live_check_performed": False,
    }

    try:
        response = client.get("/llm/health")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["configured"] is True
    assert "API_KEY" not in response.text
    assert "secret" not in response.text.lower()


def test_prompt_preview_disabled_by_default() -> None:
    response = client.post(
        "/llm/prompts/preview",
        json={
            "analysis": _analysis_payload(),
            "audience": "client",
        },
    )

    assert response.status_code == 404


def test_prompt_preview_enabled_builds_prompt_without_gemini() -> None:
    builder = CapturingPromptBuilder()
    app.dependency_overrides[
        is_prompt_preview_enabled
    ] = lambda: True
    app.dependency_overrides[get_prompt_builder] = lambda: builder

    try:
        response = client.post(
            "/llm/prompts/preview",
            json={
                "analysis": _analysis_payload(),
                "audience": "client",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["audience"] == "client"
    assert "system_prompt" in body
    assert builder.call_count == 1


def test_prompt_preview_invalid_audience_returns_422() -> None:
    app.dependency_overrides[
        is_prompt_preview_enabled
    ] = lambda: True

    try:
        response = client.post(
            "/llm/prompts/preview",
            json={
                "analysis": _analysis_payload(),
                "audience": "invalid",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422


def test_openapi_contains_expected_routes_once() -> None:
    paths = client.get("/openapi.json").json()["paths"]

    for path in [
        "/health",
        "/llm/health",
        "/rebalance",
        "/rebalance/explain",
        "/portfolio-analysis",
        "/portfolio-explanations",
        "/backtests/buy-and-hold",
        "/backtests/threshold-rebalancing",
        "/strategy-comparisons",
        "/llm/prompts/preview",
    ]:
        assert path in paths

    route_paths = [
        route.path
        for route in app.routes
        if hasattr(route, "methods")
    ]
    assert route_paths.count("/backtests/threshold-rebalancing") == 1


def test_route_state_pollution_can_be_reproduced() -> None:
    client.get("/openapi.json")

    app.router.routes[:] = [
        route
        for route in app.router.routes
        if route.path
        in {
            "/openapi.json",
            "/docs",
            "/docs/oauth2-redirect",
            "/redoc",
            "/health",
        }
    ]

    route_paths = [
        route.path
        for route in app.routes
        if hasattr(route, "methods")
    ]

    assert "/backtests/threshold-rebalancing" not in route_paths


def test_api_route_state_is_intact_after_previous_test() -> None:
    paths = client.get("/openapi.json").json()["paths"]
    route_paths = [
        route.path
        for route in app.routes
        if hasattr(route, "methods")
    ]

    assert "/backtests/threshold-rebalancing" in paths
    assert route_paths.count("/backtests/threshold-rebalancing") == 1


def test_llm_health_response_is_immutable() -> None:
    health = LlmHealthResponse(
        status="configured",
        provider="gemini",
        model_name="gemini-3.6-flash",
        configured=True,
        live_check_performed=False,
    )

    with pytest.raises(ValidationError):
        health.status = "changed"
