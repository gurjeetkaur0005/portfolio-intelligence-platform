from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
import os
from typing import Protocol

import numpy as np
import pandas as pd

from src.agents.explanation_agent import ExplanationAgent
from src.agents.orchestrator_agent import OrchestratorAgent
from src.agents.portfolio_analyst_agent import PortfolioAnalystAgent
from src.backtesting.backtest_engine import (
    BacktestResult,
    run_buy_and_hold_backtest,
    run_threshold_rebalancing_backtest,
)
from src.backtesting.strategy_comparison import (
    StrategyComparisonResult,
    compare_backtest_results,
)
from src.llm.gemini_language_model import (
    DEFAULT_GEMINI_MODEL,
    GEMINI_API_KEY_ENV,
)
from fastapi import Depends
from sqlalchemy.orm import Session
from src.llm.language_model import LanguageModelProtocol
from src.llm.language_model_factory import (
    LanguageModelProvider,
    create_language_model,
)
from src.llm.prompt_builder import PromptBuilder
from src.database.persistence_service import (
    RebalancePersistenceService,
)
from src.database.repositories import (
    PortfolioRepository,
    RebalanceRunRepository,
)
from src.database.session import get_database_session
from src.services.rebalance_application_service import (
    RebalanceApplicationService,
)
from src.services.portfolio_read_application_service import (
    PortfolioReadApplicationService,
)

ENABLE_PROMPT_PREVIEW_ENV = "ENABLE_PROMPT_PREVIEW"


class BuyAndHoldBacktestProtocol(Protocol):
    """Contract for the Buy & Hold backtest runner."""

    def __call__(
        self,
        initial_weights: Sequence[float],
        market_returns: pd.DataFrame,
        initial_portfolio_value: float = 100_000.0,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> BacktestResult:
        """Run a Buy & Hold backtest."""
        ...


class ThresholdRebalancingBacktestProtocol(Protocol):
    """Contract for the Threshold Rebalancing backtest runner."""

    def __call__(
        self,
        initial_weights: Sequence[float],
        target_weights: Sequence[float],
        market_returns: pd.DataFrame,
        initial_portfolio_value: float = 100_000.0,
        drift_band: float = 0.05,
        transaction_cost_rate: float = 0.002,
        tax_rate: float = 0.20,
        turnover_budget: float = 0.10,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
        portfolio_id: str = "BACKTEST",
        covariance_matrix: np.ndarray | None = None,
    ) -> BacktestResult:
        """Run a Threshold Rebalancing backtest."""
        ...


class StrategyComparisonProtocol(Protocol):
    """Contract for deterministic strategy comparison."""

    def __call__(
        self,
        buy_and_hold_result: object,
        threshold_rebalancing_result: object,
    ) -> StrategyComparisonResult:
        """Compare two strategy result objects."""
        ...


@lru_cache(maxsize=1)
def get_orchestrator_agent() -> OrchestratorAgent:
    """Return the shared Orchestrator Agent."""

    return OrchestratorAgent()


@lru_cache(maxsize=1)
def get_language_model() -> LanguageModelProtocol:
    """Return the configured language-model provider."""

    return create_language_model(
        provider=LanguageModelProvider.GEMINI,
    )


def get_optional_language_model() -> LanguageModelProtocol | None:
    """Return the configured language model, if safely available."""

    try:
        return get_language_model()
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def get_portfolio_analyst_agent() -> PortfolioAnalystAgent:
    """Return the shared Portfolio Analyst Agent."""

    return PortfolioAnalystAgent()


def get_explanation_agent(
    language_model: LanguageModelProtocol | None = None,
) -> ExplanationAgent:
    """Return an Explanation Agent with an optional language model."""

    return ExplanationAgent(
        language_model=language_model,
    )


def get_prompt_builder() -> PromptBuilder:
    """Return the provider-neutral prompt builder."""

    return PromptBuilder()


def get_buy_and_hold_backtest_runner(
) -> BuyAndHoldBacktestProtocol:
    """Return the production Buy & Hold backtest function."""

    return run_buy_and_hold_backtest


def get_threshold_rebalancing_backtest_runner(
) -> ThresholdRebalancingBacktestProtocol:
    """Return the production Threshold Rebalancing backtest function."""

    return run_threshold_rebalancing_backtest


def get_strategy_comparison_runner() -> StrategyComparisonProtocol:
    """Return the deterministic strategy comparison function."""

    return compare_backtest_results


def get_llm_health_status() -> dict[str, object]:
    """Return safe LLM configuration status without live network calls."""

    configured = bool(os.getenv(GEMINI_API_KEY_ENV, "").strip())

    return {
        "status": "configured" if configured else "not_configured",
        "provider": LanguageModelProvider.GEMINI.value,
        "model_name": DEFAULT_GEMINI_MODEL,
        "configured": configured,
        "live_check_performed": False,
    }


def is_prompt_preview_enabled() -> bool:
    """Return whether development-only prompt preview is enabled."""

    return os.getenv(ENABLE_PROMPT_PREVIEW_ENV) == "true"


def get_rebalance_application_service(
    session: Session = Depends(get_database_session),
    orchestrator: OrchestratorAgent = Depends(
        get_orchestrator_agent
    ),
) -> RebalanceApplicationService:
    """Build the database-backed rebalance service."""

    portfolio_repository = PortfolioRepository(session)

    rebalance_run_repository = RebalanceRunRepository(
        session
    )

    persistence_service = RebalancePersistenceService(
        rebalance_run_repository
    )

    return RebalanceApplicationService(
        portfolio_repository=portfolio_repository,
        orchestrator=orchestrator,
        persistence_service=persistence_service,
    )


def get_portfolio_read_application_service(
    session: Session = Depends(get_database_session),
) -> PortfolioReadApplicationService:
    """Build the database-backed read service."""

    portfolio_repository = PortfolioRepository(session)
    rebalance_run_repository = RebalanceRunRepository(
        session
    )

    return PortfolioReadApplicationService(
        portfolio_repository=portfolio_repository,
        rebalance_repository=rebalance_run_repository,
    )
