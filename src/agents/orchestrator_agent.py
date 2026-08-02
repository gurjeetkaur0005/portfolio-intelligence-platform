from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Protocol

import pandas as pd

from src.agents.explanation_agent import (
    ExplanationAgent,
    PortfolioExplanation,
)
from src.agents.portfolio_analyst_agent import (
    PortfolioAnalystAgent,
    PortfolioAnalysis,
)
from src.pipeline.rebalance_pipeline import (
    run_rebalance_pipeline,
)


DEFAULT_NUMBER_OF_CLIENTS = 1
DEFAULT_PORTFOLIO_VALUE = 1_000_000.0
DEFAULT_TRANSACTION_COST_RATE = 0.002


class AgentExecutionStatus(StrEnum):
    """
    Status of an orchestrated workflow execution.
    """

    SUCCESS = "success"
    FAILED = "failed"


class RebalancePipelineProtocol(Protocol):
    """
    Interface required by the Orchestrator Agent.

    Any callable matching this interface can be used as the rebalance
    pipeline. This supports dependency injection and isolated testing.
    """

    def __call__(
        self,
        number_of_clients: int = DEFAULT_NUMBER_OF_CLIENTS,
        evaluation_date: date | None = None,
        portfolio_value: float = DEFAULT_PORTFOLIO_VALUE,
        transaction_cost_rate: float = (
            DEFAULT_TRANSACTION_COST_RATE
        ),
    ) -> pd.DataFrame:
        """
        Run the deterministic portfolio rebalancing workflow.
        """

        ...


class PortfolioAnalystProtocol(Protocol):
    """Interface required from portfolio analysis agents."""

    def analyze(
        self,
        trade_list: pd.DataFrame,
    ) -> list[PortfolioAnalysis]:
        """Analyze deterministic trade results."""
        ...


class ExplanationAgentProtocol(Protocol):
    """Interface required from portfolio explanation agents."""

    def explain(
        self,
        analyses: list[PortfolioAnalysis],
    ) -> list[PortfolioExplanation]:
        """Generate portfolio-level explanations."""
        ...


@dataclass(frozen=True)
class OrchestratorRequest:
    """
    Input required to execute the rebalance workflow.
    """

    number_of_clients: int = DEFAULT_NUMBER_OF_CLIENTS
    evaluation_date: date | None = None
    portfolio_value: float = DEFAULT_PORTFOLIO_VALUE
    transaction_cost_rate: float = (
        DEFAULT_TRANSACTION_COST_RATE
    )


@dataclass(frozen=True)
class OrchestratorResponse:
    """
    Structured output returned by the Orchestrator Agent.
    """

    status: AgentExecutionStatus
    workflow_name: str
    message: str
    result: pd.DataFrame | None


@dataclass(frozen=True)
class OrchestratorExplanationResponse:
    """
    Structured output for end-to-end explained rebalance workflows.
    """

    status: AgentExecutionStatus
    workflow_name: str
    message: str
    result: pd.DataFrame | None
    analyses: list[PortfolioAnalysis] | None
    explanations: list[PortfolioExplanation] | None


class OrchestratorAgent:
    """
    Coordinate deterministic portfolio workflows.

    The agent delegates all financial calculations to the existing
    pipeline. It does not calculate drift, optimize portfolios, generate
    trades, estimate taxes, or produce financial metrics itself.
    """

    WORKFLOW_NAME = "portfolio_rebalancing"

    def __init__(
        self,
        rebalance_pipeline: RebalancePipelineProtocol = (
            run_rebalance_pipeline
        ),
        portfolio_analyst: PortfolioAnalystProtocol | None = None,
        explanation_agent: ExplanationAgentProtocol | None = None,
    ) -> None:
        """
        Initialize the agent with a rebalance pipeline dependency.

        Args:
            rebalance_pipeline:
                Callable responsible for executing the deterministic
                portfolio rebalancing workflow.
        """

        self._rebalance_pipeline = rebalance_pipeline
        self._portfolio_analyst = (
            portfolio_analyst
            if portfolio_analyst is not None
            else PortfolioAnalystAgent()
        )
        self._explanation_agent = (
            explanation_agent
            if explanation_agent is not None
            else ExplanationAgent()
        )

    def execute_rebalance(
        self,
        request: OrchestratorRequest,
    ) -> OrchestratorResponse:
        """
        Validate the request and execute the rebalance pipeline.

        Args:
            request:
                Structured workflow input.

        Returns:
            A structured response containing the pipeline result or a
            safe failure message.
        """

        self._validate_request(request)

        try:
            result = self._rebalance_pipeline(
                number_of_clients=request.number_of_clients,
                evaluation_date=request.evaluation_date,
                portfolio_value=request.portfolio_value,
                transaction_cost_rate=(
                    request.transaction_cost_rate
                ),
            )
        except Exception as error:
            return OrchestratorResponse(
                status=AgentExecutionStatus.FAILED,
                workflow_name=self.WORKFLOW_NAME,
                message=(
                    "The portfolio rebalancing workflow failed: "
                    f"{error}"
                ),
                result=None,
            )

        self._validate_pipeline_result(result)

        return OrchestratorResponse(
            status=AgentExecutionStatus.SUCCESS,
            workflow_name=self.WORKFLOW_NAME,
            message=(
                "The portfolio rebalancing workflow completed "
                "successfully."
            ),
            result=result.copy(),
        )

    def execute_rebalance_with_explanations(
        self,
        request: OrchestratorRequest,
    ) -> OrchestratorExplanationResponse:
        """
        Execute the rebalance pipeline and portfolio explanation flow.

        This method preserves the existing execute_rebalance contract while
        proving the connected runtime path:

        pipeline -> portfolio analyst -> explanation agent -> optional LLM.
        """

        response = self.execute_rebalance(request)

        if response.status is AgentExecutionStatus.FAILED:
            return OrchestratorExplanationResponse(
                status=response.status,
                workflow_name=response.workflow_name,
                message=response.message,
                result=response.result,
                analyses=None,
                explanations=None,
            )

        if response.result is None or response.result.empty:
            return OrchestratorExplanationResponse(
                status=AgentExecutionStatus.SUCCESS,
                workflow_name=self.WORKFLOW_NAME,
                message=(
                    "The portfolio rebalancing workflow completed "
                    "successfully. No explanations were generated "
                    "because no trades were produced."
                ),
                result=response.result,
                analyses=[],
                explanations=[],
            )

        try:
            analyses = self._portfolio_analyst.analyze(
                response.result
            )
            explanations = self._explanation_agent.explain(
                analyses
            )
        except Exception as error:
            return OrchestratorExplanationResponse(
                status=AgentExecutionStatus.FAILED,
                workflow_name=self.WORKFLOW_NAME,
                message=(
                    "The portfolio explanation workflow failed: "
                    f"{error}"
                ),
                result=response.result,
                analyses=None,
                explanations=None,
            )

        return OrchestratorExplanationResponse(
            status=AgentExecutionStatus.SUCCESS,
            workflow_name=self.WORKFLOW_NAME,
            message=(
                "The portfolio rebalancing and explanation workflow "
                "completed successfully."
            ),
            result=response.result.copy(),
            analyses=list(analyses),
            explanations=list(explanations),
        )

    @staticmethod
    def _validate_request(
        request: OrchestratorRequest,
    ) -> None:
        """
        Validate orchestration inputs before calling the pipeline.
        """

        if not isinstance(request.number_of_clients, int):
            raise ValueError(
                "number_of_clients must be an integer.",
            )

        if request.number_of_clients <= 0:
            raise ValueError(
                "number_of_clients must be greater than zero.",
            )

        if not isinstance(
            request.portfolio_value,
            int | float,
        ):
            raise ValueError(
                "portfolio_value must be numeric.",
            )

        if request.portfolio_value <= 0.0:
            raise ValueError(
                "portfolio_value must be greater than zero.",
            )

        if not isinstance(
            request.transaction_cost_rate,
            int | float,
        ):
            raise ValueError(
                "transaction_cost_rate must be numeric.",
            )

        if request.transaction_cost_rate < 0.0:
            raise ValueError(
                "transaction_cost_rate cannot be negative.",
            )

        if request.transaction_cost_rate > 1.0:
            raise ValueError(
                "transaction_cost_rate cannot exceed 1.0.",
            )

        if (
            request.evaluation_date is not None
            and not isinstance(request.evaluation_date, date)
        ):
            raise ValueError(
                "evaluation_date must be a date or None.",
            )

    @staticmethod
    def _validate_pipeline_result(
        result: pd.DataFrame,
    ) -> None:
        """
        Validate the contract returned by the deterministic pipeline.
        """

        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                "The rebalance pipeline must return a pandas "
                "DataFrame."
            )
