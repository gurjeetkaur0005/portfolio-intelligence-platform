from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Protocol

import pandas as pd

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
    ) -> None:
        """
        Initialize the agent with a rebalance pipeline dependency.

        Args:
            rebalance_pipeline:
                Callable responsible for executing the deterministic
                portfolio rebalancing workflow.
        """

        self._rebalance_pipeline = rebalance_pipeline

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