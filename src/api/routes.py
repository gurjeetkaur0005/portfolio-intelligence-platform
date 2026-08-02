from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.encoders import jsonable_encoder

from src.agents.orchestrator_agent import (
    AgentExecutionStatus,
    OrchestratorAgent,
    OrchestratorRequest,
)
from src.api.dependencies import (
    get_language_model,
    get_orchestrator_agent,
)
from src.api.schemas import (
    PortfolioExplanationResponse,
    RebalanceExplanationResponse,
    RebalanceRequest,
    RebalanceResponse,
)
from src.llm.language_model import LanguageModelProtocol

router = APIRouter()


@router.post(
    "/rebalance",
    response_model=RebalanceResponse,
    status_code=status.HTTP_200_OK,
    tags=["Rebalancing"],
)
def rebalance_portfolios(
    request: RebalanceRequest,
    orchestrator: OrchestratorAgent = Depends(
        get_orchestrator_agent
    ),
) -> RebalanceResponse:
    """
    Run the deterministic portfolio-rebalancing workflow.

    FastAPI validates the incoming request before this function is
    executed. The endpoint delegates all financial work to the existing
    Orchestrator Agent.
    """

    orchestrator_request = OrchestratorRequest(
        number_of_clients=request.number_of_clients,
        evaluation_date=request.evaluation_date,
        portfolio_value=request.portfolio_value,
        transaction_cost_rate=request.transaction_cost_rate,
    )

    orchestrator_response = orchestrator.execute_rebalance(
        orchestrator_request
    )

    if (
        orchestrator_response.status
        == AgentExecutionStatus.FAILED
    ):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=orchestrator_response.message,
        )

    if orchestrator_response.result is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "The rebalance workflow completed without a result."
            ),
        )

    records = _serialize_dataframe(
        orchestrator_response.result
    )

    return RebalanceResponse(
        status=orchestrator_response.status.value,
        workflow_name=orchestrator_response.workflow_name,
        message=orchestrator_response.message,
        records=records,
        record_count=len(records),
    )


def _serialize_dataframe(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Convert a pandas DataFrame into JSON-safe records.

    Missing pandas values are converted into None so FastAPI can
    serialize them as JSON null values.
    """

    safe_dataframe = dataframe.astype(object).where(
        pd.notna(dataframe),
        None,
    )

    raw_records = safe_dataframe.to_dict(
        orient="records"
    )

    encoded_records = jsonable_encoder(raw_records)

    if not isinstance(encoded_records, list):
        raise TypeError(
            "Serialized DataFrame records must be a list."
        )

    return [
        dict(record)
        for record in encoded_records
    ]

@router.post(
    "/rebalance/explain",
    response_model=RebalanceExplanationResponse,
    tags=["Rebalancing"],
)
def rebalance_with_explanations(
    request: RebalanceRequest,
    orchestrator: OrchestratorAgent = Depends(
        get_orchestrator_agent
    ),
    language_model: LanguageModelProtocol = Depends(
        get_language_model
    ),
) -> RebalanceExplanationResponse:
    """
    Run portfolio rebalancing and generate portfolio-level explanations.

    Financial calculations remain inside the deterministic pipeline.
    The language model is used only for client communication.
    """

    orchestrator_request = OrchestratorRequest(
        number_of_clients=request.number_of_clients,
        evaluation_date=request.evaluation_date,
        portfolio_value=request.portfolio_value,
        transaction_cost_rate=request.transaction_cost_rate,
    )

    result = orchestrator.execute_rebalance_with_explanations(
        request=orchestrator_request,
        language_model=language_model,
    )

    if result.status == AgentExecutionStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.message,
        )

    if result.explanations is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "The explanation workflow completed without "
                "explanations."
            ),
        )

    explanations = [
        PortfolioExplanationResponse(
            portfolio_id=str(explanation.portfolio_id),
            client_summary=explanation.client_summary,
            advisor_summary=explanation.advisor_summary,
            compliance_summary=explanation.compliance_summary,
        )
        for explanation in result.explanations
    ]

    return RebalanceExplanationResponse(
        status=result.status.value,
        workflow_name=result.workflow_name,
        message=result.message,
        explanations=explanations,
        portfolio_count=len(explanations),
    )
