from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from src.agents.orchestrator_agent import (
    AgentExecutionStatus,
    OrchestratorAgent,
    OrchestratorRequest,
)
from src.api.dependencies import get_orchestrator_agent
from src.api.schemas import (
    RebalanceRequest,
    RebalanceResponse,
)


router = APIRouter()


@router.post(
    "/rebalance",
    response_model=RebalanceResponse,
    tags=["Rebalancing"],
)
def rebalance(
    request: RebalanceRequest,
    orchestrator: OrchestratorAgent = Depends(
        get_orchestrator_agent
    ),
) -> RebalanceResponse:
    """
    Run the deterministic rebalance workflow and serialize the result.
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

    if orchestrator_response.status is AgentExecutionStatus.FAILED:
        raise HTTPException(
            status_code=500,
            detail=orchestrator_response.message,
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
    dataframe: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    """Convert a DataFrame into JSON-safe row dictionaries."""

    if dataframe is None or dataframe.empty:
        return []

    safe_dataframe = dataframe.astype(object).where(
        pd.notna(dataframe),
        None,
    )

    return [
        dict(record)
        for record in safe_dataframe.to_dict(
            orient="records"
        )
    ]
