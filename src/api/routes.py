from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.encoders import jsonable_encoder

from src.agents.explanation_agent import ExplanationAgent
from src.agents.orchestrator_agent import (
    AgentExecutionStatus,
    OrchestratorAgent,
    OrchestratorRequest,
)
from src.agents.portfolio_analyst_agent import (
    PortfolioAnalystAgent,
    PortfolioAnalysis,
)
from src.api.dependencies import (
    BuyAndHoldBacktestProtocol,
    StrategyComparisonProtocol,
    ThresholdRebalancingBacktestProtocol,
    get_buy_and_hold_backtest_runner,
    get_llm_health_status,
    get_language_model,
    get_optional_language_model,
    get_orchestrator_agent,
    get_portfolio_analyst_agent,
    get_prompt_builder,
    get_portfolio_read_application_service,
    get_rebalance_application_service,
    get_strategy_comparison_runner,
    get_threshold_rebalancing_backtest_runner,
    is_prompt_preview_enabled,
)
from src.api.schemas import (
    BacktestMetricsResponse,
    BuyAndHoldBacktestRequest,
    BuyAndHoldBacktestResponse,
    DatabaseRebalanceRequest,
    DatabaseRebalanceResponse,
    LlmHealthResponse,
    PortfolioDetailResponse,
    PortfolioHoldingResponse,
    PortfolioAnalysisApiPayload,
    PortfolioAnalysisEndpointResponse,
    PortfolioAnalysisResponse,
    PortfolioExplanationsRequest,
    PortfolioExplanationsResponse,
    PortfolioExplanationResponse,
    PortfolioListResponse,
    PortfolioRebalanceListResponse,
    PortfolioSummaryResponse,
    PromptPreviewRequest,
    PromptPreviewResponse,
    RebalanceApprovalResponse,
    RebalanceAuditEntryResponse,
    RebalanceAuditResponse,
    RebalanceExplanationResponse,
    RebalanceRequest,
    RebalanceResponse,
    RebalanceRunDetailResponse,
    RebalanceRunSummaryResponse,
    RebalanceTradeListResponse,
    RebalanceTradeResponse,
    StrategyComparisonRequest,
    StrategyComparisonResponse,
    StrategyMetricsResponse,
    ThresholdRebalancingBacktestRequest,
    ThresholdRebalancingBacktestResponse,
)
from src.backtesting.strategy_comparison import (
    THRESHOLD_REBALANCING_NAME,
)
from src.llm.language_model import LanguageModelProtocol
from src.llm.prompt_builder import PromptBuilder
from src.services.rebalance_application_service import (
    RebalanceApplicationService,
    RebalanceExecutionError,
    RebalancePersistenceError,
)
from src.services.portfolio_read_application_service import (
    PortfolioReadApplicationService,
)

router = APIRouter()

DEFAULT_PAGE_LIMIT = 20
MAX_PAGE_LIMIT = 50
DEFAULT_PAGE_OFFSET = 0


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


def _build_market_returns(
    asset_names: list[str],
    market_return_rows: list[list[float]],
) -> pd.DataFrame:
    """Build a market-return DataFrame from API request lists."""

    invalid_rows = [
        row
        for row in market_return_rows
        if len(row) != len(asset_names)
    ]

    if invalid_rows:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "Each market return row must contain one value "
                "for every asset."
            ),
        )

    return pd.DataFrame(
        market_return_rows,
        columns=asset_names,
    )


def _validate_weight_lengths(
    asset_names: list[str],
    **weight_sets: list[float],
) -> None:
    """Validate that every supplied weight list matches assets."""

    for field_name, weights in weight_sets.items():
        if len(asset_names) != len(weights):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    "asset_names and "
                    f"{field_name} must have the same length."
                ),
            )


def _metrics_response_from_result(
    result: Any,
) -> BacktestMetricsResponse:
    """Build API metrics from an existing backtest result object."""

    return BacktestMetricsResponse(
        total_return=float(result.total_return),
        annualized_return=float(result.annualized_return),
        volatility=float(result.volatility),
        sharpe_ratio=float(result.sharpe_ratio),
        maximum_drawdown=float(result.maximum_drawdown),
    )


def _analysis_response(
    analysis: PortfolioAnalysis,
) -> PortfolioAnalysisResponse:
    """Convert a domain PortfolioAnalysis into an API response."""

    return PortfolioAnalysisResponse(
        portfolio_id=str(analysis.portfolio_id),
        rebalance_required=analysis.rebalance_required,
        highest_threshold_severity=(
            analysis.highest_threshold_severity
        ),
        threshold_breached=analysis.threshold_breached,
        threshold_breach_count=analysis.threshold_breach_count,
        assets_to_buy=list(analysis.assets_to_buy),
        assets_to_sell=list(analysis.assets_to_sell),
        assets_to_hold=list(analysis.assets_to_hold),
        total_transaction_cost=analysis.total_transaction_cost,
        total_estimated_tax_liability=(
            analysis.total_estimated_tax_liability
        ),
        client_explanations=list(analysis.client_explanations),
        advisor_explanations=list(analysis.advisor_explanations),
        compliance_explanations=list(
            analysis.compliance_explanations
        ),
    )


def _analysis_from_payload(
    payload: PortfolioAnalysisApiPayload,
) -> PortfolioAnalysis:
    """Convert an API analysis payload into a domain dataclass."""

    return PortfolioAnalysis(
        portfolio_id=payload.portfolio_id,
        rebalance_required=payload.rebalance_required,
        highest_threshold_severity=(
            payload.highest_threshold_severity
        ),
        threshold_breached=payload.threshold_breached,
        threshold_breach_count=payload.threshold_breach_count,
        assets_to_buy=tuple(payload.assets_to_buy),
        assets_to_sell=tuple(payload.assets_to_sell),
        assets_to_hold=tuple(payload.assets_to_hold),
        total_transaction_cost=payload.total_transaction_cost,
        total_estimated_tax_liability=(
            payload.total_estimated_tax_liability
        ),
        client_explanations=tuple(payload.client_explanations),
        advisor_explanations=tuple(payload.advisor_explanations),
        compliance_explanations=tuple(
            payload.compliance_explanations
        ),
    )


def _explanation_response(
    explanation,
) -> PortfolioExplanationResponse:
    """Convert a domain portfolio explanation into an API response."""

    return PortfolioExplanationResponse(
        portfolio_id=str(explanation.portfolio_id),
        client_summary=explanation.client_summary,
        advisor_summary=explanation.advisor_summary,
        compliance_summary=explanation.compliance_summary,
    )


def _strategy_metrics_response(
    metrics,
) -> StrategyMetricsResponse:
    """Convert comparison metrics into an API response."""

    return StrategyMetricsResponse(
        strategy_name=metrics.strategy_name,
        total_return=metrics.total_return,
        annualized_return=metrics.annualized_return,
        volatility=metrics.volatility,
        sharpe_ratio=metrics.sharpe_ratio,
        maximum_drawdown=metrics.maximum_drawdown,
        transaction_costs=metrics.transaction_costs,
        taxes_paid=metrics.taxes_paid,
        number_of_rebalances=metrics.number_of_rebalances,
        total_implementation_cost=(
            metrics.total_implementation_cost
        ),
    )


def _portfolio_summary_response(
    portfolio: Any,
) -> PortfolioSummaryResponse:
    """Convert a portfolio summary DTO into an API response."""

    return PortfolioSummaryResponse(
        portfolio_id=portfolio.portfolio_id,
        client_id=portfolio.client_id,
        portfolio_value=float(portfolio.portfolio_value),
        currency=portfolio.currency,
    )


def _portfolio_detail_response(
    portfolio: Any,
) -> PortfolioDetailResponse:
    """Convert a portfolio detail DTO into an API response."""

    holdings = [
        PortfolioHoldingResponse(
            asset=holding.asset,
            current_weight=float(holding.current_weight),
            current_value=float(holding.current_value),
            cost_basis=float(holding.cost_basis),
        )
        for holding in portfolio.holdings
    ]

    return PortfolioDetailResponse(
        portfolio_id=portfolio.portfolio_id,
        client_id=portfolio.client_id,
        portfolio_value=float(portfolio.portfolio_value),
        currency=portfolio.currency,
        holdings=holdings,
        holding_count=len(holdings),
    )


def _rebalance_summary_response(
    rebalance: Any,
) -> RebalanceRunSummaryResponse:
    """Convert a rebalance summary DTO into an API response."""

    return RebalanceRunSummaryResponse(
        run_id=rebalance.run_id,
        status=rebalance.status,
        created_at=rebalance.created_at,
        transaction_cost=float(
            rebalance.transaction_cost
        ),
        portfolio_value=float(rebalance.portfolio_value),
    )


def _rebalance_detail_response(
    rebalance: Any,
) -> RebalanceRunDetailResponse:
    """Convert a rebalance detail DTO into an API response."""

    return RebalanceRunDetailResponse(
        run_id=rebalance.run_id,
        portfolio_id=rebalance.portfolio_id,
        status=rebalance.status,
        created_at=rebalance.created_at,
        completed_at=rebalance.completed_at,
        portfolio_value=float(rebalance.portfolio_value),
        transaction_cost_rate=float(
            rebalance.transaction_cost_rate
        ),
        trade_count=rebalance.trade_count,
        transaction_cost=float(
            rebalance.transaction_cost
        ),
        estimated_tax_liability=float(
            rebalance.estimated_tax_liability
        ),
        approval_required_count=(
            rebalance.approval_required_count
        ),
        pending_approval_count=(
            rebalance.pending_approval_count
        ),
    )


def _rebalance_trade_response(
    trade: Any,
) -> RebalanceTradeResponse:
    """Convert a rebalance trade DTO into an API response."""

    approval = (
        None
        if trade.approval is None
        else RebalanceApprovalResponse(
            required=trade.approval.required,
            status=trade.approval.status,
            reason=trade.approval.reason,
            reviewed_by=trade.approval.reviewed_by,
            reviewed_at=trade.approval.reviewed_at,
        )
    )

    return RebalanceTradeResponse(
        asset=trade.asset,
        action=trade.action,
        current_weight=float(trade.current_weight),
        trade_weight=float(trade.trade_weight),
        post_trade_weight=float(trade.post_trade_weight),
        trade_value=float(trade.trade_value),
        estimated_tax=float(trade.estimated_tax),
        estimated_transaction_cost=float(
            trade.estimated_transaction_cost
        ),
        threshold_breached=trade.threshold_breached,
        threshold_severity=trade.threshold_severity,
        breach_ratio=float(trade.breach_ratio),
        final_trigger_type=trade.final_trigger_type,
        final_priority=trade.final_priority,
        contributing_triggers=trade.contributing_triggers,
        client_explanation=trade.client_explanation,
        advisor_explanation=trade.advisor_explanation,
        compliance_explanation=(
            trade.compliance_explanation
        ),
        created_at=trade.created_at,
        approval=approval,
    )


def _rebalance_audit_entry_response(
    audit_entry: Any,
) -> RebalanceAuditEntryResponse:
    """Convert a rebalance audit DTO into an API response."""

    return RebalanceAuditEntryResponse(
        audit_id=audit_entry.audit_id,
        approval_status=audit_entry.approval_status,
        timestamp=audit_entry.timestamp,
        event_type=audit_entry.event_type,
        audit_message=audit_entry.audit_message,
        asset=audit_entry.asset,
        action=audit_entry.action,
        approval_reason=audit_entry.approval_reason,
        reviewed_by=audit_entry.reviewed_by,
        reviewed_at=audit_entry.reviewed_at,
    )


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


@router.post(
    "/backtests/buy-and-hold",
    response_model=BuyAndHoldBacktestResponse,
    tags=["Backtesting"],
)
def run_buy_and_hold_backtest_endpoint(
    request: BuyAndHoldBacktestRequest,
    backtest_runner: BuyAndHoldBacktestProtocol = Depends(
        get_buy_and_hold_backtest_runner
    ),
) -> BuyAndHoldBacktestResponse:
    """
    Run a deterministic Buy & Hold backtest.

    The API layer adapts request data into a DataFrame and delegates all
    backtest calculations to the existing deterministic engine.
    """

    if len(request.asset_names) != len(
        request.initial_weights
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "asset_names and initial_weights must have "
                "the same length."
            ),
        )

    invalid_rows = [
        row
        for row in request.market_returns
        if len(row) != len(request.asset_names)
    ]

    if invalid_rows:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "Each market return row must contain one value "
                "for every asset."
            ),
        )

    market_returns = pd.DataFrame(
        request.market_returns,
        columns=request.asset_names,
    )

    try:
        result = backtest_runner(
            initial_weights=request.initial_weights,
            market_returns=market_returns,
            initial_portfolio_value=(
                request.initial_portfolio_value
            ),
            risk_free_rate=request.risk_free_rate,
            periods_per_year=request.periods_per_year,
        )
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error

    history_records = _serialize_dataframe(
        result.portfolio_history
    )

    return BuyAndHoldBacktestResponse(
        strategy_name="Buy & Hold",
        metrics=BacktestMetricsResponse(
            total_return=result.total_return,
            annualized_return=result.annualized_return,
            volatility=result.volatility,
            sharpe_ratio=result.sharpe_ratio,
            maximum_drawdown=result.maximum_drawdown,
        ),
        portfolio_history=history_records,
        history_record_count=len(history_records),
    )


@router.post(
    "/backtests/threshold-rebalancing",
    response_model=ThresholdRebalancingBacktestResponse,
    tags=["Backtesting"],
)
def run_threshold_rebalancing_backtest_endpoint(
    request: ThresholdRebalancingBacktestRequest,
    backtest_runner: ThresholdRebalancingBacktestProtocol = Depends(
        get_threshold_rebalancing_backtest_runner
    ),
) -> ThresholdRebalancingBacktestResponse:
    """
    Run a deterministic Threshold Rebalancing backtest.

    FastAPI only adapts request lists into DataFrame and NumPy inputs.
    The existing backtest engine performs all threshold, trade, tax, and
    performance calculations.
    """

    _validate_weight_lengths(
        request.asset_names,
        initial_weights=request.initial_weights,
        target_weights=request.target_weights,
    )
    market_returns = _build_market_returns(
        asset_names=request.asset_names,
        market_return_rows=request.market_returns,
    )
    covariance_matrix = (
        None
        if request.covariance_matrix is None
        else np.asarray(request.covariance_matrix, dtype=float)
    )

    try:
        result = backtest_runner(
            initial_weights=request.initial_weights,
            target_weights=request.target_weights,
            market_returns=market_returns,
            initial_portfolio_value=(
                request.initial_portfolio_value
            ),
            drift_band=request.drift_band,
            transaction_cost_rate=request.transaction_cost_rate,
            tax_rate=request.tax_rate,
            turnover_budget=request.turnover_budget,
            risk_free_rate=request.risk_free_rate,
            periods_per_year=request.periods_per_year,
            portfolio_id=request.portfolio_id,
            covariance_matrix=covariance_matrix,
        )
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error

    history_records = _serialize_dataframe(
        result.portfolio_history
    )

    return ThresholdRebalancingBacktestResponse(
        strategy_name=THRESHOLD_REBALANCING_NAME,
        metrics=_metrics_response_from_result(result),
        portfolio_history=history_records,
        history_record_count=len(history_records),
    )


@router.post(
    "/strategy-comparisons",
    response_model=StrategyComparisonResponse,
    tags=["Backtesting"],
)
def compare_strategies_endpoint(
    request: StrategyComparisonRequest,
    comparison_runner: StrategyComparisonProtocol = Depends(
        get_strategy_comparison_runner
    ),
) -> StrategyComparisonResponse:
    """Compare existing Buy & Hold and Threshold Rebalancing results."""

    buy_and_hold_result = SimpleNamespace(
        total_return=request.buy_and_hold.total_return,
        annualized_return=request.buy_and_hold.annualized_return,
        volatility=request.buy_and_hold.volatility,
        sharpe_ratio=request.buy_and_hold.sharpe_ratio,
        maximum_drawdown=request.buy_and_hold.maximum_drawdown,
    )
    threshold_result = SimpleNamespace(
        total_return=request.threshold_rebalancing.total_return,
        annualized_return=(
            request.threshold_rebalancing.annualized_return
        ),
        volatility=request.threshold_rebalancing.volatility,
        sharpe_ratio=request.threshold_rebalancing.sharpe_ratio,
        maximum_drawdown=(
            request.threshold_rebalancing.maximum_drawdown
        ),
        total_transaction_costs=(
            request.threshold_rebalancing.transaction_costs
        ),
        total_taxes_paid=request.threshold_rebalancing.taxes_paid,
        number_of_rebalances=(
            request.threshold_rebalancing.number_of_rebalances
        ),
    )

    try:
        result = comparison_runner(
            buy_and_hold_result=buy_and_hold_result,
            threshold_rebalancing_result=threshold_result,
        )
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error

    return StrategyComparisonResponse(
        buy_and_hold=_strategy_metrics_response(
            result.buy_and_hold
        ),
        threshold_rebalancing=_strategy_metrics_response(
            result.threshold_rebalancing
        ),
        performance_summary=result.performance_summary,
    )


@router.post(
    "/portfolio-analysis",
    response_model=PortfolioAnalysisEndpointResponse,
    tags=["Analysis"],
)
def analyze_portfolios_endpoint(
    request: RebalanceRequest,
    orchestrator: OrchestratorAgent = Depends(
        get_orchestrator_agent
    ),
    analyst: PortfolioAnalystAgent = Depends(
        get_portfolio_analyst_agent
    ),
) -> PortfolioAnalysisEndpointResponse:
    """Run deterministic rebalancing and return portfolio analysis."""

    orchestrator_request = OrchestratorRequest(
        number_of_clients=request.number_of_clients,
        evaluation_date=request.evaluation_date,
        portfolio_value=request.portfolio_value,
        transaction_cost_rate=request.transaction_cost_rate,
    )
    workflow_result = orchestrator.execute_rebalance(
        orchestrator_request
    )

    if workflow_result.status == AgentExecutionStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The portfolio analysis workflow failed.",
        )

    if workflow_result.result is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "The portfolio analysis workflow completed "
                "without a rebalance result."
            ),
        )

    try:
        analyses = analyst.analyze(workflow_result.result)
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error

    analysis_records = [
        _analysis_response(analysis)
        for analysis in analyses
    ]

    return PortfolioAnalysisEndpointResponse(
        analyses=analysis_records,
        analysis_count=len(analysis_records),
    )


@router.post(
    "/portfolio-explanations",
    response_model=PortfolioExplanationsResponse,
    tags=["Analysis"],
)
def explain_portfolios_endpoint(
    request: PortfolioExplanationsRequest,
    language_model: LanguageModelProtocol | None = Depends(
        get_optional_language_model
    ),
) -> PortfolioExplanationsResponse:
    """Generate portfolio explanations from supplied analysis facts."""

    analyses = [
        _analysis_from_payload(payload)
        for payload in request.analyses
    ]
    agent = ExplanationAgent(
        language_model=language_model,
    )

    try:
        explanations = agent.explain(analyses)
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error

    explanation_records = [
        _explanation_response(explanation)
        for explanation in explanations
    ]

    return PortfolioExplanationsResponse(
        explanations=explanation_records,
        portfolio_count=len(explanation_records),
    )


@router.get(
    "/llm/health",
    response_model=LlmHealthResponse,
    tags=["LLM"],
)
def llm_health_endpoint(
    health_status: dict[str, object] = Depends(
        get_llm_health_status
    ),
) -> LlmHealthResponse:
    """Return safe language-model configuration health."""

    return LlmHealthResponse(
        status=str(health_status["status"]),
        provider=str(health_status["provider"]),
        model_name=str(health_status["model_name"]),
        configured=bool(health_status["configured"]),
        live_check_performed=bool(
            health_status["live_check_performed"]
        ),
    )


@router.post(
    "/llm/prompts/preview",
    response_model=PromptPreviewResponse,
    tags=["LLM"],
)
def preview_prompt_endpoint(
    request: PromptPreviewRequest,
    enabled: bool = Depends(is_prompt_preview_enabled),
    prompt_builder: PromptBuilder = Depends(get_prompt_builder),
) -> PromptPreviewResponse:
    """
    Preview a provider-neutral prompt for development.

    This endpoint never initializes Gemini and never calls a language
    model provider.
    """

    if not enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prompt preview is disabled.",
        )

    try:
        language_model_request = prompt_builder.build(
            analysis=_analysis_from_payload(request.analysis),
            audience=request.audience,
        )
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error

    return PromptPreviewResponse(
        audience=request.audience.lower(),
        system_prompt=language_model_request.system_prompt,
        user_prompt=language_model_request.user_prompt,
        temperature=float(language_model_request.temperature),
        max_output_tokens=(
            language_model_request.max_output_tokens
        ),
    )


@router.get(
    "/portfolios",
    response_model=PortfolioListResponse,
    tags=["Portfolios"],
)
def list_portfolios_endpoint(
    limit: int = Query(
        default=DEFAULT_PAGE_LIMIT,
        ge=1,
        le=MAX_PAGE_LIMIT,
    ),
    offset: int = Query(
        default=DEFAULT_PAGE_OFFSET,
        ge=0,
    ),
    service: PortfolioReadApplicationService = Depends(
        get_portfolio_read_application_service
    ),
) -> PortfolioListResponse:
    """Return all persisted portfolios."""

    page = service.list_portfolios(
        limit=limit,
        offset=offset,
    )
    items = [
        _portfolio_summary_response(portfolio)
        for portfolio in page.items
    ]

    return PortfolioListResponse(
        items=items,
        limit=page.limit,
        offset=page.offset,
        count=page.count,
    )


@router.get(
    "/portfolios/{portfolio_id}",
    response_model=PortfolioDetailResponse,
    tags=["Portfolios"],
)
def get_portfolio_endpoint(
    portfolio_id: str,
    service: PortfolioReadApplicationService = Depends(
        get_portfolio_read_application_service
    ),
) -> PortfolioDetailResponse:
    """Return one persisted portfolio with holdings."""

    portfolio = service.get_portfolio(portfolio_id)

    return _portfolio_detail_response(portfolio)


@router.get(
    "/portfolios/{portfolio_id}/rebalances",
    response_model=PortfolioRebalanceListResponse,
    tags=["Rebalancing"],
)
def list_portfolio_rebalances_endpoint(
    portfolio_id: str,
    limit: int = Query(
        default=DEFAULT_PAGE_LIMIT,
        ge=1,
        le=MAX_PAGE_LIMIT,
    ),
    offset: int = Query(
        default=DEFAULT_PAGE_OFFSET,
        ge=0,
    ),
    service: PortfolioReadApplicationService = Depends(
        get_portfolio_read_application_service
    ),
) -> PortfolioRebalanceListResponse:
    """Return rebalance runs for one persisted portfolio."""

    page = service.list_portfolio_rebalances(
        portfolio_id,
        limit=limit,
        offset=offset,
    )
    items = [
        _rebalance_summary_response(rebalance)
        for rebalance in page.items
    ]

    return PortfolioRebalanceListResponse(
        items=items,
        limit=page.limit,
        offset=page.offset,
        count=page.count,
    )


@router.get(
    "/rebalances/{run_id}",
    response_model=RebalanceRunDetailResponse,
    tags=["Rebalancing"],
)
def get_rebalance_endpoint(
    run_id: str,
    service: PortfolioReadApplicationService = Depends(
        get_portfolio_read_application_service
    ),
) -> RebalanceRunDetailResponse:
    """Return one persisted rebalance run."""

    rebalance = service.get_rebalance(run_id)

    return _rebalance_detail_response(rebalance)


@router.get(
    "/rebalances/{run_id}/trades",
    response_model=RebalanceTradeListResponse,
    tags=["Rebalancing"],
)
def list_rebalance_trades_endpoint(
    run_id: str,
    limit: int = Query(
        default=DEFAULT_PAGE_LIMIT,
        ge=1,
        le=MAX_PAGE_LIMIT,
    ),
    offset: int = Query(
        default=DEFAULT_PAGE_OFFSET,
        ge=0,
    ),
    service: PortfolioReadApplicationService = Depends(
        get_portfolio_read_application_service
    ),
) -> RebalanceTradeListResponse:
    """Return persisted trades for one rebalance run."""

    page = service.list_rebalance_trades(
        run_id,
        limit=limit,
        offset=offset,
    )
    items = [
        _rebalance_trade_response(trade)
        for trade in page.items
    ]

    return RebalanceTradeListResponse(
        items=items,
        limit=page.limit,
        offset=page.offset,
        count=page.count,
    )


@router.get(
    "/rebalances/{run_id}/audit",
    response_model=RebalanceAuditResponse,
    tags=["Rebalancing"],
)
def list_rebalance_audit_endpoint(
    run_id: str,
    limit: int = Query(
        default=DEFAULT_PAGE_LIMIT,
        ge=1,
        le=MAX_PAGE_LIMIT,
    ),
    offset: int = Query(
        default=DEFAULT_PAGE_OFFSET,
        ge=0,
    ),
    service: PortfolioReadApplicationService = Depends(
        get_portfolio_read_application_service
    ),
) -> RebalanceAuditResponse:
    """Return persisted audit entries for one rebalance run."""

    page = service.list_rebalance_audit(
        run_id,
        limit=limit,
        offset=offset,
    )
    items = [
        _rebalance_audit_entry_response(entry)
        for entry in page.items
    ]

    return RebalanceAuditResponse(
        items=items,
        limit=page.limit,
        offset=page.offset,
        count=page.count,
    )


@router.post(
    "/portfolios/{portfolio_id}/rebalance",
    response_model=DatabaseRebalanceResponse,
    tags=["Rebalancing"],
    summary="Rebalance and persist an existing portfolio",
)
def rebalance_stored_portfolio(
    portfolio_id: str,
    request: DatabaseRebalanceRequest,
    service: RebalanceApplicationService = Depends(
        get_rebalance_application_service
    ),
) -> DatabaseRebalanceResponse:
    """Run and persist rebalancing for a stored portfolio."""

    try:
        result = service.execute_rebalance(
            portfolio_id=portfolio_id,
            transaction_cost_rate=Decimal(
                str(request.transaction_cost_rate)
            ),
        )
    except (
        RebalanceExecutionError,
        RebalancePersistenceError,
    ) as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The rebalance workflow could not be completed.",
        ) from error

    return DatabaseRebalanceResponse(
        status=result.workflow_status,
        portfolio_id=result.portfolio_id,
        run_id=result.run_id,
        trade_count=result.trade_count,
        database_run_id=result.database_run_id,
        message=result.workflow_message,
    )
