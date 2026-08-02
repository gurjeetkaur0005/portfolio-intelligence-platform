from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Represent the API health-check response."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    status: str
    service: str


class RebalanceRequest(BaseModel):
    """
    Represent an API request to run portfolio rebalancing.

    Validation happens at the API boundary before the request reaches
    the Orchestrator Agent.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    number_of_clients: int = Field(
        default=1,
        gt=0,
        description=(
            "Number of synthetic client portfolios to process."
        ),
    )

    evaluation_date: date | None = Field(
        default=None,
        description=(
            "Date used for calendar and event-trigger evaluation."
        ),
    )

    portfolio_value: float = Field(
        default=1_000_000.0,
        gt=0.0,
        description=(
            "Starting monetary value assigned to each portfolio."
        ),
    )

    transaction_cost_rate: float = Field(
        default=0.002,
        ge=0.0,
        le=1.0,
        description=(
            "Transaction-cost rate expressed as a decimal."
        ),
    )


class TradeRecordResponse(BaseModel):
    """
    Represent one JSON-safe row returned by the rebalance pipeline.

    The pipeline currently returns a pandas DataFrame. The API layer
    converts every DataFrame row into this JSON-compatible structure.
    """

    model_config = ConfigDict(
        extra="allow",
        frozen=True,
    )

    portfolio_id: Any
    asset: str
    action: str


class RebalanceResponse(BaseModel):
    """Represent a successful rebalance API response."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    status: str
    workflow_name: str
    message: str
    records: list[dict[str, Any]]
    record_count: int


class ErrorResponse(BaseModel):
    """Represent a safe API error response."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    error: str
    detail: str


class PortfolioExplanationResponse(BaseModel):
    """Represent one portfolio-level explanation package."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    portfolio_id: str
    client_summary: str
    advisor_summary: str
    compliance_summary: str


class RebalanceExplanationResponse(BaseModel):
    """Represent a successful AI-assisted rebalance response."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    status: str
    workflow_name: str
    message: str
    explanations: list[PortfolioExplanationResponse]
    portfolio_count: int


class BuyAndHoldBacktestRequest(BaseModel):
    """Represent a Buy & Hold backtest API request."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    asset_names: list[str] = Field(
        min_length=1,
        description=(
            "Asset names matching the market-return columns."
        ),
    )

    market_returns: list[list[float]] = Field(
        min_length=1,
        description=(
            "Return rows where each inner list contains "
            "one return per asset."
        ),
    )

    initial_weights: list[float] = Field(
        min_length=1,
        description=(
            "Initial portfolio weights in the same order "
            "as asset_names."
        ),
    )

    initial_portfolio_value: float = Field(
        default=100_000.0,
        gt=0.0,
    )

    risk_free_rate: float = 0.0

    periods_per_year: int = Field(
        default=252,
        gt=0,
    )


class BacktestMetricsResponse(BaseModel):
    """Represent deterministic backtest performance metrics."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    maximum_drawdown: float


class BuyAndHoldBacktestResponse(BaseModel):
    """Represent a completed Buy & Hold backtest."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    strategy_name: str
    metrics: BacktestMetricsResponse
    portfolio_history: list[dict[str, Any]]
    history_record_count: int


class ThresholdRebalancingBacktestRequest(BaseModel):
    """Represent a Threshold Rebalancing backtest API request."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    asset_names: list[str] = Field(
        min_length=1,
        description=(
            "Asset names matching the market-return columns."
        ),
    )
    market_returns: list[list[float]] = Field(
        min_length=1,
        description=(
            "Return rows where each inner list contains one "
            "return per asset."
        ),
    )
    initial_weights: list[float] = Field(
        min_length=1,
        description="Initial portfolio weights.",
    )
    target_weights: list[float] = Field(
        min_length=1,
        description="Target portfolio weights.",
    )
    initial_portfolio_value: float = Field(
        default=100_000.0,
        gt=0.0,
    )
    drift_band: float = Field(
        default=0.05,
        gt=0.0,
    )
    transaction_cost_rate: float = Field(
        default=0.002,
        ge=0.0,
    )
    tax_rate: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
    )
    turnover_budget: float = Field(
        default=0.10,
        gt=0.0,
    )
    risk_free_rate: float = 0.0
    periods_per_year: int = Field(
        default=252,
        gt=0,
    )
    portfolio_id: str = Field(
        default="BACKTEST",
        min_length=1,
    )
    covariance_matrix: list[list[float]] | None = Field(
        default=None,
        description=(
            "Optional square covariance matrix matching asset_names."
        ),
    )


class ThresholdRebalancingBacktestResponse(BaseModel):
    """Represent a completed Threshold Rebalancing backtest."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    strategy_name: str
    metrics: BacktestMetricsResponse
    portfolio_history: list[dict[str, Any]]
    history_record_count: int


class StrategyMetricsRequest(BaseModel):
    """Represent strategy metrics supplied for comparison."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    maximum_drawdown: float
    transaction_costs: float = Field(default=0.0, ge=0.0)
    taxes_paid: float = Field(default=0.0, ge=0.0)
    number_of_rebalances: int = Field(default=0, ge=0)


class StrategyComparisonRequest(BaseModel):
    """Represent a request to compare two strategy results."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    buy_and_hold: StrategyMetricsRequest
    threshold_rebalancing: StrategyMetricsRequest


class StrategyMetricsResponse(BaseModel):
    """Represent normalized strategy comparison metrics."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    strategy_name: str
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    maximum_drawdown: float
    transaction_costs: float
    taxes_paid: float
    number_of_rebalances: int
    total_implementation_cost: float


class StrategyComparisonResponse(BaseModel):
    """Represent a deterministic strategy comparison."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    buy_and_hold: StrategyMetricsResponse
    threshold_rebalancing: StrategyMetricsResponse
    performance_summary: str


class PortfolioAnalysisResponse(BaseModel):
    """Represent one deterministic portfolio analysis."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    portfolio_id: str
    rebalance_required: bool
    highest_threshold_severity: str
    threshold_breached: bool
    threshold_breach_count: int
    assets_to_buy: list[str]
    assets_to_sell: list[str]
    assets_to_hold: list[str]
    total_transaction_cost: float
    total_estimated_tax_liability: float
    client_explanations: list[str]
    advisor_explanations: list[str]
    compliance_explanations: list[str]


class PortfolioAnalysisApiPayload(BaseModel):
    """Represent PortfolioAnalysis facts in JSON-compatible form."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    portfolio_id: str
    rebalance_required: bool
    highest_threshold_severity: str
    threshold_breached: bool
    threshold_breach_count: int = Field(ge=0)
    assets_to_buy: list[str] = Field(default_factory=list)
    assets_to_sell: list[str] = Field(default_factory=list)
    assets_to_hold: list[str] = Field(default_factory=list)
    total_transaction_cost: float = Field(ge=0.0)
    total_estimated_tax_liability: float = Field(ge=0.0)
    client_explanations: list[str] = Field(default_factory=list)
    advisor_explanations: list[str] = Field(default_factory=list)
    compliance_explanations: list[str] = Field(default_factory=list)


class PortfolioAnalysisEndpointResponse(BaseModel):
    """Represent portfolio-analysis endpoint output."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    analyses: list[PortfolioAnalysisResponse]
    analysis_count: int


class PortfolioExplanationsRequest(BaseModel):
    """Represent a request for portfolio explanations."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    analyses: list[PortfolioAnalysisApiPayload] = Field(
        min_length=1,
    )


class PortfolioExplanationsResponse(BaseModel):
    """Represent generated portfolio explanations."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    explanations: list[PortfolioExplanationResponse]
    portfolio_count: int


class LlmHealthResponse(BaseModel):
    """Represent safe LLM configuration health."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    status: str
    provider: str
    model_name: str
    configured: bool
    live_check_performed: bool


class PromptPreviewRequest(BaseModel):
    """Represent a development-only prompt preview request."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    analysis: PortfolioAnalysisApiPayload
    audience: str = Field(
        description="Prompt audience: client, advisor, or compliance.",
    )


class PromptPreviewResponse(BaseModel):
    """Represent a provider-neutral prompt preview."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    audience: str
    system_prompt: str
    user_prompt: str
    temperature: float
    max_output_tokens: int
