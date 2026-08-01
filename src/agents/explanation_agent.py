from __future__ import annotations

from dataclasses import dataclass

from src.agents.portfolio_analyst_agent import (
    PortfolioAnalysis,
)


@dataclass(frozen=True, slots=True)
class PortfolioExplanation:
    """
    Store audience-specific portfolio communication.

    All portfolio facts come from PortfolioAnalysis. This object contains
    only communication produced from those existing facts.
    """

    portfolio_id: object
    client_summary: str
    advisor_summary: str
    compliance_summary: str


class ExplanationAgent:
    """
    Convert structured portfolio analysis into audience-specific summaries.

    The agent does not:

    - Calculate drift
    - Evaluate triggers
    - Generate trades
    - Calculate transaction costs
    - Calculate taxes
    - Count threshold breaches
    - Determine threshold severity
    - Group raw trade rows
    - Generate individual trade explanations

    Those responsibilities belong to upstream deterministic modules and
    the Portfolio Analyst Agent.
    """

    def explain(
        self,
        analyses: list[PortfolioAnalysis],
    ) -> list[PortfolioExplanation]:
        """
        Generate one explanation package for each portfolio analysis.

        Args:
            analyses:
                Structured results returned by PortfolioAnalystAgent.

        Returns:
            One PortfolioExplanation object per portfolio.

        Raises:
            TypeError:
                If analyses is not a list or contains invalid objects.
        """

        _validate_analyses(analyses)

        return [
            _build_portfolio_explanation(analysis)
            for analysis in analyses
        ]


def _validate_analyses(
    analyses: list[PortfolioAnalysis],
) -> None:
    """Validate PortfolioAnalysis inputs."""

    if not isinstance(analyses, list):
        raise TypeError(
            "Analyses must be provided as a list."
        )

    invalid_analyses = [
        analysis
        for analysis in analyses
        if not isinstance(analysis, PortfolioAnalysis)
    ]

    if invalid_analyses:
        raise TypeError(
            "Analyses must contain PortfolioAnalysis objects."
        )


def _build_portfolio_explanation(
    analysis: PortfolioAnalysis,
) -> PortfolioExplanation:
    """Build one audience-specific explanation package."""

    return PortfolioExplanation(
        portfolio_id=analysis.portfolio_id,
        client_summary=_build_client_summary(analysis),
        advisor_summary=_build_advisor_summary(analysis),
        compliance_summary=_build_compliance_summary(analysis),
    )


def _build_client_summary(
    analysis: PortfolioAnalysis,
) -> str:
    """Build a simple client-facing portfolio summary."""

    if not analysis.rebalance_required:
        return (
            f"Portfolio {analysis.portfolio_id} does not require "
            "rebalancing. All recommended positions remain unchanged."
        )

    return (
        f"Portfolio {analysis.portfolio_id} requires rebalancing. "
        f"The plan increases "
        f"{_format_asset_list(analysis.assets_to_buy)} and decreases "
        f"{_format_asset_list(analysis.assets_to_sell)}. "
        "The estimated transaction cost is "
        f"{_format_currency(analysis.total_transaction_cost)}, "
        "and the estimated tax liability is "
        f"{_format_currency(
            analysis.total_estimated_tax_liability
        )}."
    )


def _build_advisor_summary(
    analysis: PortfolioAnalysis,
) -> str:
    """Build a detailed advisor-facing portfolio summary."""

    recommendation = (
        "Rebalancing is recommended."
        if analysis.rebalance_required
        else "Rebalancing is not required."
    )

    return (
        f"Portfolio {analysis.portfolio_id}: {recommendation} "
        f"Threshold breached: {analysis.threshold_breached}. "
        f"Threshold breach count: "
        f"{analysis.threshold_breach_count}. "
        f"Highest threshold severity: "
        f"{analysis.highest_threshold_severity}. "
        f"Assets to buy: "
        f"{_format_asset_list(analysis.assets_to_buy)}. "
        f"Assets to sell: "
        f"{_format_asset_list(analysis.assets_to_sell)}. "
        f"Assets to hold: "
        f"{_format_asset_list(analysis.assets_to_hold)}. "
        "Total estimated transaction cost: "
        f"{_format_currency(analysis.total_transaction_cost)}. "
        "Total estimated tax liability: "
        f"{_format_currency(
            analysis.total_estimated_tax_liability
        )}."
    )


def _build_compliance_summary(
    analysis: PortfolioAnalysis,
) -> str:
    """Build a traceable compliance-facing portfolio summary."""

    return (
        "Portfolio-level explanation generated from deterministic "
        "portfolio analysis. "
        f"Portfolio ID: {analysis.portfolio_id}. "
        f"Rebalancing required: {analysis.rebalance_required}. "
        f"Threshold breached: {analysis.threshold_breached}. "
        f"Threshold breach count: "
        f"{analysis.threshold_breach_count}. "
        f"Highest threshold severity: "
        f"{analysis.highest_threshold_severity}. "
        "No financial calculations or trade decisions were performed "
        "inside the Explanation Agent."
    )


def _format_asset_list(
    assets: tuple[str, ...],
) -> str:
    """Format internal asset names for readable communication."""

    if not assets:
        return "none"

    return ", ".join(
        asset.replace("_", " ")
        for asset in assets
    )


def _format_currency(
    value: float,
) -> str:
    """Format a monetary value for communication."""

    return f"${value:,.2f}"