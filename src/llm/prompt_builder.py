from __future__ import annotations

from enum import StrEnum

from src.agents.portfolio_analyst_agent import (
    PortfolioAnalysis,
)
from src.llm.language_model import (
    LanguageModelRequest,
)


class PromptAudience(StrEnum):
    """Supported audiences for portfolio explanation prompts."""

    CLIENT = "client"
    ADVISOR = "advisor"
    COMPLIANCE = "compliance"


TOKEN_LIMITS = {
    PromptAudience.CLIENT: 1_200,
    PromptAudience.ADVISOR: 1_800,
    PromptAudience.COMPLIANCE: 1_800,
}

GROUNDING_INSTRUCTIONS = (
    "Use only the supplied facts. "
    "Do not invent or change any number. "
    "Do not calculate financial values. "
    "Do not provide new investment advice. "
    "If a fact is unavailable, state that it is unavailable."
)


class PromptBuilder:
    """
    Build provider-neutral language model requests from portfolio analysis.

    The builder converts structured facts into prompts. It does not call a
    language model, inspect raw trade DataFrames, or calculate financial
    values.
    """

    def build(
        self,
        analysis: PortfolioAnalysis,
        audience: PromptAudience | str,
    ) -> LanguageModelRequest:
        """
        Build a language model request for one audience.

        Args:
            analysis:
                Structured deterministic facts from PortfolioAnalystAgent.
            audience:
                Target communication audience.

        Returns:
            A validated provider-independent language model request.

        Raises:
            TypeError:
                If analysis is not a PortfolioAnalysis.
            ValueError:
                If audience is unsupported.
        """

        _validate_analysis(analysis)
        normalized_audience = _normalize_audience(audience)

        system_prompt = _build_system_prompt(normalized_audience)
        user_prompt = _build_user_prompt(
            analysis=analysis,
            audience=normalized_audience,
        )

        return LanguageModelRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            max_output_tokens=TOKEN_LIMITS[normalized_audience],
        )


def _validate_analysis(
    analysis: PortfolioAnalysis,
) -> None:
    """Validate the public analysis boundary."""

    if not isinstance(analysis, PortfolioAnalysis):
        raise TypeError(
            "analysis must be a PortfolioAnalysis."
        )


def _normalize_audience(
    audience: PromptAudience | str,
) -> PromptAudience:
    """Normalize and validate audience values in one place."""

    if isinstance(audience, PromptAudience):
        return audience

    if isinstance(audience, str):
        try:
            return PromptAudience(audience.lower())
        except ValueError as error:
            raise ValueError(
                "audience must be one of client, advisor, or compliance."
            ) from error

    raise TypeError(
        "audience must be a PromptAudience or string."
    )


def _build_system_prompt(
    audience: PromptAudience,
) -> str:
    """Build the system prompt for one audience."""

    if audience is PromptAudience.CLIENT:
        return (
            "You are a careful financial communication assistant. "
            "Write in simple, client-friendly language. Avoid unnecessary "
            "technical jargon. Write one complete paragraph in plain text "
            "only. Do not use Markdown, headings, bullet points, bold text, "
            "or tables. Write approximately 80 to 120 words. Never stop in "
            "the middle of a sentence. Explain the rebalance from the "
            f"supplied facts only. {GROUNDING_INSTRUCTIONS} Never change "
            "recommendations."
        )

    if audience is PromptAudience.ADVISOR:
        return (
            "You are a portfolio advisory communication assistant. "
            "Use concise technical language suitable for an advisor. "
            "Preserve every supplied value exactly and explain only from "
            f"the provided facts. {GROUNDING_INSTRUCTIONS}"
        )

    return (
        "You are a compliance communication assistant. Produce factual, "
        "traceable language. Calculations came from deterministic "
        "portfolio modules. Do not alter, infer, or calculate financial "
        f"values. {GROUNDING_INSTRUCTIONS}"
    )


def _build_user_prompt(
    analysis: PortfolioAnalysis,
    audience: PromptAudience,
) -> str:
    """Build the user prompt from structured portfolio facts."""

    facts = _format_analysis_facts(analysis)
    explanations = _format_trade_explanations(
        explanations=_select_explanations(
            analysis=analysis,
            audience=audience,
        ),
    )

    if audience is PromptAudience.CLIENT:
        task = (
            "Create a client-facing portfolio explanation. Write one "
            "complete paragraph. Use plain text only. Do not use Markdown, "
            "headings, bullet points, bold text, or tables. Write "
            "approximately 80 to 120 words. Never stop in the middle of a "
            "sentence. Use simple language and do not add investment "
            "advice. Never change recommendations."
        )
    elif audience is PromptAudience.ADVISOR:
        task = (
            "Create an advisor-facing portfolio explanation. Include "
            "severity, threshold status, asset actions, costs, and taxes."
        )
    else:
        task = (
            "Create a compliance-facing portfolio explanation. Make it "
            "factual and traceable, and do not provide investment advice."
        )

    return (
        f"{task}\n\n"
        f"Grounding instructions:\n{GROUNDING_INSTRUCTIONS}\n\n"
        f"Supplied facts:\n{facts}\n\n"
        f"Existing {audience.value} trade-level explanations:\n"
        f"{explanations}"
    )


def _format_analysis_facts(
    analysis: PortfolioAnalysis,
) -> str:
    """Format deterministic portfolio facts for a prompt."""

    return "\n".join(
        [
            f"- Portfolio ID: {analysis.portfolio_id}",
            (
                "- Rebalance required: "
                f"{_format_bool(analysis.rebalance_required)}"
            ),
            (
                "- Threshold breached: "
                f"{_format_bool(analysis.threshold_breached)}"
            ),
            (
                "- Threshold breach count: "
                f"{analysis.threshold_breach_count}"
            ),
            (
                "- Highest threshold severity: "
                f"{analysis.highest_threshold_severity}"
            ),
            (
                "- Assets to buy: "
                f"{_format_asset_list(analysis.assets_to_buy)}"
            ),
            (
                "- Assets to sell: "
                f"{_format_asset_list(analysis.assets_to_sell)}"
            ),
            (
                "- Assets to hold: "
                f"{_format_asset_list(analysis.assets_to_hold)}"
            ),
            (
                "- Total transaction cost: "
                f"{_format_currency(analysis.total_transaction_cost)}"
            ),
            (
                "- Total estimated tax liability: "
                f"{_format_currency(
                    analysis.total_estimated_tax_liability
                )}"
            ),
        ]
    )


def _select_explanations(
    analysis: PortfolioAnalysis,
    audience: PromptAudience,
) -> tuple[str, ...]:
    """Select existing trade-level explanations for the audience."""

    if audience is PromptAudience.CLIENT:
        return analysis.client_explanations

    if audience is PromptAudience.ADVISOR:
        return analysis.advisor_explanations

    return analysis.compliance_explanations


def _format_trade_explanations(
    explanations: tuple[str, ...],
) -> str:
    """Format existing trade-level explanations."""

    if not explanations:
        return "none"

    return "\n".join(
        f"- {explanation}"
        for explanation in explanations
    )


def _format_asset_list(
    assets: tuple[str, ...],
) -> str:
    """Format internal asset names for provider-neutral prompts."""

    if not assets:
        return "none"

    return ", ".join(
        asset.replace("_", " ")
        for asset in assets
    )


def _format_bool(
    value: bool,
) -> str:
    """Format Boolean values consistently."""

    return "yes" if value else "no"


def _format_currency(
    value: float,
) -> str:
    """Format currency without changing the underlying fact."""

    return f"${value:,.2f}"
