from __future__ import annotations

from dataclasses import replace

import pytest

from src.agents.portfolio_analyst_agent import (
    PortfolioAnalysis,
)
from src.llm.language_model import (
    LanguageModelRequest,
)
from src.llm.prompt_builder import (
    PromptAudience,
    PromptBuilder,
)


def _build_analysis(
    *,
    assets_to_buy: tuple[str, ...] = (
        "domestic_equity",
        "international_equity",
    ),
    assets_to_sell: tuple[str, ...] = ("fixed_income",),
    assets_to_hold: tuple[str, ...] = ("cash",),
) -> PortfolioAnalysis:
    """Build a real PortfolioAnalysis fixture."""

    return PortfolioAnalysis(
        portfolio_id="portfolio-123",
        rebalance_required=True,
        highest_threshold_severity="high",
        threshold_breached=True,
        threshold_breach_count=2,
        assets_to_buy=assets_to_buy,
        assets_to_sell=assets_to_sell,
        assets_to_hold=assets_to_hold,
        total_transaction_cost=125.5,
        total_estimated_tax_liability=42.75,
        client_explanations=(
            "Client explanation for domestic equity.",
        ),
        advisor_explanations=(
            "Advisor explanation with trade details.",
        ),
        compliance_explanations=(
            "Compliance explanation with traceable controls.",
        ),
    )


def test_client_prompt_returns_language_model_request() -> None:
    request = PromptBuilder().build(
        analysis=_build_analysis(),
        audience=PromptAudience.CLIENT,
    )

    assert isinstance(request, LanguageModelRequest)


def test_advisor_prompt_returns_language_model_request() -> None:
    request = PromptBuilder().build(
        analysis=_build_analysis(),
        audience=PromptAudience.ADVISOR,
    )

    assert isinstance(request, LanguageModelRequest)


def test_compliance_prompt_returns_language_model_request() -> None:
    request = PromptBuilder().build(
        analysis=_build_analysis(),
        audience=PromptAudience.COMPLIANCE,
    )

    assert isinstance(request, LanguageModelRequest)


def test_string_audience_is_supported_for_simple_call() -> None:
    request = PromptBuilder().build(
        analysis=_build_analysis(),
        audience="client",
    )

    assert isinstance(request, LanguageModelRequest)
    assert request.max_output_tokens == 300


def test_temperature_is_zero() -> None:
    request = PromptBuilder().build(
        analysis=_build_analysis(),
        audience=PromptAudience.CLIENT,
    )

    assert request.temperature == 0.0


@pytest.mark.parametrize(
    ("audience", "expected_tokens"),
    [
        (PromptAudience.CLIENT, 300),
        (PromptAudience.ADVISOR, 500),
        (PromptAudience.COMPLIANCE, 500),
    ],
)
def test_correct_audience_specific_token_limit(
    audience: PromptAudience,
    expected_tokens: int,
) -> None:
    request = PromptBuilder().build(
        analysis=_build_analysis(),
        audience=audience,
    )

    assert request.max_output_tokens == expected_tokens


def test_portfolio_id_is_included() -> None:
    request = PromptBuilder().build(
        analysis=_build_analysis(),
        audience=PromptAudience.CLIENT,
    )

    assert "portfolio-123" in request.user_prompt


def test_buy_assets_are_included() -> None:
    request = PromptBuilder().build(
        analysis=_build_analysis(),
        audience=PromptAudience.ADVISOR,
    )

    assert "domestic equity" in request.user_prompt
    assert "international equity" in request.user_prompt


def test_sell_assets_are_included() -> None:
    request = PromptBuilder().build(
        analysis=_build_analysis(),
        audience=PromptAudience.ADVISOR,
    )

    assert "fixed income" in request.user_prompt


def test_hold_assets_are_included() -> None:
    request = PromptBuilder().build(
        analysis=_build_analysis(),
        audience=PromptAudience.ADVISOR,
    )

    assert "cash" in request.user_prompt


def test_transaction_cost_is_included_exactly() -> None:
    request = PromptBuilder().build(
        analysis=_build_analysis(),
        audience=PromptAudience.CLIENT,
    )

    assert "$125.50" in request.user_prompt


def test_tax_liability_is_included_exactly() -> None:
    request = PromptBuilder().build(
        analysis=_build_analysis(),
        audience=PromptAudience.CLIENT,
    )

    assert "$42.75" in request.user_prompt


def test_highest_severity_is_included() -> None:
    request = PromptBuilder().build(
        analysis=_build_analysis(),
        audience=PromptAudience.COMPLIANCE,
    )

    assert "Highest threshold severity: high" in request.user_prompt


def test_threshold_breach_count_is_included() -> None:
    request = PromptBuilder().build(
        analysis=_build_analysis(),
        audience=PromptAudience.COMPLIANCE,
    )

    assert "Threshold breach count: 2" in request.user_prompt


def test_grounding_instructions_are_included() -> None:
    request = PromptBuilder().build(
        analysis=_build_analysis(),
        audience=PromptAudience.COMPLIANCE,
    )

    combined_prompt = (
        f"{request.system_prompt}\n{request.user_prompt}"
    )

    assert "Use only the supplied facts" in combined_prompt
    assert "Do not invent or change any number" in combined_prompt
    assert "Do not calculate financial values" in combined_prompt
    assert "Do not provide new investment advice" in combined_prompt
    assert "If a fact is unavailable" in combined_prompt


def test_missing_asset_groups_render_as_none() -> None:
    request = PromptBuilder().build(
        analysis=_build_analysis(
            assets_to_buy=(),
            assets_to_sell=(),
            assets_to_hold=(),
        ),
        audience=PromptAudience.ADVISOR,
    )

    assert "Assets to buy: none" in request.user_prompt
    assert "Assets to sell: none" in request.user_prompt
    assert "Assets to hold: none" in request.user_prompt


def test_internal_asset_names_are_made_readable() -> None:
    request = PromptBuilder().build(
        analysis=_build_analysis(
            assets_to_buy=("emerging_market_equity",),
        ),
        audience=PromptAudience.CLIENT,
    )

    assert "emerging market equity" in request.user_prompt
    assert "emerging_market_equity" not in request.user_prompt


@pytest.mark.parametrize(
    ("audience", "expected_text", "unexpected_text"),
    [
        (
            PromptAudience.CLIENT,
            "Client explanation for domestic equity.",
            "Advisor explanation with trade details.",
        ),
        (
            PromptAudience.ADVISOR,
            "Advisor explanation with trade details.",
            "Compliance explanation with traceable controls.",
        ),
        (
            PromptAudience.COMPLIANCE,
            "Compliance explanation with traceable controls.",
            "Client explanation for domestic equity.",
        ),
    ],
)
def test_existing_audience_specific_trade_explanations_are_included(
    audience: PromptAudience,
    expected_text: str,
    unexpected_text: str,
) -> None:
    request = PromptBuilder().build(
        analysis=_build_analysis(),
        audience=audience,
    )

    assert expected_text in request.user_prompt
    assert unexpected_text not in request.user_prompt


def test_input_portfolio_analysis_is_not_mutated() -> None:
    analysis = _build_analysis()
    original_analysis = replace(analysis)

    PromptBuilder().build(
        analysis=analysis,
        audience=PromptAudience.CLIENT,
    )

    assert analysis == original_analysis


def test_invalid_analysis_raises_type_error() -> None:
    with pytest.raises(
        TypeError,
        match="analysis must be a PortfolioAnalysis",
    ):
        PromptBuilder().build(
            analysis="not analysis",  # type: ignore[arg-type]
            audience=PromptAudience.CLIENT,
        )


def test_invalid_string_audience_raises_value_error() -> None:
    with pytest.raises(
        ValueError,
        match="audience must be one of",
    ):
        PromptBuilder().build(
            analysis=_build_analysis(),
            audience="regulator",
        )


def test_invalid_audience_type_raises_type_error() -> None:
    with pytest.raises(
        TypeError,
        match="audience must be a PromptAudience or string",
    ):
        PromptBuilder().build(
            analysis=_build_analysis(),
            audience=1,  # type: ignore[arg-type]
        )


def test_prompt_generation_is_deterministic() -> None:
    builder = PromptBuilder()
    analysis = _build_analysis()

    first_request = builder.build(
        analysis=analysis,
        audience=PromptAudience.ADVISOR,
    )
    second_request = builder.build(
        analysis=analysis,
        audience=PromptAudience.ADVISOR,
    )

    assert first_request == second_request


def test_prompts_contain_no_unsupported_claims() -> None:
    request = PromptBuilder().build(
        analysis=_build_analysis(),
        audience=PromptAudience.CLIENT,
    )

    combined_prompt = (
        f"{request.system_prompt}\n{request.user_prompt}"
    ).lower()

    unsupported_claims = [
        "market rally",
        "market rallies",
        "life event",
        "cash need",
        "cash needs",
        "job change",
        "retirement goal",
    ]

    assert all(
        claim not in combined_prompt
        for claim in unsupported_claims
    )
