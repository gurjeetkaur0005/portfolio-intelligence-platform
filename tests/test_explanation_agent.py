from __future__ import annotations

import pytest

from src.agents.explanation_agent import (
    ExplanationAgent,
)
from src.agents.portfolio_analyst_agent import (
    PortfolioAnalysis,
)
from src.llm.fake_language_model import (
    FakeLanguageModel,
)
from src.llm.language_model import (
    LanguageModelRequest,
)
from src.llm.prompt_builder import (
    PromptAudience,
    PromptBuilder,
)


class RecordingPromptBuilder(PromptBuilder):
    """Prompt builder spy for ExplanationAgent tests."""

    def __init__(self) -> None:
        self.call_count = 0
        self.audiences: list[PromptAudience | str] = []

    def build(
        self,
        analysis: PortfolioAnalysis,
        audience: PromptAudience | str,
    ) -> LanguageModelRequest:
        self.call_count += 1
        self.audiences.append(audience)

        return super().build(
            analysis=analysis,
            audience=audience,
        )


class FailingPromptBuilder(PromptBuilder):
    """Prompt builder that fails if the deterministic path calls it."""

    def build(
        self,
        analysis: PortfolioAnalysis,
        audience: PromptAudience | str,
    ) -> LanguageModelRequest:
        raise AssertionError(
            "PromptBuilder.build should not be called."
        )


def _build_analysis() -> PortfolioAnalysis:
    return PortfolioAnalysis(
        portfolio_id="portfolio_1",
        rebalance_required=True,
        highest_threshold_severity="high",
        threshold_breached=True,
        threshold_breach_count=2,
        assets_to_buy=("fixed_income",),
        assets_to_sell=("domestic_equity",),
        assets_to_hold=("cash",),
        total_transaction_cost=180.0,
        total_estimated_tax_liability=250.0,
        client_explanations=(
            "Domestic equity will be reduced.",
            "Fixed income will be increased.",
            "Cash remains unchanged.",
        ),
        advisor_explanations=(
            "Sell domestic equity.",
            "Buy fixed income.",
            "Hold cash.",
        ),
        compliance_explanations=(
            "SELL recommendation recorded.",
            "BUY recommendation recorded.",
            "HOLD recommendation recorded.",
        ),
    )


def test_explain_returns_portfolio_explanation() -> None:
    agent = ExplanationAgent()

    results = agent.explain(
        [_build_analysis()]
    )

    assert len(results) == 1
    assert results[0].portfolio_id == "portfolio_1"


def test_constructor_works_without_dependencies() -> None:
    agent = ExplanationAgent()

    result = agent.explain(
        [_build_analysis()]
    )[0]

    assert result.portfolio_id == "portfolio_1"


def test_client_summary_contains_portfolio_actions() -> None:
    agent = ExplanationAgent()

    result = agent.explain(
        [_build_analysis()]
    )[0]

    assert "requires rebalancing" in result.client_summary
    assert "fixed income" in result.client_summary
    assert "domestic equity" in result.client_summary


def test_no_language_model_uses_deterministic_summary() -> None:
    agent = ExplanationAgent(
        prompt_builder=FailingPromptBuilder(),
    )

    result = agent.explain(
        [_build_analysis()]
    )[0]

    assert "requires rebalancing" in result.client_summary
    assert "fixed income" in result.client_summary
    assert "domestic equity" in result.client_summary


def test_advisor_summary_contains_analysis_facts() -> None:
    agent = ExplanationAgent()

    result = agent.explain(
        [_build_analysis()]
    )[0]

    assert "Threshold breach count: 2" in result.advisor_summary
    assert "Highest threshold severity: high" in (
        result.advisor_summary
    )
    assert "$180.00" in result.advisor_summary
    assert "$250.00" in result.advisor_summary


def test_compliance_summary_states_no_calculations() -> None:
    agent = ExplanationAgent()

    result = agent.explain(
        [_build_analysis()]
    )[0]

    assert (
        "No financial calculations or trade decisions"
        in result.compliance_summary
    )


def test_empty_analysis_list_returns_empty_list() -> None:
    agent = ExplanationAgent()

    assert agent.explain([]) == []


def test_constructor_works_with_prompt_builder_and_language_model() -> None:
    fake_model = FakeLanguageModel(
        responses=[
            "AI generated explanation.",
        ]
    )

    agent = ExplanationAgent(
        prompt_builder=PromptBuilder(),
        language_model=fake_model,
    )

    result = agent.explain(
        [_build_analysis()]
    )[0]

    assert result.client_summary == "AI generated explanation."


def test_prompt_builder_build_is_called_once_with_language_model() -> None:
    prompt_builder = RecordingPromptBuilder()
    fake_model = FakeLanguageModel(
        responses=[
            "AI generated explanation.",
        ]
    )
    agent = ExplanationAgent(
        prompt_builder=prompt_builder,
        language_model=fake_model,
    )

    agent.explain(
        [_build_analysis()]
    )

    assert prompt_builder.call_count == 1
    assert prompt_builder.audiences == [PromptAudience.CLIENT]


def test_language_model_generate_is_called_once() -> None:
    fake_model = FakeLanguageModel(
        responses=[
            "AI generated explanation.",
        ]
    )
    agent = ExplanationAgent(
        prompt_builder=PromptBuilder(),
        language_model=fake_model,
    )

    agent.explain(
        [_build_analysis()]
    )

    assert fake_model.call_count == 1


def test_ai_response_replaces_only_client_summary() -> None:
    fake_model = FakeLanguageModel(
        responses=[
            "AI generated explanation.",
        ]
    )
    agent = ExplanationAgent(
        prompt_builder=PromptBuilder(),
        language_model=fake_model,
    )

    result = agent.explain(
        [_build_analysis()]
    )[0]

    assert result.client_summary == "AI generated explanation."
    assert "Threshold breach count: 2" in result.advisor_summary
    assert (
        "No financial calculations or trade decisions"
        in result.compliance_summary
    )


def test_language_model_receives_prompt_builder_request() -> None:
    fake_model = FakeLanguageModel(
        responses=[
            "AI generated explanation.",
        ]
    )
    agent = ExplanationAgent(
        prompt_builder=PromptBuilder(),
        language_model=fake_model,
    )

    agent.explain(
        [_build_analysis()]
    )

    assert len(fake_model.requests) == 1
    assert isinstance(
        fake_model.requests[0],
        LanguageModelRequest,
    )


def test_language_model_path_does_not_mutate_analysis() -> None:
    analysis = _build_analysis()
    original_analysis = _build_analysis()
    fake_model = FakeLanguageModel(
        responses=[
            "AI generated explanation.",
        ]
    )
    agent = ExplanationAgent(
        prompt_builder=PromptBuilder(),
        language_model=fake_model,
    )

    agent.explain([analysis])

    assert analysis == original_analysis


def test_non_list_input_raises_type_error() -> None:
    agent = ExplanationAgent()

    with pytest.raises(
        TypeError,
        match="must be provided as a list",
    ):
        agent.explain(  # type: ignore[arg-type]
            _build_analysis()
        )


def test_invalid_list_item_raises_type_error() -> None:
    agent = ExplanationAgent()

    with pytest.raises(
        TypeError,
        match="must contain PortfolioAnalysis objects",
    ):
        agent.explain(  # type: ignore[list-item]
            ["invalid"]
        )
