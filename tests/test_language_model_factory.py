from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.llm.fake_language_model import FakeLanguageModel
from src.llm.gemini_language_model import GeminiLanguageModel
from src.llm.language_model_factory import (
    LanguageModelProvider,
    create_language_model,
)


@dataclass
class FakeUsageMetadata:
    prompt_token_count: int | None = None
    candidates_token_count: int | None = None


@dataclass
class FakeGeminiResponse:
    text: str | None = "Gemini response."
    usage_metadata: FakeUsageMetadata | None = field(
        default_factory=FakeUsageMetadata
    )


class FakeGeminiModels:
    def generate_content(
        self,
        *,
        model: str,
        contents: str,
        config,
    ) -> FakeGeminiResponse:
        return FakeGeminiResponse()


@dataclass
class FakeGeminiClient:
    models: FakeGeminiModels = field(
        default_factory=FakeGeminiModels
    )


def test_factory_returns_fake_language_model() -> None:
    model = create_language_model(
        provider=LanguageModelProvider.FAKE,
        fake_responses=[
            "Fake response.",
        ],
    )

    assert isinstance(model, FakeLanguageModel)


def test_factory_returns_gemini_language_model_with_injected_client() -> None:
    model = create_language_model(
        provider=LanguageModelProvider.GEMINI,
        gemini_client=FakeGeminiClient(),
    )

    assert isinstance(model, GeminiLanguageModel)


def test_factory_accepts_string_provider() -> None:
    model = create_language_model(
        provider="fake",
        fake_responses=[
            "Fake response.",
        ],
    )

    assert isinstance(model, FakeLanguageModel)


def test_factory_rejects_unknown_provider() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported language-model provider",
    ):
        create_language_model(
            provider="unknown",
            fake_responses=[
                "Fake response.",
            ],
        )


def test_factory_requires_fake_responses_for_fake_provider() -> None:
    with pytest.raises(
        ValueError,
        match="fake_responses must be provided",
    ):
        create_language_model(
            provider=LanguageModelProvider.FAKE,
        )


def test_factory_rejects_invalid_provider_type() -> None:
    with pytest.raises(
        TypeError,
        match="provider must be a string or LanguageModelProvider",
    ):
        create_language_model(  # type: ignore[arg-type]
            provider=1,
            fake_responses=[
                "Fake response.",
            ],
        )
