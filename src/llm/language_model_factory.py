from __future__ import annotations

from enum import StrEnum

from src.llm.fake_language_model import FakeLanguageModel
from src.llm.gemini_language_model import (
    DEFAULT_GEMINI_MODEL,
    GeminiClientProtocol,
    GeminiLanguageModel,
)
from src.llm.language_model import LanguageModelProtocol


class LanguageModelProvider(StrEnum):
    """Supported language-model providers."""

    GEMINI = "gemini"
    FAKE = "fake"


def create_language_model(
    provider: LanguageModelProvider | str,
    *,
    fake_responses: list[str] | None = None,
    api_key: str | None = None,
    model_name: str = DEFAULT_GEMINI_MODEL,
    gemini_client: GeminiClientProtocol | None = None,
) -> LanguageModelProtocol:
    """
    Create a language-model implementation for the selected provider.

    Args:
        provider:
            Language-model provider to create.

        fake_responses:
            Predefined responses required by FakeLanguageModel.

        api_key:
            Optional Gemini API key. If omitted, GeminiLanguageModel reads
            GEMINI_API_KEY from the environment.

        model_name:
            Gemini model name.

        gemini_client:
            Optional injected Gemini client used primarily in tests.

    Returns:
        A provider-independent language-model implementation.

    Raises:
        TypeError:
            If provider is not a string or LanguageModelProvider.

        ValueError:
            If the provider is unsupported or required configuration is
            missing.
    """

    resolved_provider = _resolve_provider(provider)

    if resolved_provider is LanguageModelProvider.GEMINI:
        return GeminiLanguageModel(
            api_key=api_key,
            model_name=model_name,
            client=gemini_client,
        )

    if fake_responses is None:
        raise ValueError(
            "fake_responses must be provided for the fake provider."
        )

    return FakeLanguageModel(
        responses=fake_responses,
    )


def _resolve_provider(
    provider: LanguageModelProvider | str,
) -> LanguageModelProvider:
    """Validate and normalize a language-model provider."""

    if isinstance(provider, LanguageModelProvider):
        return provider

    if not isinstance(provider, str):
        raise TypeError(
            "provider must be a string or LanguageModelProvider."
        )

    normalized_provider = provider.strip().lower()

    if not normalized_provider:
        raise ValueError(
            "provider must not be empty."
        )

    try:
        return LanguageModelProvider(normalized_provider)
    except ValueError as error:
        supported_providers = ", ".join(
            provider.value
            for provider in LanguageModelProvider
        )

        raise ValueError(
            "Unsupported language-model provider "
            f"'{provider}'. Supported providers: "
            f"{supported_providers}."
        ) from error