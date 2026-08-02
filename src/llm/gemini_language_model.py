from __future__ import annotations

import os
from typing import Protocol, cast

from google import genai
from google.genai import types

from src.llm.language_model import (
    LanguageModelProtocol,
    LanguageModelRequest,
    LanguageModelResponse,
)


DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
LLM_DEBUG_ENV = "LLM_DEBUG"


class GeminiResponseProtocol(Protocol):
    """Describe the Gemini response fields used by this provider."""

    text: str | None
    usage_metadata: object | None


class GeminiModelsProtocol(Protocol):
    """Describe the Gemini models client used by this provider."""

    def generate_content(
        self,
        *,
        model: str,
        contents: str,
        config: types.GenerateContentConfig,
    ) -> GeminiResponseProtocol:
        """Generate content through Gemini."""
        ...


class GeminiClientProtocol(Protocol):
    """Describe the Gemini client dependency used by this provider."""

    models: GeminiModelsProtocol


class GeminiLanguageModelError(RuntimeError):
    """Raised when Gemini cannot return a usable response."""


class GeminiLanguageModel(LanguageModelProtocol):
    """Generate provider-independent responses using Gemini."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = DEFAULT_GEMINI_MODEL,
        client: GeminiClientProtocol | None = None,
    ) -> None:
        """
        Initialize the Gemini provider.

        A client may be injected in unit tests. In normal execution,
        the provider creates the official Gemini client using the API key.
        """

        self._model_name = _validate_model_name(model_name)

        if client is not None:
            self._client = client
            return

        resolved_api_key = _resolve_api_key(api_key)

        self._client = cast(
            GeminiClientProtocol,
            genai.Client(
                api_key=resolved_api_key,
            ),
        )

    def generate(
        self,
        request: LanguageModelRequest,
    ) -> LanguageModelResponse:
        """Generate text for a provider-independent request."""

        if not isinstance(request, LanguageModelRequest):
            raise TypeError(
                "request must be a LanguageModelRequest."
            )

        if _debug_enabled():
            print("Gemini LanguageModelRequest:")
            print(f"system_prompt={request.system_prompt!r}")
            print(f"user_prompt={request.user_prompt!r}")
            print(f"temperature={request.temperature!r}")
            print(f"max_output_tokens={request.max_output_tokens!r}")

        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=request.user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=request.system_prompt,
                    temperature=request.temperature,
                    max_output_tokens=request.max_output_tokens,
                ),
            )
        except Exception as error:
            raise GeminiLanguageModelError(
                "Gemini failed to generate a response."
            ) from error

        response_text = _extract_response_text(response)
        input_tokens, output_tokens = _extract_token_counts(
            response
        )
        finish_reason = _extract_finish_reason(response)

        if _debug_enabled():
            print(f"Gemini raw response.text={response.text!r}")
            print(f"Gemini extracted response_text={response_text!r}")
            print(f"Gemini input_tokens={input_tokens!r}")
            print(f"Gemini output_tokens={output_tokens!r}")
            print(f"Gemini finish_reason={finish_reason!r}")

        return LanguageModelResponse(
            text=response_text,
            model_name=self._model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


def _resolve_api_key(
    api_key: str | None,
) -> str:
    """Resolve and validate the Gemini API key."""

    if api_key is not None and not isinstance(api_key, str):
        raise TypeError(
            "api_key must be a string or None."
        )

    resolved_api_key = (
        api_key
        if api_key is not None
        else os.getenv(GEMINI_API_KEY_ENV)
    )

    if (
        resolved_api_key is None
        or not resolved_api_key.strip()
    ):
        raise ValueError(
            "Gemini API key must be supplied directly or through "
            "the GEMINI_API_KEY environment variable."
        )

    return resolved_api_key.strip()


def _validate_model_name(
    model_name: str,
) -> str:
    """Validate and normalize the Gemini model name."""

    if not isinstance(model_name, str):
        raise TypeError(
            "model_name must be a string."
        )

    normalized_model_name = model_name.strip()

    if not normalized_model_name:
        raise ValueError(
            "model_name must not be empty."
        )

    return normalized_model_name


def _extract_response_text(
    response: GeminiResponseProtocol,
) -> str:
    """Extract non-empty generated text."""

    response_text = response.text

    if (
        response_text is None
        or not isinstance(response_text, str)
        or not response_text.strip()
    ):
        raise GeminiLanguageModelError(
            "Gemini returned an empty response."
        )

    return response_text.strip()


def _extract_token_counts(
    response: GeminiResponseProtocol,
) -> tuple[int | None, int | None]:
    """Extract optional request and response token counts."""

    usage_metadata = response.usage_metadata

    if usage_metadata is None:
        return None, None

    input_tokens = getattr(
        usage_metadata,
        "prompt_token_count",
        None,
    )
    output_tokens = getattr(
        usage_metadata,
        "candidates_token_count",
        None,
    )

    return (
        _normalize_token_count(input_tokens),
        _normalize_token_count(output_tokens),
    )


def _normalize_token_count(
    value: object,
) -> int | None:
    """Normalize an SDK token-count value."""

    if not isinstance(value, int):
        return None

    if value < 0:
        return None

    return value


def _extract_finish_reason(
    response: GeminiResponseProtocol,
) -> str | None:
    """Extract the first Gemini finish reason when available."""

    candidates = getattr(response, "candidates", None)

    if not candidates:
        return None

    first_candidate = candidates[0]
    finish_reason = getattr(first_candidate, "finish_reason", None)

    if finish_reason is None:
        return None

    return str(finish_reason)


def _debug_enabled() -> bool:
    """Return whether LLM request/response debugging is enabled."""

    return os.getenv(LLM_DEBUG_ENV) == "1"
