from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pytest

from src.llm.gemini_language_model import (
    DEFAULT_GEMINI_MODEL,
    GeminiLanguageModel,
    GeminiLanguageModelError,
)
from src.llm.language_model import (
    LanguageModelRequest,
    LanguageModelResponse,
)


@dataclass
class FakeUsageMetadata:
    prompt_token_count: int | None = 12
    candidates_token_count: int | None = 5


@dataclass
class FakeCandidate:
    finish_reason: str | None = "STOP"


@dataclass
class FakeGeminiResponse:
    text: str | None
    usage_metadata: FakeUsageMetadata | None = (
        field(default_factory=FakeUsageMetadata)
    )
    candidates: list[FakeCandidate] = field(
        default_factory=lambda: [FakeCandidate()]
    )


class FakeGeminiModels:
    def __init__(
        self,
        response: FakeGeminiResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    def generate_content(
        self,
        *,
        model: str,
        contents: str,
        config,
    ) -> FakeGeminiResponse:
        self.calls.append(
            {
                "model": model,
                "contents": contents,
                "config": config,
            }
        )

        if self.error is not None:
            raise self.error

        if self.response is None:
            raise RuntimeError("No fake response configured.")

        return self.response


class FakeGeminiClient:
    def __init__(
        self,
        models: FakeGeminiModels,
    ) -> None:
        self.models = models


def _build_request() -> LanguageModelRequest:
    return LanguageModelRequest(
        system_prompt="System prompt.",
        user_prompt="User prompt.",
        temperature=0.0,
        max_output_tokens=300,
    )


def test_gemini_returns_language_model_response() -> None:
    models = FakeGeminiModels(
        response=FakeGeminiResponse(
            text="Generated response.",
        )
    )
    client = FakeGeminiClient(models=models)
    model = GeminiLanguageModel(
        api_key=None,
        client=client,
    )

    response = model.generate(_build_request())

    assert isinstance(response, LanguageModelResponse)
    assert response.text == "Generated response."
    assert response.model_name == DEFAULT_GEMINI_MODEL
    assert response.input_tokens == 12
    assert response.output_tokens == 5


def test_gemini_passes_request_fields_to_client() -> None:
    models = FakeGeminiModels(
        response=FakeGeminiResponse(
            text="Generated response.",
        )
    )
    client = FakeGeminiClient(models=models)
    model = GeminiLanguageModel(
        api_key=None,
        model_name="gemini-test-model",
        client=client,
    )
    request = _build_request()

    model.generate(request)

    assert models.calls[0]["model"] == "gemini-test-model"
    assert models.calls[0]["contents"] == request.user_prompt
    config = models.calls[0]["config"]
    assert config.system_instruction == request.system_prompt
    assert config.temperature == request.temperature
    assert config.max_output_tokens == request.max_output_tokens


def test_gemini_rejects_invalid_request() -> None:
    model = GeminiLanguageModel(
        api_key=None,
        client=FakeGeminiClient(
            models=FakeGeminiModels(
                response=FakeGeminiResponse(
                    text="Generated response.",
                )
            )
        ),
    )

    with pytest.raises(
        TypeError,
        match="request must be a LanguageModelRequest",
    ):
        model.generate("invalid")  # type: ignore[arg-type]


def test_gemini_wraps_provider_errors() -> None:
    model = GeminiLanguageModel(
        api_key=None,
        client=FakeGeminiClient(
            models=FakeGeminiModels(
                error=RuntimeError("provider failed"),
            )
        ),
    )

    with pytest.raises(
        GeminiLanguageModelError,
        match="Gemini failed to generate a response",
    ):
        model.generate(_build_request())


def test_gemini_rejects_empty_response() -> None:
    model = GeminiLanguageModel(
        api_key=None,
        client=FakeGeminiClient(
            models=FakeGeminiModels(
                response=FakeGeminiResponse(text="  "),
            )
        ),
    )

    with pytest.raises(
        GeminiLanguageModelError,
        match="Gemini returned an empty response",
    ):
        model.generate(_build_request())


def test_gemini_requires_api_key_without_injected_client(
    monkeypatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(
        ValueError,
        match="Gemini API key must be supplied",
    ):
        GeminiLanguageModel()


def test_gemini_debug_mode_logs_finish_reason(
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.setenv("LLM_DEBUG", "1")
    caplog.set_level(logging.INFO)
    models = FakeGeminiModels(
        response=FakeGeminiResponse(
            text="Generated response.",
        )
    )
    model = GeminiLanguageModel(
        api_key=None,
        client=FakeGeminiClient(models=models),
    )

    model.generate(_build_request())

    assert "gemini_response" in caplog.text
    assert "finish_reason=STOP" in caplog.text
