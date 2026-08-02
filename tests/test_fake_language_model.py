from __future__ import annotations

import pytest

from src.llm.fake_language_model import (
    FakeLanguageModel,
)
from src.llm.language_model import (
    LanguageModelRequest,
)


def _build_request(
    user_prompt: str = "Explain this portfolio.",
) -> LanguageModelRequest:
    """Build a valid test request."""

    return LanguageModelRequest(
        system_prompt="You are a financial advisor.",
        user_prompt=user_prompt,
        temperature=0.0,
        max_output_tokens=300,
    )


def test_fake_model_returns_configured_response() -> None:
    model = FakeLanguageModel(
        responses=[
            "The portfolio requires rebalancing.",
        ]
    )

    response = model.generate(
        _build_request()
    )

    assert response.text == (
        "The portfolio requires rebalancing."
    )
    assert response.model_name == "fake-language-model"


def test_fake_model_returns_responses_in_order() -> None:
    model = FakeLanguageModel(
        responses=[
            "First response.",
            "Second response.",
        ]
    )

    first_response = model.generate(
        _build_request("First request.")
    )
    second_response = model.generate(
        _build_request("Second request.")
    )

    assert first_response.text == "First response."
    assert second_response.text == "Second response."


def test_fake_model_records_requests() -> None:
    model = FakeLanguageModel(
        responses=[
            "Generated response.",
        ]
    )

    request = _build_request()

    model.generate(request)

    assert model.call_count == 1
    assert model.requests == [request]


def test_call_count_increases_after_each_request() -> None:
    model = FakeLanguageModel(
        responses=[
            "First response.",
            "Second response.",
        ]
    )

    assert model.call_count == 0

    model.generate(
        _build_request("First request.")
    )

    assert model.call_count == 1

    model.generate(
        _build_request("Second request.")
    )

    assert model.call_count == 2


def test_fake_model_raises_when_responses_are_exhausted() -> None:
    model = FakeLanguageModel(
        responses=[
            "Only response.",
        ]
    )

    model.generate(
        _build_request()
    )

    with pytest.raises(
        RuntimeError,
        match="No configured fake responses remain",
    ):
        model.generate(
            _build_request()
        )


def test_fake_model_rejects_empty_response_list() -> None:
    with pytest.raises(
        ValueError,
        match="responses must not be empty",
    ):
        FakeLanguageModel(
            responses=[],
        )


def test_fake_model_rejects_non_list_responses() -> None:
    with pytest.raises(
        TypeError,
        match="responses must be a list",
    ):
        FakeLanguageModel(
            responses="response",  # type: ignore[arg-type]
        )


def test_fake_model_rejects_empty_response_text() -> None:
    with pytest.raises(
        ValueError,
        match="non-empty strings",
    ):
        FakeLanguageModel(
            responses=[""],
        )


def test_fake_model_rejects_whitespace_response_text() -> None:
    with pytest.raises(
        ValueError,
        match="non-empty strings",
    ):
        FakeLanguageModel(
            responses=["   "],
        )


def test_fake_model_rejects_empty_model_name() -> None:
    with pytest.raises(
        ValueError,
        match="model_name must not be empty",
    ):
        FakeLanguageModel(
            responses=[
                "Response.",
            ],
            model_name="",
        )


def test_fake_model_rejects_invalid_request() -> None:
    model = FakeLanguageModel(
        responses=[
            "Response.",
        ]
    )

    with pytest.raises(
        TypeError,
        match="request must be a LanguageModelRequest",
    ):
        model.generate(  # type: ignore[arg-type]
            "invalid request"
        )


def test_fake_response_has_no_token_metadata() -> None:
    model = FakeLanguageModel(
        responses=[
            "Response.",
        ]
    )

    response = model.generate(
        _build_request()
    )

    assert response.input_tokens is None
    assert response.output_tokens is None