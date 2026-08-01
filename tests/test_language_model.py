from __future__ import annotations

import pytest

from src.llm.language_model import (
    LanguageModelRequest,
    LanguageModelResponse,
)


def test_request_stores_valid_values() -> None:
    request = LanguageModelRequest(
        system_prompt="You are a financial advisor.",
        user_prompt="Explain this portfolio.",
        temperature=0.2,
        max_output_tokens=300,
    )

    assert request.system_prompt == (
        "You are a financial advisor."
    )
    assert request.user_prompt == (
        "Explain this portfolio."
    )
    assert request.temperature == pytest.approx(0.2)
    assert request.max_output_tokens == 300


def test_request_uses_safe_defaults() -> None:
    request = LanguageModelRequest(
        system_prompt="You are a financial advisor.",
        user_prompt="Explain the recommendation.",
    )

    assert request.temperature == 0.0
    assert request.max_output_tokens == 500


def test_request_rejects_empty_system_prompt() -> None:
    with pytest.raises(
        ValueError,
        match="system_prompt must not be empty",
    ):
        LanguageModelRequest(
            system_prompt="   ",
            user_prompt="Explain this portfolio.",
        )


def test_request_rejects_empty_user_prompt() -> None:
    with pytest.raises(
        ValueError,
        match="user_prompt must not be empty",
    ):
        LanguageModelRequest(
            system_prompt="You are a financial advisor.",
            user_prompt="",
        )


@pytest.mark.parametrize(
    "temperature",
    [-0.1, 2.1],
)
def test_request_rejects_invalid_temperature(
    temperature: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="temperature must be between",
    ):
        LanguageModelRequest(
            system_prompt="You are a financial advisor.",
            user_prompt="Explain this portfolio.",
            temperature=temperature,
        )


def test_request_rejects_zero_max_output_tokens() -> None:
    with pytest.raises(
        ValueError,
        match="max_output_tokens must be greater than zero",
    ):
        LanguageModelRequest(
            system_prompt="You are a financial advisor.",
            user_prompt="Explain this portfolio.",
            max_output_tokens=0,
        )


def test_response_stores_generated_text() -> None:
    response = LanguageModelResponse(
        text="Your portfolio requires rebalancing.",
        model_name="test-model",
        input_tokens=100,
        output_tokens=25,
    )

    assert response.text == (
        "Your portfolio requires rebalancing."
    )
    assert response.model_name == "test-model"
    assert response.input_tokens == 100
    assert response.output_tokens == 25


def test_response_allows_missing_metadata() -> None:
    response = LanguageModelResponse(
        text="Portfolio explanation.",
    )

    assert response.model_name is None
    assert response.input_tokens is None
    assert response.output_tokens is None


def test_response_rejects_empty_text() -> None:
    with pytest.raises(
        ValueError,
        match="text must not be empty",
    ):
        LanguageModelResponse(
            text="   ",
        )


def test_response_rejects_negative_input_tokens() -> None:
    with pytest.raises(
        ValueError,
        match="input_tokens must not be negative",
    ):
        LanguageModelResponse(
            text="Valid response.",
            input_tokens=-1,
        )


def test_response_rejects_negative_output_tokens() -> None:
    with pytest.raises(
        ValueError,
        match="output_tokens must not be negative",
    ):
        LanguageModelResponse(
            text="Valid response.",
            output_tokens=-1,
        )