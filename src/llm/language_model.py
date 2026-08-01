from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class LanguageModelRequest:
    """
    Store a provider-independent language model request.

    The same request object can be used with OpenAI, Gemini, Claude,
    or a fake language model used during testing.
    """

    system_prompt: str
    user_prompt: str
    temperature: float = 0.0
    max_output_tokens: int = 500

    def __post_init__(self) -> None:
        """Validate the language model request."""

        if not isinstance(self.system_prompt, str):
            raise TypeError(
                "system_prompt must be a string."
            )

        if not self.system_prompt.strip():
            raise ValueError(
                "system_prompt must not be empty."
            )

        if not isinstance(self.user_prompt, str):
            raise TypeError(
                "user_prompt must be a string."
            )

        if not self.user_prompt.strip():
            raise ValueError(
                "user_prompt must not be empty."
            )

        if not isinstance(
            self.temperature,
            int | float,
        ):
            raise TypeError(
                "temperature must be numeric."
            )

        if not 0.0 <= float(self.temperature) <= 2.0:
            raise ValueError(
                "temperature must be between 0.0 and 2.0."
            )

        if not isinstance(
            self.max_output_tokens,
            int,
        ):
            raise TypeError(
                "max_output_tokens must be an integer."
            )

        if self.max_output_tokens <= 0:
            raise ValueError(
                "max_output_tokens must be greater than zero."
            )


@dataclass(frozen=True, slots=True)
class LanguageModelResponse:
    """
    Store a provider-independent language model response.
    """

    text: str
    model_name: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    def __post_init__(self) -> None:
        """Validate the language model response."""

        if not isinstance(self.text, str):
            raise TypeError(
                "text must be a string."
            )

        if not self.text.strip():
            raise ValueError(
                "text must not be empty."
            )

        if (
            self.model_name is not None
            and not isinstance(self.model_name, str)
        ):
            raise TypeError(
                "model_name must be a string or None."
            )

        if (
            isinstance(self.model_name, str)
            and not self.model_name.strip()
        ):
            raise ValueError(
                "model_name must not be empty."
            )

        _validate_optional_token_count(
            value=self.input_tokens,
            field_name="input_tokens",
        )

        _validate_optional_token_count(
            value=self.output_tokens,
            field_name="output_tokens",
        )


class LanguageModelProtocol(Protocol):
    """
    Define the interface required from every language model provider.

    The agents depend on this protocol instead of depending directly on
    OpenAI, Gemini, Claude, or another provider.
    """

    def generate(
        self,
        request: LanguageModelRequest,
    ) -> LanguageModelResponse:
        """
        Generate text for a validated language model request.
        """
        ...


def _validate_optional_token_count(
    value: int | None,
    field_name: str,
) -> None:
    """Validate an optional token count."""

    if value is None:
        return

    if not isinstance(value, int):
        raise TypeError(
            f"{field_name} must be an integer or None."
        )

    if value < 0:
        raise ValueError(
            f"{field_name} must not be negative."
        )