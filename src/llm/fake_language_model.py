from __future__ import annotations

from dataclasses import dataclass, field

from src.llm.language_model import (
    LanguageModelRequest,
    LanguageModelResponse,
)


@dataclass
class FakeLanguageModel:
    """
    Deterministic language model for tests and local development.

    The fake model does not call an external API. It returns predefined
    responses in order and records every request it receives.
    """

    responses: list[str]
    model_name: str = "fake-language-model"
    requests: list[LanguageModelRequest] = field(
        default_factory=list,
        init=False,
    )
    _response_index: int = field(
        default=0,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate fake model configuration."""

        if not isinstance(self.responses, list):
            raise TypeError(
                "responses must be a list."
            )

        if not self.responses:
            raise ValueError(
                "responses must not be empty."
            )

        invalid_responses = [
            response
            for response in self.responses
            if (
                not isinstance(response, str)
                or not response.strip()
            )
        ]

        if invalid_responses:
            raise ValueError(
                "responses must contain non-empty strings."
            )

        if not isinstance(self.model_name, str):
            raise TypeError(
                "model_name must be a string."
            )

        if not self.model_name.strip():
            raise ValueError(
                "model_name must not be empty."
            )

    def generate(
        self,
        request: LanguageModelRequest,
    ) -> LanguageModelResponse:
        """
        Return the next configured fake response.

        Raises:
            TypeError:
                If request is not a LanguageModelRequest.

            RuntimeError:
                If no configured responses remain.
        """

        if not isinstance(request, LanguageModelRequest):
            raise TypeError(
                "request must be a LanguageModelRequest."
            )

        if self._response_index >= len(self.responses):
            raise RuntimeError(
                "No configured fake responses remain."
            )

        self.requests.append(request)

        response_text = self.responses[
            self._response_index
        ]
        self._response_index += 1

        return LanguageModelResponse(
            text=response_text,
            model_name=self.model_name,
            input_tokens=None,
            output_tokens=None,
        )

    @property
    def call_count(self) -> int:
        """Return the number of requests received."""

        return len(self.requests)