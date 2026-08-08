from __future__ import annotations

from requests import Response

from streamlit_app.services.api_client import (
    ApiResponseError,
    FastApiClient,
    JsonObject,
    REBALANCE_CONFLICT_MESSAGE,
)


class FakeSession:
    """Return a predefined HTTP response."""

    def __init__(
        self,
        response: Response,
    ) -> None:
        self.response = response

    def post(
        self,
        url: str,
        *,
        json: JsonObject,
        timeout: float,
    ) -> Response:
        return self.response


def _json_response(
    *,
    status_code: int,
    body: bytes,
) -> Response:
    """Build a minimal requests response for client tests."""

    response = Response()
    response.status_code = status_code
    response._content = body
    response.headers["Content-Type"] = "application/json"

    return response


def test_rebalance_conflict_returns_friendly_message() -> None:
    """HTTP 409 responses are presented as a helpful user message."""

    client = FastApiClient(
        base_url="http://api.test",
        session=FakeSession(
            _json_response(
                status_code=409,
                body=b'{"detail":"database lock unavailable"}',
            )
        ),
    )

    try:
        client.run_portfolio_rebalance(
            portfolio_id="P00001",
            transaction_cost_rate=0.002,
        )
    except ApiResponseError as error:
        assert error.status_code == 409
        assert str(error) == REBALANCE_CONFLICT_MESSAGE
    else:
        raise AssertionError("Expected ApiResponseError.")
