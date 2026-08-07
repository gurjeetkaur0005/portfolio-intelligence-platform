from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import requests
from requests import Response, Session


JsonValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | list["JsonValue"]
    | dict[str, "JsonValue"]
)

JsonObject: TypeAlias = dict[str, JsonValue]

QueryParams: TypeAlias = dict[str, str | int | float]


class ApiClientError(RuntimeError):
    """Base error raised for frontend API failures."""


class ApiConnectionError(ApiClientError):
    """Raised when the FastAPI backend cannot be reached."""


class ApiResponseError(ApiClientError):
    """Raised when FastAPI returns an unsuccessful response."""

    def __init__(
        self,
        *,
        status_code: int,
        message: str,
    ) -> None:
        self.status_code = status_code
        super().__init__(message)


class ApiPayloadError(ApiClientError):
    """Raised when the backend returns an unexpected payload."""


@dataclass(frozen=True, slots=True)
class PaginatedResponse:
    """Represent the backend pagination contract."""

    items: list[JsonObject]
    limit: int
    offset: int
    count: int


class FastApiClient:
    """Provide reusable HTTP access to the FastAPI backend."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float = 10.0,
        session: Session | None = None,
    ) -> None:
        normalized_base_url = base_url.strip().rstrip("/")

        if not normalized_base_url:
            raise ValueError("base_url must not be empty.")

        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be greater than zero."
            )

        self._base_url = normalized_base_url
        self._timeout_seconds = timeout_seconds
        self._session = session or requests.Session()

    def get_health(self) -> JsonObject:
        """Return the FastAPI liveness response."""

        return self._get_json("/health")

    def get_readiness(self) -> JsonObject:
        """Return FastAPI and PostgreSQL readiness information."""

        return self._get_json("/ready")

    def list_portfolios(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> PaginatedResponse:
        """Return a paginated collection of stored portfolios."""

        self._validate_pagination(
            limit=limit,
            offset=offset,
        )

        payload = self._get_json(
            "/portfolios",
            params={
                "limit": limit,
                "offset": offset,
            },
        )

        return self._parse_paginated_response(payload)

    def get_portfolio(
        self,
        portfolio_id: str,
    ) -> JsonObject:
        """Return one stored portfolio with holdings."""

        normalized_portfolio_id = portfolio_id.strip()

        if not normalized_portfolio_id:
            raise ValueError(
                "portfolio_id must not be empty."
            )

        payload = self._get_json(
            f"/portfolios/{normalized_portfolio_id}"
        )

        return self._parse_portfolio_detail(payload)

    def run_portfolio_rebalance(
        self,
        *,
        portfolio_id: str,
        transaction_cost_rate: float = 0.002,
    ) -> JsonObject:
        """Run and persist rebalancing for one stored portfolio."""

        normalized_portfolio_id = portfolio_id.strip()

        if not normalized_portfolio_id:
            raise ValueError(
                "portfolio_id must not be empty."
            )

        if not 0.0 <= transaction_cost_rate <= 1.0:
            raise ValueError(
                "transaction_cost_rate must be between 0 and 1."
            )

        payload = self._post_json(
            f"/portfolios/{normalized_portfolio_id}/rebalance",
            json={
                "transaction_cost_rate": transaction_cost_rate,
            },
        )

        return self._parse_database_rebalance_response(
            payload
        )

    def get_rebalance(
        self,
        run_id: str,
    ) -> JsonObject:
        """Return one persisted rebalance run."""

        normalized_run_id = run_id.strip()

        if not normalized_run_id:
            raise ValueError("run_id must not be empty.")

        return self._get_json(
            f"/rebalances/{normalized_run_id}"
        )

    def list_portfolio_rebalances(
        self,
        *,
        portfolio_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> PaginatedResponse:
        """Return persisted rebalance runs for one portfolio."""

        normalized_portfolio_id = portfolio_id.strip()

        if not normalized_portfolio_id:
            raise ValueError(
                "portfolio_id must not be empty."
            )

        self._validate_pagination(
            limit=limit,
            offset=offset,
        )

        payload = self._get_json(
            f"/portfolios/{normalized_portfolio_id}/rebalances",
            params={
                "limit": limit,
                "offset": offset,
            },
        )

        return self._parse_paginated_response(
            payload
        )

    def list_rebalance_trades(
        self,
        *,
        run_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> PaginatedResponse:
        """Return paginated trades for one persisted rebalance run."""

        normalized_run_id = run_id.strip()

        if not normalized_run_id:
            raise ValueError("run_id must not be empty.")

        self._validate_pagination(
            limit=limit,
            offset=offset,
        )

        payload = self._get_json(
            f"/rebalances/{normalized_run_id}/trades",
            params={
                "limit": limit,
                "offset": offset,
            },
        )

        return self._parse_paginated_response(payload)

    def list_rebalance_audit(
        self,
        *,
        run_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> PaginatedResponse:
        """Return paginated audit records for one rebalance run."""

        normalized_run_id = run_id.strip()

        if not normalized_run_id:
            raise ValueError("run_id must not be empty.")

        self._validate_pagination(
            limit=limit,
            offset=offset,
        )

        payload = self._get_json(
            f"/rebalances/{normalized_run_id}/audit",
            params={
                "limit": limit,
                "offset": offset,
            },
        )

        return self._parse_paginated_response(payload)

    def _get_json(
        self,
        path: str,
        *,
        params: QueryParams | None = None,
    ) -> JsonObject:
        """Send a GET request and return a validated JSON object."""

        url = f"{self._base_url}{path}"

        try:
            response = self._session.get(
                url,
                params=params,
                timeout=self._timeout_seconds,
            )
        except requests.RequestException as exc:
            raise ApiConnectionError(
                "Could not connect to the FastAPI service."
            ) from exc

        self._raise_for_status(response)

        return self._decode_json_object(response)

    def _post_json(
        self,
        path: str,
        *,
        json: JsonObject,
    ) -> JsonObject:
        """Send a POST request and return a validated JSON object."""

        url = f"{self._base_url}{path}"

        try:
            response = self._session.post(
                url,
                json=json,
                timeout=self._timeout_seconds,
            )
        except requests.RequestException as exc:
            raise ApiConnectionError(
                "Could not connect to the FastAPI service."
            ) from exc

        self._raise_for_status(response)

        return self._decode_json_object(response)

    @staticmethod
    def _raise_for_status(
        response: Response,
    ) -> None:
        """Raise a safe frontend error for unsuccessful responses."""

        if response.ok:
            return

        message = "The FastAPI service returned an error."

        try:
            payload = response.json()
        except requests.JSONDecodeError:
            payload = None

        if isinstance(payload, dict):
            detail = payload.get("detail")

            if isinstance(detail, str) and detail.strip():
                message = detail.strip()

        raise ApiResponseError(
            status_code=response.status_code,
            message=message,
        )

    @staticmethod
    def _decode_json_object(
        response: Response,
    ) -> JsonObject:
        """Decode and validate a JSON object response."""

        try:
            payload = response.json()
        except requests.JSONDecodeError as exc:
            raise ApiPayloadError(
                "The FastAPI service returned invalid JSON."
            ) from exc

        if not isinstance(payload, dict):
            raise ApiPayloadError(
                "The FastAPI service returned an unexpected payload."
            )

        return payload

    @staticmethod
    def _validate_pagination(
        *,
        limit: int,
        offset: int,
    ) -> None:
        """Validate frontend pagination arguments."""

        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero."
            )

        if limit > 50:
            raise ValueError(
                "limit must not exceed 50."
            )

        if offset < 0:
            raise ValueError(
                "offset must not be negative."
            )

    @staticmethod
    def _parse_paginated_response(
        payload: JsonObject,
    ) -> PaginatedResponse:
        """Validate a paginated backend response."""

        items_value = payload.get("items")

        if not isinstance(items_value, list):
            raise ApiPayloadError(
                "Paginated response field 'items' must be a list."
            )

        items: list[JsonObject] = []

        for item in items_value:
            if not isinstance(item, dict):
                raise ApiPayloadError(
                    "Every paginated item must be an object."
                )

            items.append(item)

        parsed_limit = FastApiClient._required_integer(
            payload.get("limit"),
            "limit",
        )

        parsed_offset = FastApiClient._required_integer(
            payload.get("offset"),
            "offset",
        )

        parsed_count = FastApiClient._required_integer(
            payload.get("count"),
            "count",
        )

        return PaginatedResponse(
            items=items,
            limit=parsed_limit,
            offset=parsed_offset,
            count=parsed_count,
        )

    @staticmethod
    def _parse_database_rebalance_response(
        payload: JsonObject,
    ) -> JsonObject:
        """Validate a stored-portfolio rebalance response."""

        for field_name in (
            "status",
            "portfolio_id",
            "run_id",
            "message",
        ):
            FastApiClient._required_string(
                payload.get(field_name),
                field_name,
            )

        for field_name in (
            "trade_count",
            "database_run_id",
        ):
            FastApiClient._required_integer(
                payload.get(field_name),
                field_name,
            )

        return payload

    @staticmethod
    def _parse_portfolio_detail(
        payload: JsonObject,
    ) -> JsonObject:
        """Validate one portfolio-detail API response."""

        FastApiClient._required_string(
            payload.get("portfolio_id"),
            "portfolio_id",
        )

        FastApiClient._required_string(
            payload.get("client_id"),
            "client_id",
        )

        FastApiClient._required_number(
            payload.get("portfolio_value"),
            "portfolio_value",
        )

        FastApiClient._required_string(
            payload.get("currency"),
            "currency",
        )

        holdings_value = payload.get("holdings")

        if not isinstance(holdings_value, list):
            raise ApiPayloadError(
                "Portfolio detail field 'holdings' must be a list."
            )

        for holding in holdings_value:
            if not isinstance(holding, dict):
                raise ApiPayloadError(
                    "Every portfolio holding must be an object."
                )

            FastApiClient._validate_portfolio_holding(
                holding
            )

        FastApiClient._required_integer(
            payload.get("holding_count"),
            "holding_count",
        )

        return payload

    @staticmethod
    def _validate_portfolio_holding(
        holding: JsonObject,
    ) -> None:
        """Validate one portfolio holding."""

        FastApiClient._required_string(
            holding.get("asset"),
            "asset",
        )

        for field_name in (
            "current_weight",
            "current_value",
            "cost_basis",
        ):
            FastApiClient._required_number(
                holding.get(field_name),
                field_name,
            )

    @staticmethod
    def _required_string(
        value: JsonValue,
        field_name: str,
    ) -> str:
        """Return a validated non-empty string."""

        if not isinstance(value, str) or not value.strip():
            raise ApiPayloadError(
                f"Response field '{field_name}' "
                "must be a non-empty string."
            )

        return value

    @staticmethod
    def _required_integer(
        value: JsonValue,
        field_name: str,
    ) -> int:
        """Return a validated integer."""

        if not isinstance(value, int) or isinstance(value, bool):
            raise ApiPayloadError(
                f"Response field '{field_name}' "
                "must be an integer."
            )

        return value

    @staticmethod
    def _required_number(
        value: JsonValue,
        field_name: str,
    ) -> float:
        """Return a validated numeric value."""

        if not isinstance(
            value,
            (int, float),
        ) or isinstance(value, bool):
            raise ApiPayloadError(
                f"Response field '{field_name}' "
                "must be numeric."
            )

        return float(value)