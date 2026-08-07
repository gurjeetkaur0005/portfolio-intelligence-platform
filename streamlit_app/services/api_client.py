from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import requests
from requests import Response, Session


JsonObject = dict[str, Any]


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

        if limit <= 0:
            raise ValueError("limit must be greater than zero.")

        if limit > 50:
            raise ValueError("limit must not exceed 50.")

        if offset < 0:
            raise ValueError("offset must not be negative.")

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
    def _get_json(
        self,
        path: str,
        *,
        params: Mapping[str, object] | None = None,
    ) -> JsonObject:
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

    @staticmethod
    def _raise_for_status(response: Response) -> None:
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
    def _decode_json_object(response: Response) -> JsonObject:
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
    def _parse_paginated_response(
        payload: JsonObject,
    ) -> PaginatedResponse:
        items = payload.get("items")
        limit = payload.get("limit")
        offset = payload.get("offset")
        count = payload.get("count")

        if not isinstance(items, list):
            raise ApiPayloadError(
                "Paginated response field 'items' must be a list."
            )

        if not all(isinstance(item, dict) for item in items):
            raise ApiPayloadError(
                "Every portfolio item must be an object."
            )

        for field_name, value in {
            "limit": limit,
            "offset": offset,
            "count": count,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise ApiPayloadError(
                    f"Paginated response field '{field_name}' "
                    "must be an integer."
                )

        return PaginatedResponse(
            items=items,
            limit=limit,
            offset=offset,
            count=count,
        )

    @staticmethod
    def _parse_portfolio_detail(
        payload: JsonObject,
    ) -> JsonObject:
        """Validate one portfolio-detail API response."""

        portfolio_id = payload.get("portfolio_id")
        client_id = payload.get("client_id")
        portfolio_value = payload.get("portfolio_value")
        currency = payload.get("currency")
        holdings = payload.get("holdings")
        holding_count = payload.get("holding_count")

        if not isinstance(portfolio_id, str) or not portfolio_id.strip():
            raise ApiPayloadError(
                "Portfolio detail field 'portfolio_id' "
                "must be a non-empty string."
            )

        if not isinstance(client_id, str) or not client_id.strip():
            raise ApiPayloadError(
                "Portfolio detail field 'client_id' "
                "must be a non-empty string."
            )

        if not isinstance(
            portfolio_value,
            (int, float),
        ) or isinstance(portfolio_value, bool):
            raise ApiPayloadError(
                "Portfolio detail field 'portfolio_value' "
                "must be numeric."
            )

        if not isinstance(currency, str) or not currency.strip():
            raise ApiPayloadError(
                "Portfolio detail field 'currency' "
                "must be a non-empty string."
            )

        if not isinstance(holdings, list):
            raise ApiPayloadError(
                "Portfolio detail field 'holdings' must be a list."
            )

        for holding in holdings:
            if not isinstance(holding, dict):
                raise ApiPayloadError(
                    "Every portfolio holding must be an object."
                )

            FastApiClient._validate_portfolio_holding(
                holding,
            )

        if not isinstance(holding_count, int) or isinstance(
            holding_count,
            bool,
        ):
            raise ApiPayloadError(
                "Portfolio detail field 'holding_count' "
                "must be an integer."
            )

        return payload

    @staticmethod
    def _validate_portfolio_holding(
        holding: JsonObject,
    ) -> None:
        """Validate one portfolio holding."""

        asset = holding.get("asset")
        current_weight = holding.get("current_weight")
        current_value = holding.get("current_value")
        cost_basis = holding.get("cost_basis")

        if not isinstance(asset, str) or not asset.strip():
            raise ApiPayloadError(
                "Portfolio holding field 'asset' "
                "must be a non-empty string."
            )

        for field_name, value in {
            "current_weight": current_weight,
            "current_value": current_value,
            "cost_basis": cost_basis,
        }.items():
            if not isinstance(
                value,
                (int, float),
            ) or isinstance(value, bool):
                raise ApiPayloadError(
                    f"Portfolio holding field "
                    f"'{field_name}' must be numeric."
                )