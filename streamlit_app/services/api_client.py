from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, overload

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


REBALANCE_CONFLICT_MESSAGE = (
    "A rebalance is already running for this portfolio. "
    "Please wait for it to finish."
)


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

    @overload
    def run_buy_and_hold_backtest(
        self,
        payload: JsonObject,
    ) -> JsonObject:
        ...

    @overload
    def run_buy_and_hold_backtest(
        self,
        payload: None = None,
        *,
        asset_names: list[str],
        market_returns: list[list[float]],
        initial_weights: list[float],
        initial_portfolio_value: float = 100_000.0,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> JsonObject:
        ...

    def run_buy_and_hold_backtest(
        self,
        payload: JsonObject | None = None,
        *,
        asset_names: list[str] | None = None,
        market_returns: list[list[float]] | None = None,
        initial_weights: list[float] | None = None,
        initial_portfolio_value: float = 100_000.0,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> JsonObject:
        """Run a Buy & Hold backtest through FastAPI."""

        request_payload = (
            payload
            if payload is not None
            else _backtest_payload_from_inputs(
                asset_names=asset_names,
                market_returns=market_returns,
                initial_weights=initial_weights,
                initial_portfolio_value=initial_portfolio_value,
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
            )
        )

        response = self._post_json(
            "/backtests/buy-and-hold",
            json=request_payload,
        )

        return self._parse_backtest_response(response)

    @overload
    def run_threshold_backtest(
        self,
        payload: JsonObject,
    ) -> JsonObject:
        ...

    @overload
    def run_threshold_backtest(
        self,
        payload: None = None,
        *,
        asset_names: list[str],
        market_returns: list[list[float]],
        initial_weights: list[float],
        target_weights: list[float],
        initial_portfolio_value: float = 100_000.0,
        drift_band: float = 0.05,
        transaction_cost_rate: float = 0.002,
        tax_rate: float = 0.20,
        turnover_budget: float = 0.10,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
        portfolio_id: str = "STREAMLIT-BACKTEST",
    ) -> JsonObject:
        ...

    def run_threshold_backtest(
        self,
        payload: JsonObject | None = None,
        *,
        asset_names: list[str] | None = None,
        market_returns: list[list[float]] | None = None,
        initial_weights: list[float] | None = None,
        target_weights: list[float] | None = None,
        initial_portfolio_value: float = 100_000.0,
        drift_band: float = 0.05,
        transaction_cost_rate: float = 0.002,
        tax_rate: float = 0.20,
        turnover_budget: float = 0.10,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
        portfolio_id: str = "STREAMLIT-BACKTEST",
    ) -> JsonObject:
        """Run a Threshold Rebalancing backtest through FastAPI."""

        request_payload = (
            payload
            if payload is not None
            else _threshold_payload_from_inputs(
                asset_names=asset_names,
                market_returns=market_returns,
                initial_weights=initial_weights,
                target_weights=target_weights,
                initial_portfolio_value=initial_portfolio_value,
                drift_band=drift_band,
                transaction_cost_rate=transaction_cost_rate,
                tax_rate=tax_rate,
                turnover_budget=turnover_budget,
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
                portfolio_id=portfolio_id,
            )
        )

        response = self._post_json(
            "/backtests/threshold-rebalancing",
            json=request_payload,
        )

        return self._parse_backtest_response(response)

    @overload
    def compare_strategies(
        self,
        payload: JsonObject,
    ) -> JsonObject:
        ...

    @overload
    def compare_strategies(
        self,
        payload: None = None,
        *,
        buy_and_hold: JsonObject,
        threshold_rebalancing: JsonObject,
    ) -> JsonObject:
        ...

    def compare_strategies(
        self,
        payload: JsonObject | None = None,
        *,
        buy_and_hold: JsonObject | None = None,
        threshold_rebalancing: JsonObject | None = None,
    ) -> JsonObject:
        """Compare strategy metrics through FastAPI."""

        request_payload = (
            payload
            if payload is not None
            else _comparison_payload_from_backtests(
                buy_and_hold=buy_and_hold,
                threshold_rebalancing=threshold_rebalancing,
            )
        )

        response = self._post_json(
            "/strategy-comparisons",
            json=request_payload,
        )

        return self._parse_strategy_comparison_response(
            response
        )

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

    def get_llm_health(self) -> JsonObject:
        """Return language-model health information."""

        return self._get_json("/llm/health")

    @staticmethod
    def _raise_for_status(
        response: Response,
    ) -> None:
        """Raise a safe frontend error for unsuccessful responses."""

        if response.ok:
            return

        message = "The FastAPI service returned an error."

        if response.status_code == 409:
            raise ApiResponseError(
                status_code=response.status_code,
                message=REBALANCE_CONFLICT_MESSAGE,
            )

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
    def _parse_backtest_response(
        payload: JsonObject,
    ) -> JsonObject:
        """Validate a backtest response payload."""

        FastApiClient._required_string(
            payload.get("strategy_name"),
            "strategy_name",
        )
        FastApiClient._required_object(
            payload.get("metrics"),
            "metrics",
        )
        FastApiClient._required_object_list(
            payload.get("portfolio_history"),
            "portfolio_history",
        )
        FastApiClient._required_integer(
            payload.get("history_record_count"),
            "history_record_count",
        )

        _validate_metrics_object(
            FastApiClient._required_object(
                payload.get("metrics"),
                "metrics",
            )
        )

        return payload

    @staticmethod
    def _parse_strategy_comparison_response(
        payload: JsonObject,
    ) -> JsonObject:
        """Validate a strategy-comparison response."""

        for field_name in (
            "buy_and_hold",
            "threshold_rebalancing",
        ):
            _validate_strategy_metrics_object(
                FastApiClient._required_object(
                    payload.get(field_name),
                    field_name,
                )
            )

        FastApiClient._required_string(
            payload.get("performance_summary"),
            "performance_summary",
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
    def _required_object(
        value: JsonValue,
        field_name: str,
    ) -> JsonObject:
        """Return a validated JSON object."""

        if not isinstance(value, dict):
            raise ApiPayloadError(
                f"Response field '{field_name}' must be an object."
            )

        return value

    @staticmethod
    def _required_object_list(
        value: JsonValue,
        field_name: str,
    ) -> list[JsonObject]:
        """Return a validated list of JSON objects."""

        if not isinstance(value, list):
            raise ApiPayloadError(
                f"Response field '{field_name}' must be a list."
            )

        objects: list[JsonObject] = []

        for item in value:
            if not isinstance(item, dict):
                raise ApiPayloadError(
                    f"Response field '{field_name}' must "
                    "contain only objects."
                )
            objects.append(item)

        return objects

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


def _backtest_payload_from_inputs(
    *,
    asset_names: list[str] | None,
    market_returns: list[list[float]] | None,
    initial_weights: list[float] | None,
    initial_portfolio_value: float,
    risk_free_rate: float,
    periods_per_year: int,
) -> JsonObject:
    """Build a shared backtest request payload."""

    if asset_names is None:
        raise ValueError("asset_names must be provided.")

    if market_returns is None:
        raise ValueError("market_returns must be provided.")

    if initial_weights is None:
        raise ValueError("initial_weights must be provided.")

    _validate_backtest_inputs(
        asset_names=asset_names,
        market_returns=market_returns,
        initial_weights=initial_weights,
        initial_portfolio_value=initial_portfolio_value,
        periods_per_year=periods_per_year,
    )

    return {
        "asset_names": [str(asset) for asset in asset_names],
        "market_returns": _json_float_matrix(market_returns),
        "initial_weights": _json_float_list(initial_weights),
        "initial_portfolio_value": initial_portfolio_value,
        "risk_free_rate": risk_free_rate,
        "periods_per_year": periods_per_year,
    }


def _threshold_payload_from_inputs(
    *,
    asset_names: list[str] | None,
    market_returns: list[list[float]] | None,
    initial_weights: list[float] | None,
    target_weights: list[float] | None,
    initial_portfolio_value: float,
    drift_band: float,
    transaction_cost_rate: float,
    tax_rate: float,
    turnover_budget: float,
    risk_free_rate: float,
    periods_per_year: int,
    portfolio_id: str,
) -> JsonObject:
    """Build a threshold backtest request payload."""

    if target_weights is None:
        raise ValueError("target_weights must be provided.")

    payload = _backtest_payload_from_inputs(
        asset_names=asset_names,
        market_returns=market_returns,
        initial_weights=initial_weights,
        initial_portfolio_value=initial_portfolio_value,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )
    payload["target_weights"] = _json_float_list(
        target_weights
    )
    payload["drift_band"] = drift_band
    payload["transaction_cost_rate"] = (
        transaction_cost_rate
    )
    payload["tax_rate"] = tax_rate
    payload["turnover_budget"] = turnover_budget
    payload["portfolio_id"] = portfolio_id

    return payload


def _comparison_payload_from_backtests(
    *,
    buy_and_hold: JsonObject | None,
    threshold_rebalancing: JsonObject | None,
) -> JsonObject:
    """Build a comparison request from backtest responses."""

    if buy_and_hold is None:
        raise ValueError("buy_and_hold must be provided.")

    if threshold_rebalancing is None:
        raise ValueError(
            "threshold_rebalancing must be provided."
        )

    return {
        "buy_and_hold": (
            _strategy_metrics_payload(
                buy_and_hold,
                include_threshold_costs=False,
            )
        ),
        "threshold_rebalancing": (
            _strategy_metrics_payload(
                threshold_rebalancing,
                include_threshold_costs=True,
            )
        ),
    }


def _validate_backtest_inputs(
    *,
    asset_names: list[str],
    market_returns: list[list[float]],
    initial_weights: list[float],
    initial_portfolio_value: float,
    periods_per_year: int,
) -> None:
    """Validate common backtest request inputs."""

    if not asset_names:
        raise ValueError("asset_names must not be empty.")

    if not market_returns:
        raise ValueError("market_returns must not be empty.")

    if len(asset_names) != len(initial_weights):
        raise ValueError(
            "asset_names and initial_weights must have the same length."
        )

    for row in market_returns:
        if len(row) != len(asset_names):
            raise ValueError(
                "Each market return row must match asset_names."
            )

    if initial_portfolio_value <= 0:
        raise ValueError(
            "initial_portfolio_value must be greater than zero."
        )

    if periods_per_year <= 0:
        raise ValueError(
            "periods_per_year must be greater than zero."
        )


def _json_float_list(
    values: list[float],
) -> list[JsonValue]:
    """Return floats as JSON values."""

    return [
        float(value)
        for value in values
    ]


def _json_float_matrix(
    rows: list[list[float]],
) -> list[JsonValue]:
    """Return float rows as JSON values."""

    return [
        _json_float_list(row)
        for row in rows
    ]


def _validate_metrics_object(
    metrics: JsonObject,
) -> None:
    """Validate common backtest metrics."""

    for field_name in (
        "total_return",
        "annualized_return",
        "volatility",
        "sharpe_ratio",
        "maximum_drawdown",
    ):
        FastApiClient._required_number(
            metrics.get(field_name),
            field_name,
        )


def _validate_strategy_metrics_object(
    metrics: JsonObject,
) -> None:
    """Validate comparison metrics."""

    FastApiClient._required_string(
        metrics.get("strategy_name"),
        "strategy_name",
    )
    _validate_metrics_object(metrics)

    for field_name in (
        "transaction_costs",
        "taxes_paid",
        "total_implementation_cost",
    ):
        FastApiClient._required_number(
            metrics.get(field_name),
            field_name,
        )

    FastApiClient._required_integer(
        metrics.get("number_of_rebalances"),
        "number_of_rebalances",
    )


def _strategy_metrics_payload(
    backtest_response: JsonObject,
    *,
    include_threshold_costs: bool,
) -> JsonObject:
    """Build strategy comparison metrics from a backtest response."""

    metrics = FastApiClient._required_object(
        backtest_response.get("metrics"),
        "metrics",
    )
    _validate_metrics_object(metrics)

    payload: JsonObject = {
        "total_return": FastApiClient._required_number(
            metrics.get("total_return"),
            "total_return",
        ),
        "annualized_return": FastApiClient._required_number(
            metrics.get("annualized_return"),
            "annualized_return",
        ),
        "volatility": FastApiClient._required_number(
            metrics.get("volatility"),
            "volatility",
        ),
        "sharpe_ratio": FastApiClient._required_number(
            metrics.get("sharpe_ratio"),
            "sharpe_ratio",
        ),
        "maximum_drawdown": FastApiClient._required_number(
            metrics.get("maximum_drawdown"),
            "maximum_drawdown",
        ),
    }

    if include_threshold_costs:
        history = FastApiClient._required_object_list(
            backtest_response.get("portfolio_history"),
            "portfolio_history",
        )
        payload["transaction_costs"] = _sum_history_number(
            history,
            "transaction_cost",
        )
        payload["taxes_paid"] = _sum_history_number(
            history,
            "estimated_tax_liability",
        )
        payload["number_of_rebalances"] = _sum_history_integer(
            history,
            "rebalanced",
        )

    return payload


def _sum_history_number(
    history: list[JsonObject],
    field_name: str,
) -> float:
    """Sum one numeric field from history rows."""

    total = 0.0

    for row in history:
        value = row.get(field_name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total += float(value)

    return total


def _sum_history_integer(
    history: list[JsonObject],
    field_name: str,
) -> int:
    """Sum truthy/integer values from history rows."""

    total = 0

    for row in history:
        value = row.get(field_name)
        if isinstance(value, bool):
            total += int(value)
        elif isinstance(value, int):
            total += value

    return total
    
