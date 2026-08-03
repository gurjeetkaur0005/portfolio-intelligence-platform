from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any
from uuid import uuid4

import pandas as pd

from src.database.models import (
    ApprovalModel,
    AuditRecordModel,
    PortfolioModel,
    RebalanceRunModel,
    TradeModel,
)
from src.database.repositories import (
    RebalanceRunRepository,
)


REQUIRED_COLUMNS = {
    "portfolio_id",
    "asset",
    "action",
    "current_weight",
    "trade_weight",
    "post_trade_weight",
    "trade_value",
    "transaction_cost",
    "estimated_tax_liability",
    "threshold_breached",
    "threshold_severity",
    "breach_ratio",
    "final_trigger_type",
    "final_priority",
    "contributing_triggers",
    "client_explanation",
    "advisor_explanation",
    "compliance_explanation",
    "approval_required",
    "approval_status",
    "approval_reason",
    "audit_id",
    "audit_timestamp",
}


class RebalancePersistenceService:
    """
    Convert deterministic rebalance output into database models.

    This service performs data translation only. It does not calculate
    portfolio values, trades, costs, taxes, approvals, or explanations.
    """

    def __init__(
        self,
        repository: RebalanceRunRepository,
    ) -> None:
        """Initialize the service with a rebalance repository."""

        if not isinstance(
            repository,
            RebalanceRunRepository,
        ):
            raise TypeError(
                "repository must be a RebalanceRunRepository."
            )

        self._repository = repository

    def persist_rebalance_result(
        self,
        *,
        portfolio: PortfolioModel,
        trade_results: pd.DataFrame,
        portfolio_value: Decimal,
        transaction_cost_rate: Decimal,
        run_id: str | None = None,
        status: str = "SUCCESS",
        completed_at: datetime | None = None,
    ) -> RebalanceRunModel:
        """
        Persist one portfolio's complete rebalance result.

        Args:
            portfolio:
                Existing persisted portfolio.
            trade_results:
                Final DataFrame returned by the rebalance pipeline.
            portfolio_value:
                Portfolio value used for the workflow.
            transaction_cost_rate:
                Transaction-cost rate used for the workflow.
            run_id:
                Optional external run identifier.
            status:
                Workflow completion status.
            completed_at:
                Optional workflow completion timestamp.

        Returns:
            Persisted RebalanceRunModel.
        """

        _validate_portfolio(portfolio)
        _validate_trade_results(trade_results)

        normalized_portfolio_value = _to_decimal(
            portfolio_value,
            "portfolio_value",
        )
        normalized_transaction_cost_rate = _to_decimal(
            transaction_cost_rate,
            "transaction_cost_rate",
        )

        if normalized_portfolio_value <= 0:
            raise ValueError(
                "portfolio_value must be positive."
            )

        if not (
            Decimal("0")
            <= normalized_transaction_cost_rate
            <= Decimal("1")
        ):
            raise ValueError(
                "transaction_cost_rate must be between 0 and 1."
            )

        portfolio_rows = trade_results.loc[
            trade_results["portfolio_id"]
            == portfolio.portfolio_id
        ]

        if portfolio_rows.empty:
            raise ValueError(
                "trade_results does not contain rows for "
                f"portfolio {portfolio.portfolio_id!r}."
            )

        unique_portfolio_ids = set(
            trade_results["portfolio_id"].astype(str)
        )

        if unique_portfolio_ids != {
            portfolio.portfolio_id
        }:
            raise ValueError(
                "trade_results must contain exactly one portfolio."
            )

        rebalance_run = RebalanceRunModel(
            run_id=(
                _validate_non_empty_string(
                    run_id,
                    "run_id",
                )
                if run_id is not None
                else _generate_run_id()
            ),
            status=_validate_non_empty_string(
                status,
                "status",
            ),
            portfolio_value=normalized_portfolio_value,
            transaction_cost_rate=(
                normalized_transaction_cost_rate
            ),
            completed_at=completed_at,
        )

        for _, row in portfolio_rows.iterrows():
            rebalance_run.trades.append(
                _build_trade_model(row)
            )

        portfolio.rebalance_runs.append(
            rebalance_run
        )

        return self._repository.save_rebalance_run(
            rebalance_run
        )


def _build_trade_model(
    row: pd.Series,
) -> TradeModel:
    """Convert one final pipeline row into a TradeModel graph."""

    trade = TradeModel(
        asset=_required_string(row, "asset"),
        action=_required_string(row, "action"),
        current_weight=_required_decimal(
            row,
            "current_weight",
        ),
        trade_weight=_required_decimal(
            row,
            "trade_weight",
        ),
        post_trade_weight=_required_decimal(
            row,
            "post_trade_weight",
        ),
        trade_value=_required_decimal(
            row,
            "trade_value",
        ),
        transaction_cost=_required_decimal(
            row,
            "transaction_cost",
        ),
        estimated_tax_liability=_required_decimal(
            row,
            "estimated_tax_liability",
        ),
        threshold_breached=_required_bool(
            row,
            "threshold_breached",
        ),
        threshold_severity=_required_string(
            row,
            "threshold_severity",
        ),
        breach_ratio=_required_decimal(
            row,
            "breach_ratio",
        ),
        final_trigger_type=_required_string(
            row,
            "final_trigger_type",
        ),
        final_priority=_required_string(
            row,
            "final_priority",
        ),
        contributing_triggers=_required_string(
            row,
            "contributing_triggers",
        ),
        client_explanation=_required_string(
            row,
            "client_explanation",
        ),
        advisor_explanation=_required_string(
            row,
            "advisor_explanation",
        ),
        compliance_explanation=_required_string(
            row,
            "compliance_explanation",
        ),
    )

    trade.approval = ApprovalModel(
        approval_required=_required_bool(
            row,
            "approval_required",
        ),
        approval_status=_required_string(
            row,
            "approval_status",
        ),
        approval_reason=_required_string(
            row,
            "approval_reason",
        ),
        reviewed_by=_optional_string(
            row,
            "reviewed_by",
        ),
        reviewed_at=_optional_datetime(
            row,
            "reviewed_at",
        ),
    )

    trade.audit_record = AuditRecordModel(
        audit_id=_required_string(
            row,
            "audit_id",
        ),
        audit_timestamp=_required_datetime(
            row,
            "audit_timestamp",
        ),
        event_type=(
            _optional_string(
                row,
                "audit_event_type",
            )
            or "TRADE_RECOMMENDATION"
        ),
        details=(
            _optional_string(
                row,
                "audit_details",
            )
            or _build_default_audit_details(row)
        ),
    )

    return trade


def _validate_trade_results(
    trade_results: pd.DataFrame,
) -> None:
    """Validate final pipeline output before translation."""

    if not isinstance(trade_results, pd.DataFrame):
        raise TypeError(
            "trade_results must be a pandas DataFrame."
        )

    missing_columns = (
        REQUIRED_COLUMNS
        - set(trade_results.columns)
    )

    if missing_columns:
        formatted_columns = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            "trade_results is missing required columns: "
            f"{formatted_columns}."
        )

    if trade_results.empty:
        raise ValueError(
            "trade_results must not be empty."
        )


def _validate_portfolio(
    portfolio: PortfolioModel,
) -> None:
    """Validate the target portfolio model."""

    if not isinstance(portfolio, PortfolioModel):
        raise TypeError(
            "portfolio must be a PortfolioModel."
        )

    if portfolio.id is None:
        raise ValueError(
            "portfolio must be persisted before "
            "saving a rebalance run."
        )


def _required_string(
    row: pd.Series,
    column: str,
) -> str:
    """Read and validate a required string column."""

    value = row[column]

    if not isinstance(value, str):
        raise TypeError(
            f"{column} must contain strings."
        )

    return _validate_non_empty_string(
        value,
        column,
    )


def _optional_string(
    row: pd.Series,
    column: str,
) -> str | None:
    """Read an optional string column."""

    if column not in row.index:
        return None

    value = row[column]

    if pd.isna(value):
        return None

    if not isinstance(value, str):
        raise TypeError(
            f"{column} must contain strings or missing values."
        )

    normalized_value = value.strip()

    return normalized_value or None


def _required_decimal(
    row: pd.Series,
    column: str,
) -> Decimal:
    """Read and convert a required numeric column."""

    return _to_decimal(
        row[column],
        column,
    )


def _required_bool(
    row: pd.Series,
    column: str,
) -> bool:
    """Read a strict Boolean column."""

    value = row[column]

    if not isinstance(value, bool):
        raise TypeError(
            f"{column} must contain Boolean values."
        )

    return value


def _required_datetime(
    row: pd.Series,
    column: str,
) -> datetime:
    """Read a required datetime column."""

    value = row[column]

    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()

    if isinstance(value, datetime):
        return value

    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError(
                f"{column} must contain a valid ISO datetime."
            ) from error

    raise TypeError(
        f"{column} must contain datetime values."
    )


def _optional_datetime(
    row: pd.Series,
    column: str,
) -> datetime | None:
    """Read an optional datetime column."""

    if column not in row.index:
        return None

    value = row[column]

    if pd.isna(value):
        return None

    return _required_datetime(
        row,
        column,
    )


def _to_decimal(
    value: Any,
    field_name: str,
) -> Decimal:
    """Convert a finite numeric value into Decimal."""

    if isinstance(value, bool):
        raise TypeError(
            f"{field_name} must be numeric."
        )

    if pd.isna(value):
        raise ValueError(
            f"{field_name} must not be missing."
        )

    try:
        decimal_value = Decimal(
            str(value)
        )
    except (
        InvalidOperation,
        TypeError,
        ValueError,
    ) as error:
        raise TypeError(
            f"{field_name} must be numeric."
        ) from error

    if not decimal_value.is_finite():
        raise ValueError(
            f"{field_name} must be finite."
        )

    return decimal_value


def _validate_non_empty_string(
    value: str,
    field_name: str,
) -> str:
    """Validate and normalize a required string."""

    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} must be a string."
        )

    normalized_value = value.strip()

    if not normalized_value:
        raise ValueError(
            f"{field_name} must not be empty."
        )

    return normalized_value


def _build_default_audit_details(
    row: pd.Series,
) -> str:
    """Build deterministic audit details from persisted facts."""

    return (
        f"Trade recommendation recorded for "
        f"{_required_string(row, 'asset')} with action "
        f"{_required_string(row, 'action')}."
    )


def _generate_run_id() -> str:
    """Generate a unique external rebalance run identifier."""

    return f"RUN-{uuid4().hex.upper()}"