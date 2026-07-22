from __future__ import annotations

import math

import pandas as pd


REQUIRED_COLUMNS = {
    "portfolio_id",
    "asset",
    "action",
    "trade_value",
    "prior_approval_required",
    "threshold_severity",
}

ALLOWED_ACTIONS = {
    "BUY",
    "SELL",
    "HOLD",
}

ALLOWED_THRESHOLD_SEVERITIES = {
    "none",
    "medium",
    "high",
    "critical",
}

APPROVAL_REQUIRED_STATUS = "PENDING"
APPROVAL_NOT_REQUIRED_STATUS = "NOT_REQUIRED"

DEFAULT_APPROVAL_THRESHOLD = 100_000.0
ZERO_TOLERANCE = 1e-9


def evaluate_trade_approvals(
    trades: pd.DataFrame,
    approval_threshold: float = DEFAULT_APPROVAL_THRESHOLD,
) -> pd.DataFrame:
    """
    Determine whether each proposed trade requires human approval.

    Approval is required when at least one of the following is true:

    - The client requires prior approval.
    - The absolute trade value is greater than or equal to the configured
      approval threshold.
    - The portfolio has a critical threshold breach.

    HOLD rows do not require approval because no trade will be executed.

    The input DataFrame is not mutated.
    """

    _validate_inputs(
        trades=trades,
        approval_threshold=approval_threshold,
    )

    result = trades.copy()

    approval_results = result.apply(
        lambda row: _evaluate_trade_row(
            row=row,
            approval_threshold=approval_threshold,
        ),
        axis=1,
        result_type="expand",
    )

    approval_results.columns = [
        "approval_required",
        "approval_status",
        "approval_reason",
    ]

    result[
        [
            "approval_required",
            "approval_status",
            "approval_reason",
        ]
    ] = approval_results

    return result


def _validate_inputs(
    trades: pd.DataFrame,
    approval_threshold: float,
) -> None:
    """
    Validate the trade DataFrame and approval configuration.
    """

    if not isinstance(trades, pd.DataFrame):
        raise TypeError("Trades must be provided as a pandas DataFrame.")

    missing_columns = REQUIRED_COLUMNS.difference(trades.columns)

    if missing_columns:
        missing_column_list = ", ".join(sorted(missing_columns))

        raise ValueError(
            "Trades are missing required columns: "
            f"{missing_column_list}"
        )

    _validate_approval_threshold(approval_threshold)
    _validate_actions(trades)
    _validate_trade_values(trades)
    _validate_prior_approval_flags(trades)
    _validate_threshold_severities(trades)


def _validate_approval_threshold(
    approval_threshold: float,
) -> None:
    """
    Validate the configured monetary approval threshold.
    """

    if isinstance(approval_threshold, bool):
        raise ValueError(
            "Approval threshold must be a finite positive number."
        )

    try:
        numeric_threshold = float(approval_threshold)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Approval threshold must be a finite positive number."
        ) from error

    if (
        not math.isfinite(numeric_threshold)
        or numeric_threshold <= 0.0
    ):
        raise ValueError(
            "Approval threshold must be a finite positive number."
        )


def _validate_actions(
    trades: pd.DataFrame,
) -> None:
    """
    Validate trade actions.
    """

    invalid_actions = set(trades["action"]).difference(
        ALLOWED_ACTIONS
    )

    if invalid_actions:
        invalid_action_list = ", ".join(
            sorted(str(action) for action in invalid_actions)
        )

        raise ValueError(
            f"Invalid trade actions: {invalid_action_list}"
        )


def _validate_trade_values(
    trades: pd.DataFrame,
) -> None:
    """
    Validate trade values and their action sign conventions.
    """

    numeric_trade_values = pd.to_numeric(
        trades["trade_value"],
        errors="coerce",
    )

    invalid_values = (
        numeric_trade_values.isna()
        | ~numeric_trade_values.map(math.isfinite)
    )

    if invalid_values.any():
        raise ValueError(
            "Trade values must contain only finite numeric values."
        )

    for action, trade_value in zip(
        trades["action"],
        numeric_trade_values,
        strict=True,
    ):
        if action == "BUY" and trade_value <= ZERO_TOLERANCE:
            raise ValueError(
                "BUY trades must have a positive trade value."
            )

        if action == "SELL" and trade_value >= -ZERO_TOLERANCE:
            raise ValueError(
                "SELL trades must have a negative trade value."
            )

        if (
            action == "HOLD"
            and abs(trade_value) > ZERO_TOLERANCE
        ):
            raise ValueError(
                "HOLD trades must have a zero trade value."
            )


def _validate_prior_approval_flags(
    trades: pd.DataFrame,
) -> None:
    """
    Ensure prior-approval values are actual booleans.
    """

    invalid_flags = trades[
        "prior_approval_required"
    ].map(
        lambda value: not isinstance(value, bool)
    )

    if invalid_flags.any():
        raise ValueError(
            "prior_approval_required must contain only "
            "boolean values."
        )


def _validate_threshold_severities(
    trades: pd.DataFrame,
) -> None:
    """
    Validate portfolio threshold severity values.
    """

    invalid_severities = set(
        trades["threshold_severity"]
    ).difference(
        ALLOWED_THRESHOLD_SEVERITIES
    )

    if invalid_severities:
        invalid_severity_list = ", ".join(
            sorted(
                str(severity)
                for severity in invalid_severities
            )
        )

        raise ValueError(
            "Invalid threshold severities: "
            f"{invalid_severity_list}"
        )


def _evaluate_trade_row(
    row: pd.Series,
    approval_threshold: float,
) -> tuple[bool, str, str]:
    """
    Evaluate approval rules for one trade row.
    """

    if row["action"] == "HOLD":
        return (
            False,
            APPROVAL_NOT_REQUIRED_STATUS,
            "No trade execution is required.",
        )

    approval_reasons = []

    if row["prior_approval_required"]:
        approval_reasons.append(
            "Client requires prior approval"
        )

    if abs(float(row["trade_value"])) >= approval_threshold:
        approval_reasons.append(
            "Trade value exceeds the approval threshold"
        )

    if row["threshold_severity"] == "critical":
        approval_reasons.append(
            "Portfolio has a critical threshold breach"
        )

    if approval_reasons:
        return (
            True,
            APPROVAL_REQUIRED_STATUS,
            "; ".join(approval_reasons) + ".",
        )

    return (
        False,
        APPROVAL_NOT_REQUIRED_STATUS,
        "Trade satisfies automatic approval rules.",
    )