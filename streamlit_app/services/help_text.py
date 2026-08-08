from __future__ import annotations


INPUT_HELP = {
    "portfolio": (
        "Choose the client portfolio you want to review or rebalance."
    ),
    "strategy": (
        "Choose which deterministic strategy simulation to run."
    ),
    "transaction_cost_rate": (
        "Estimated execution cost applied to each buy or sell trade. "
        "For example, 0.002 means 0.20%."
    ),
    "rebalance_threshold": (
        "The amount an asset allocation can move away from target before "
        "a rebalance is triggered."
    ),
    "initial_portfolio_value": (
        "The starting value used for the backtest simulation."
    ),
    "risk_free_rate": (
        "A low-risk reference return used when calculating the Sharpe "
        "Ratio."
    ),
    "periods_per_year": (
        "The number of return periods treated as one year. Daily market "
        "data commonly uses 252 trading days."
    ),
    "tax_rate": (
        "Estimated tax rate applied to realized gains in the simplified "
        "Version 1 tax model."
    ),
    "turnover_budget": (
        "The maximum portion of the portfolio the optimizer may trade in "
        "one rebalance simulation."
    ),
}

METRIC_HELP = {
    "portfolio_value": (
        "The current market value of the portfolio returned by the API."
    ),
    "current_weight": (
        "The asset's current share of total portfolio market value."
    ),
    "target_weight": (
        "The asset's intended allocation from the client's target mix."
    ),
    "risk_category": (
        "The client's investment risk profile. It determines the target "
        "asset allocation used by the backend."
    ),
    "drift": (
        "Difference between current allocation and target allocation."
    ),
    "transaction_cost": (
        "Estimated execution cost for proposed buy and sell trades."
    ),
    "estimated_tax": (
        "Estimated tax on realized gains under PortfolioMind's simplified "
        "Version 1 tax model."
    ),
    "cost_basis": (
        "Approximate original amount invested in the holding."
    ),
    "current_value": (
        "The current market value of this holding."
    ),
    "trade_weight": (
        "The percentage of the total portfolio being bought or sold for "
        "this asset."
    ),
    "trade_value": (
        "The dollar value represented by the proposed trade. The UI shows "
        "the absolute amount alongside the BUY or SELL action."
    ),
    "post_trade_weight": (
        "The portfolio allocation expected after the proposed trade."
    ),
    "threshold_severity": (
        "How strongly the asset's allocation exceeded its permitted drift "
        "range."
    ),
    "approval_required": (
        "Some rebalances require human review when portfolio risk or "
        "allocation drift exceeds configured limits."
    ),
    "total_return": (
        "How much the portfolio gained or lost over the entire period."
    ),
    "annualized_return": (
        "The return expressed as an average annual rate."
    ),
    "volatility": (
        "How much returns varied during the period."
    ),
    "sharpe_ratio": (
        "Return generated per unit of risk, useful for comparing similar "
        "strategies."
    ),
    "maximum_drawdown": (
        "The largest decline from a previous portfolio high."
    ),
    "turnover": (
        "How much of the portfolio was bought or sold."
    ),
    "maximum_drift": (
        "The largest allocation gap from target observed in the data."
    ),
}

CHART_HELP = {
    "current_vs_target": (
        "Compare today's allocation with the target mix defined by the "
        "client's risk profile."
    ),
    "drift": (
        "Positive values are above target; negative values are below target."
    ),
    "composition": (
        "Shows the current portfolio mix as a share of total market value."
    ),
    "rebalance_allocation": (
        "Shows how the proposed rebalance changes the portfolio."
    ),
    "trade_value": (
        "Shows where the proposed rebalance moves money into or out of "
        "the portfolio."
    ),
    "drawdown": (
        "Drawdown shows how far the portfolio value is below its "
        "previous highest value. Larger negative values represent "
        "deeper declines."
    ),
}

STATUS_HELP = {
    "healthy": "The service is responding normally.",
    "ready": "The readiness check succeeded.",
    "not_configured": (
        "This optional capability is disabled because required "
        "configuration is not present."
    ),
    "unavailable": (
        "The service could not be reached from the Streamlit app."
    ),
    "failed": "The service reported a failure state.",
}


def input_help(field_name: str) -> str:
    """Return user-facing help text for an input field."""

    return INPUT_HELP.get(field_name, "")


def metric_help(metric_name: str) -> str:
    """Return user-facing help text for a displayed metric."""

    return METRIC_HELP.get(metric_name, "")


def chart_help(chart_name: str) -> str:
    """Return user-facing help text for a chart."""

    return CHART_HELP.get(chart_name, "")


def status_help(status_name: str) -> str:
    """Return user-facing help text for an operational status."""

    return STATUS_HELP.get(status_name.strip().lower(), "")
