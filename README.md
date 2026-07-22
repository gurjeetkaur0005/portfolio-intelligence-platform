# Agentic AI Autonomous Portfolio Rebalancing Platform

## Overview

The **Agentic AI Autonomous Portfolio Rebalancing Platform** is a
production-style portfolio management project that monitors synthetic
investment portfolios, detects rebalancing triggers, optimizes trades,
estimates transaction costs and taxes, enriches trade records, and generates
client, advisor, and compliance explanations.

The project is intentionally modular so later phases can add human approval,
audit trails, agents, APIs, dashboards, databases, and deployment automation
without rewriting the core rebalancing logic.

## Project Goals

* Generate synthetic client profiles and portfolio allocations.
* Detect portfolio drift from target allocation.
* Evaluate threshold, calendar, and event rebalancing triggers.
* Consolidate trigger decisions by priority.
* Generate optimized buy, sell, and hold recommendations.
* Preserve signed trade conventions across execution, tax, and explanation
  modules.
* Estimate transaction costs and tax impact before execution.
* Produce clear explanations for clients, advisors, and compliance review.

## Technology Stack

* Python
* Pandas
* NumPy
* CVXPY
* Pytest
* Ruff

Future additions:

* Human Approval Engine
* Audit Trail
* Tax Lot Manager
* CrewAI agents
* FastAPI
* PostgreSQL
* Streamlit dashboard
* Backtesting
* AWS deployment

## Project Structure

```text
portfolio-intelligence-platform/
├── config/
│   ├── asset_classes.py
│   ├── risk_categories.py
│   └── settings.py
├── src/
│   ├── data/
│   ├── monitoring/
│   ├── triggers/
│   ├── optimization/
│   ├── execution/
│   ├── explanations/
│   ├── pipeline/
│   ├── schemas/
│   ├── agents/
│   ├── backtesting/
│   ├── dashboard/
│   ├── override/
│   └── utils/
├── tests/
├── requirements.txt
└── README.md
```

## Rebalance Pipeline

The implemented orchestration layer is in `src/pipeline/rebalance_pipeline.py`.

```text
Client Profile Generator
        │
        ▼
Portfolio Generator
        │
        ▼
Drift Calculator
        │
        ▼
Threshold Trigger
Calendar Trigger
Event Trigger
        │
        ▼
Trigger Consolidator
        │
        ▼
Portfolio Optimizer (CVXPY)
        │
        ▼
Trade List Generator
        │
        ▼
Transaction Cost Estimator
        │
        ▼
Trade Enrichment
        │
        ▼
Tax-Aware Optimizer
        │
        ▼
Explanation Generator
```

`src/data/market_data_simulator.py` is currently a standalone synthetic market
data utility. It is not part of the active rebalance pipeline yet.

## Trade Conventions

The project uses one signed trade model:

* `BUY`: `trade_weight > 0` and `trade_value > 0`
* `SELL`: `trade_weight < 0` and `trade_value < 0`
* `HOLD`: `trade_weight == 0` and `trade_value == 0`
* `transaction_cost` is always non-negative
* tax is estimated only from realized gains on sell trades

The canonical trade DataFrame schema is documented in
`src/schemas/trade_schema.py`.

## Completed Modules

### Configuration

Defines asset classes, risk categories, global settings, and reproducible
random seeds.

### Data Generation

* `client_profile_generator.py` creates synthetic client profiles.
* `portfolio_generator.py` creates target and current allocations.
* `market_data_simulator.py` creates standalone synthetic market price data.

### Monitoring

`drift_calculator.py` adds asset-level drift, maximum absolute drift, sum of
absolute drift, RMS drift, and rebalancing flags.

### Trigger Engine

* `threshold_trigger.py` classifies drift breaches as `none`, `medium`,
  `high`, or `critical`.
* `calendar_trigger.py` detects monthly, quarterly, and annual reviews.
* `event_trigger.py` detects supported market, regulatory, client, corporate,
  cash-flow, and tax-harvesting events.
* `trigger_consolidator.py` combines trigger signals and chooses the final
  priority.

### Optimization

`portfolio_optimizer.py` uses CVXPY to minimize tracking error subject to
cash-neutral trades, non-negative post-trade weights, and a turnover budget.

### Execution

* `trade_list_generator.py` turns optimized weights into asset-level BUY,
  SELL, and HOLD rows.
* `transaction_cost_estimator.py` calculates signed trade values and
  non-negative transaction costs.

### Pipeline

* `trade_enrichment.py` adds `portfolio_id`, `current_value`, `cost_basis`,
  and `tax_rate` for tax estimation.
* `rebalance_pipeline.py` orchestrates the full workflow.

### Tax And Explanations

* `tax_aware_optimizer.py` estimates realized gains and tax liability for
  sell trades.
* `explanation_generator.py` creates distinct client, advisor, and compliance
  explanations with threshold context.

## Validation

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run tests, compilation, and lint checks:

```bash
python3 -m pytest -v
python3 -m compileall src config tests
python3 -m ruff check .
```
