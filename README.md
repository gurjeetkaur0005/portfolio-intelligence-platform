# Agentic AI Autonomous Portfolio Rebalancing Platform

## Overview

The **Agentic AI Autonomous Portfolio Rebalancing Platform** is a production-style portfolio management system that automatically monitors investment portfolios, detects when rebalancing is required, optimizes trades, estimates transaction costs and taxes, and prepares the system for explainable AI and autonomous decision-making.

The goal of this project is to simulate how modern digital wealth management platforms such as Wealthfront, Betterment, and institutional portfolio management systems operate while following clean software engineering practices.

---

# Project Goals

* Monitor thousands of investment portfolios.
* Detect portfolio drift from target allocation.
* Trigger rebalancing using multiple trigger types.
* Generate optimized buy/sell trades.
* Minimize unnecessary turnover.
* Estimate transaction costs.
* Estimate tax impact before execution.
* Produce explainable and auditable investment decisions.
* Serve as a production-quality portfolio for interviews and resume projects.

---

# Technology Stack

* Python
* Pandas
* NumPy
* CVXPY
* Pytest
* Git & GitHub

Future additions:

* FastAPI
* PostgreSQL
* Docker
* Redis
* CrewAI / LangChain
* AWS

---

# Project Structure

```text
portfolio-intelligence-platform/
│
├── config/
│   ├── asset_classes.py
│   ├── risk_categories.py
│   └── settings.py
│
├── src/
│   ├── data/
│   ├── monitoring/
│   ├── triggers/
│   ├── optimization/
│   ├── explanations/
│   ├── execution/
│   └── agents/
│
├── tests/
│
├── notebooks/
│
└── docs/
```

---

# System Architecture

```text
Client Profiles
        │
        ▼
Portfolio Generator
        │
        ▼
Market Data Simulator
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
Tax-Aware Optimizer
        │
        ▼
Explainability Engine (Upcoming)
```

---

# Completed Modules

## 1. Configuration

Configured:

* Asset classes
* Investor risk categories
* Global project settings
* Random seed
* Portfolio generation parameters

---

## 2. Data Generation

### Client Profile Generator

Generates synthetic client information including:

* Portfolio ID
* Risk category
* Investment horizon
* Tax bracket
* Manual approval requirement

### Portfolio Generator

Generates diversified investment portfolios.

Features:

* Target asset allocation
* Current asset allocation
* Simulated market drift
* Multiple investor risk profiles

### Market Data Simulator

Creates synthetic market returns for:

* Domestic equity
* International equity
* Fixed income
* Real estate
* Commodities
* Cash

---

## 3. Portfolio Monitoring

### Drift Calculator

Measures portfolio deviation from target allocation.

Calculates:

* Asset-level drift
* Maximum absolute drift
* Sum of absolute drift
* RMS drift

Determines whether rebalancing is required.

---

## 4. Trigger Engine

### Threshold Trigger

Detects portfolios exceeding allowable drift.

Severity levels:

* None
* Medium
* High
* Critical

---

### Calendar Trigger

Supports:

* Monthly review
* Quarterly review
* Annual review

---

### Event Trigger

Supports:

* Market crash
* Regulatory change
* Client life event
* Corporate action
* Large cash flow
* Tax harvesting window

---

### Trigger Consolidator

Combines:

* Threshold triggers
* Calendar triggers
* Event triggers

Produces:

* Final trigger
* Final priority
* Contributing trigger list

---

## 5. Portfolio Optimization

### Portfolio Optimizer

Built using CVXPY.

Features:

* Target allocation optimization
* Turnover constraints
* Trade weight generation
* Optimization under portfolio constraints

---

### Trade List Generator

Converts optimized weights into executable trades.

Produces:

* Buy trades
* Sell trades
* Trade values
* Asset-level recommendations

---

### Transaction Cost Estimator

Estimates execution costs for every trade.

Calculates:

* Transaction cost per trade
* Portfolio transaction cost

---

### Tax-Aware Optimizer

Estimates tax impact before executing trades.

Calculates:

* Unrealized gain
* Sell value
* Sell fraction
* Estimated realized gain
* Estimated tax liability
* After-tax trade value
* Portfolio-level estimated tax
* Tax liability indicator

Validation includes:

* Required columns
* Missing values
* Invalid tax rates
* Invalid holdings
* Excessive sell orders

---

# Testing

The project uses **Pytest** for unit testing.

Current coverage includes:

* Portfolio optimization
* Trade list generation
* Transaction cost estimation
* Tax-aware optimization

Tests validate:

* Correct calculations
* Validation logic
* Edge cases
* Error handling

Run all tests:

```bash
python3 -m pytest -v
```

Run a single test file:

```bash
python3 -m pytest tests/test_tax_aware_optimizer.py -v
```

---

# Current Workflow

1. Generate synthetic clients.
2. Generate portfolios.
3. Simulate market movement.
4. Calculate portfolio drift.
5. Evaluate threshold, calendar, and event triggers.
6. Consolidate trigger decisions.
7. Optimize portfolio allocations.
8. Generate executable trades.
9. Estimate transaction costs.
10. Estimate tax impact.

---

# Upcoming Modules

The next development stages include:

* Explainability Engine
* Human Approval / Override Engine
* Audit Trail
* Agentic AI Layer
* FastAPI Service
* PostgreSQL Integration
* Background Scheduling
* Dashboard
* Historical Backtesting
* Cloud Deployment

---

# Future Enhancements

* Autonomous multi-agent workflow
* Real-time market data integration
* Tax-loss harvesting strategies
* Portfolio performance analytics
* Compliance reporting
* Advisor dashboard
* Client-facing recommendation reports

---

# Learning Outcomes

This project demonstrates practical experience in:

* Portfolio optimization
* Financial engineering fundamentals
* Data processing with Pandas
* Convex optimization using CVXPY
* Production-quality Python development
* Software architecture
* Unit testing with Pytest
* Clean code and modular design
* Building AI-ready financial systems

---

# License

This project is intended for educational, research, and portfolio demonstration purposes.
