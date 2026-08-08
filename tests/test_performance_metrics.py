import numpy as np
import pytest

from src.backtesting.performance_metrics import (
    calculate_annualized_return,
    calculate_drawdown_series,
    calculate_maximum_drawdown,
    calculate_sharpe_ratio,
    calculate_total_return,
    calculate_volatility,
)


# ============================================================
# Total Return Tests
# ============================================================


def test_calculate_total_return_for_profit() -> None:
    portfolio_values = [
        100.0,
        110.0,
        120.0,
    ]

    result = calculate_total_return(portfolio_values)

    assert result == pytest.approx(0.20)


def test_calculate_total_return_for_loss() -> None:
    portfolio_values = [
        100.0,
        90.0,
        80.0,
    ]

    result = calculate_total_return(portfolio_values)

    assert result == pytest.approx(-0.20)


def test_calculate_total_return_for_no_change() -> None:
    portfolio_values = [
        100.0,
        100.0,
    ]

    result = calculate_total_return(portfolio_values)

    assert result == pytest.approx(0.0)


def test_calculate_total_return_requires_two_values() -> None:
    with pytest.raises(
        ValueError,
        match="At least two portfolio values are required.",
    ):
        calculate_total_return([100.0])


def test_calculate_total_return_requires_positive_starting_value() -> None:
    with pytest.raises(
        ValueError,
        match="Starting portfolio value must be positive.",
    ):
        calculate_total_return(
            [
                0.0,
                100.0,
            ]
        )


# ============================================================
# Annualized Return Tests
# ============================================================


def test_calculate_annualized_return_for_one_year() -> None:
    portfolio_values = [
        100.0,
        121.0,
    ]

    result = calculate_annualized_return(
        portfolio_values=portfolio_values,
        years=1.0,
    )

    assert result == pytest.approx(0.21)


def test_calculate_annualized_return_for_multiple_years() -> None:
    portfolio_values = [
        100.0,
        121.0,
    ]

    result = calculate_annualized_return(
        portfolio_values=portfolio_values,
        years=2.0,
    )

    assert result == pytest.approx(0.10)


def test_calculate_annualized_return_for_loss() -> None:
    portfolio_values = [
        100.0,
        81.0,
    ]

    result = calculate_annualized_return(
        portfolio_values=portfolio_values,
        years=2.0,
    )

    assert result == pytest.approx(-0.10)


def test_calculate_annualized_return_requires_two_values() -> None:
    with pytest.raises(
        ValueError,
        match="At least two portfolio values are required.",
    ):
        calculate_annualized_return(
            portfolio_values=[100.0],
            years=1.0,
        )


def test_calculate_annualized_return_requires_positive_years() -> None:
    with pytest.raises(
        ValueError,
        match="Investment period must be positive.",
    ):
        calculate_annualized_return(
            portfolio_values=[
                100.0,
                120.0,
            ],
            years=0.0,
        )


def test_calculate_annualized_return_requires_positive_starting_value() -> None:
    with pytest.raises(
        ValueError,
        match="Starting portfolio value must be positive.",
    ):
        calculate_annualized_return(
            portfolio_values=[
                0.0,
                120.0,
            ],
            years=1.0,
        )


# ============================================================
# Volatility Tests
# ============================================================


def test_calculate_volatility() -> None:
    portfolio_values = [
        100.0,
        110.0,
        105.0,
        115.0,
    ]

    period_returns = np.array(
        [
            110.0 / 100.0 - 1.0,
            105.0 / 110.0 - 1.0,
            115.0 / 105.0 - 1.0,
        ]
    )

    expected_volatility = (
        np.std(
            period_returns,
            ddof=1,
        )
        * np.sqrt(252)
    )

    result = calculate_volatility(portfolio_values)

    assert result == pytest.approx(expected_volatility)


def test_calculate_volatility_with_custom_periods_per_year() -> None:
    portfolio_values = [
        100.0,
        110.0,
        105.0,
        115.0,
    ]

    periods_per_year = 12

    period_returns = np.array(
        [
            110.0 / 100.0 - 1.0,
            105.0 / 110.0 - 1.0,
            115.0 / 105.0 - 1.0,
        ]
    )

    expected_volatility = (
        np.std(
            period_returns,
            ddof=1,
        )
        * np.sqrt(periods_per_year)
    )

    result = calculate_volatility(
        portfolio_values=portfolio_values,
        periods_per_year=periods_per_year,
    )

    assert result == pytest.approx(expected_volatility)


def test_calculate_volatility_for_constant_returns() -> None:
    portfolio_values = [
        100.0,
        110.0,
        121.0,
        133.1,
    ]

    result = calculate_volatility(portfolio_values)

    assert result == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_calculate_volatility_requires_two_values() -> None:
    with pytest.raises(
        ValueError,
        match="At least two portfolio values are required.",
    ):
        calculate_volatility([100.0])


def test_calculate_volatility_requires_positive_periods_per_year() -> None:
    with pytest.raises(
        ValueError,
        match="Periods per year must be positive.",
    ):
        calculate_volatility(
            portfolio_values=[
                100.0,
                110.0,
            ],
            periods_per_year=0,
        )


# ============================================================
# Sharpe Ratio Tests
# ============================================================


def test_calculate_sharpe_ratio() -> None:
    result = calculate_sharpe_ratio(
        annualized_return=0.12,
        annualized_volatility=0.08,
        risk_free_rate=0.04,
    )

    assert result == pytest.approx(1.0)


def test_calculate_sharpe_ratio_with_default_risk_free_rate() -> None:
    result = calculate_sharpe_ratio(
        annualized_return=0.12,
        annualized_volatility=0.06,
    )

    assert result == pytest.approx(2.0)


def test_calculate_negative_sharpe_ratio() -> None:
    result = calculate_sharpe_ratio(
        annualized_return=0.02,
        annualized_volatility=0.10,
        risk_free_rate=0.04,
    )

    assert result == pytest.approx(-0.20)


def test_calculate_sharpe_ratio_rejects_negative_volatility() -> None:
    with pytest.raises(
        ValueError,
        match="Annualized volatility cannot be negative.",
    ):
        calculate_sharpe_ratio(
            annualized_return=0.12,
            annualized_volatility=-0.08,
            risk_free_rate=0.04,
        )


def test_calculate_sharpe_ratio_rejects_zero_volatility() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Sharpe ratio cannot be calculated "
            "when volatility is zero."
        ),
    ):
        calculate_sharpe_ratio(
            annualized_return=0.12,
            annualized_volatility=0.0,
            risk_free_rate=0.04,
        )


# ============================================================
# Maximum Drawdown Tests
# ============================================================


def test_calculate_maximum_drawdown() -> None:
    portfolio_values = [
        100.0,
        120.0,
        110.0,
        150.0,
        140.0,
        90.0,
        160.0,
    ]

    result = calculate_maximum_drawdown(portfolio_values)

    assert result == pytest.approx(-0.40)


def test_calculate_maximum_drawdown_with_no_loss() -> None:
    portfolio_values = [
        100.0,
        110.0,
        120.0,
        130.0,
    ]

    result = calculate_maximum_drawdown(portfolio_values)

    assert result == pytest.approx(0.0)


def test_calculate_maximum_drawdown_for_continuous_decline() -> None:
    portfolio_values = [
        100.0,
        90.0,
        80.0,
        70.0,
    ]

    result = calculate_maximum_drawdown(portfolio_values)

    assert result == pytest.approx(-0.30)


def test_calculate_maximum_drawdown_uses_peak_not_previous_value() -> None:
    portfolio_values = [
        100.0,
        150.0,
        140.0,
        90.0,
    ]

    result = calculate_maximum_drawdown(portfolio_values)

    expected_drawdown = (
        90.0 / 150.0
    ) - 1.0

    assert result == pytest.approx(expected_drawdown)


def test_calculate_drawdown_series() -> None:
    portfolio_values = [
        100.0,
        110.0,
        105.0,
        90.0,
    ]

    result = calculate_drawdown_series(portfolio_values)

    assert result == pytest.approx(
        [
            0.0,
            0.0,
            -0.045454545,
            -0.181818181,
        ]
    )


def test_calculate_drawdown_series_for_continuously_rising_values() -> None:
    portfolio_values = [
        100.0,
        110.0,
        120.0,
    ]

    result = calculate_drawdown_series(portfolio_values)

    assert result == pytest.approx(
        [
            0.0,
            0.0,
            0.0,
        ]
    )


def test_calculate_drawdown_series_for_single_period() -> None:
    result = calculate_drawdown_series([100.0])

    assert result == pytest.approx([0.0])


def test_calculate_drawdown_series_recovers_to_new_high() -> None:
    portfolio_values = [
        100.0,
        90.0,
        111.0,
    ]

    result = calculate_drawdown_series(portfolio_values)

    assert result == pytest.approx(
        [
            0.0,
            -0.10,
            0.0,
        ]
    )


def test_maximum_drawdown_equals_minimum_drawdown_history() -> None:
    portfolio_values = [
        100.0,
        110.0,
        105.0,
        90.0,
    ]

    assert calculate_maximum_drawdown(portfolio_values) == pytest.approx(
        min(calculate_drawdown_series(portfolio_values))
    )


def test_calculate_drawdown_series_rejects_non_finite_values() -> None:
    with pytest.raises(
        ValueError,
        match="Portfolio values must contain only finite numeric values.",
    ):
        calculate_drawdown_series([100.0, float("nan")])


def test_calculate_drawdown_series_rejects_non_positive_values() -> None:
    with pytest.raises(
        ValueError,
        match="Portfolio values must be positive.",
    ):
        calculate_drawdown_series([100.0, 0.0])


def test_calculate_maximum_drawdown_requires_two_values() -> None:
    with pytest.raises(
        ValueError,
        match="At least two portfolio values are required.",
    ):
        calculate_maximum_drawdown([100.0])
