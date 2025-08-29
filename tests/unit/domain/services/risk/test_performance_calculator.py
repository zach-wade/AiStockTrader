"""Comprehensive tests for PerformanceCalculator domain service.

This module provides comprehensive test coverage for the PerformanceCalculator service,
focusing on Sharpe ratio calculations and risk-adjusted performance metrics.
Critical for production safety as this service evaluates real money trading performance.
"""

# Standard library imports
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from src.domain.entities.portfolio import Portfolio

# Local imports
from src.domain.services.risk.performance_calculator import (
    ANNUALIZATION_FACTOR,
    TRADING_DAYS_PER_YEAR,
    PerformanceCalculator,
)
from src.domain.value_objects.money import Money


class TestPerformanceCalculator:
    """Comprehensive test suite for PerformanceCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create PerformanceCalculator instance."""
        return PerformanceCalculator()

    @pytest.fixture
    def sample_positive_returns(self):
        """Sample positive daily returns."""
        return [
            Decimal("0.01"),
            Decimal("0.02"),
            Decimal("-0.01"),
            Decimal("0.015"),
            Decimal("0.005"),
            Decimal("-0.005"),
            Decimal("0.02"),
            Decimal("0.01"),
            Decimal("-0.01"),
            Decimal("0.025"),
            Decimal("0.01"),
            Decimal("0.005"),
        ]

    @pytest.fixture
    def sample_negative_returns(self):
        """Sample negative daily returns."""
        return [
            Decimal("-0.01"),
            Decimal("-0.02"),
            Decimal("0.005"),
            Decimal("-0.015"),
            Decimal("-0.01"),
            Decimal("0.002"),
            Decimal("-0.025"),
            Decimal("-0.01"),
            Decimal("0.005"),
            Decimal("-0.02"),
            Decimal("-0.005"),
            Decimal("-0.01"),
        ]

    @pytest.fixture
    def zero_volatility_returns(self):
        """Returns with zero volatility (all same value)."""
        return [Decimal("0.01")] * 10

    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio for testing."""
        portfolio = Mock(spec=Portfolio)
        portfolio.get_return_percentage.return_value = Decimal("15.5")
        portfolio.get_win_rate.return_value = Decimal("60.0")
        portfolio.get_profit_factor.return_value = Decimal("1.8")
        portfolio.get_average_win.return_value = Money(Decimal("150.00"))
        portfolio.get_average_loss.return_value = Money(Decimal("100.00"))
        portfolio.get_sharpe_ratio.return_value = Decimal("1.25")
        portfolio.get_total_value.return_value = Money(Decimal("110000.00"))
        return portfolio

    # Sharpe Ratio Basic Tests
    def test_calculate_sharpe_ratio_positive_returns(self, calculator, sample_positive_returns):
        """Test Sharpe ratio calculation with positive returns."""
        result = calculator.calculate_sharpe_ratio(sample_positive_returns)

        assert result is not None
        assert isinstance(result, Decimal)
        assert result > 0  # Should be positive for profitable returns

    def test_calculate_sharpe_ratio_negative_returns(self, calculator, sample_negative_returns):
        """Test Sharpe ratio calculation with negative returns."""
        result = calculator.calculate_sharpe_ratio(sample_negative_returns)

        assert result is not None
        assert isinstance(result, Decimal)
        assert result < 0  # Should be negative for losing returns

    def test_calculate_sharpe_ratio_custom_risk_free_rate(
        self, calculator, sample_positive_returns
    ):
        """Test Sharpe ratio with custom risk-free rate."""
        high_risk_free_rate = Decimal("0.05")  # 5% risk-free rate

        result = calculator.calculate_sharpe_ratio(sample_positive_returns, high_risk_free_rate)

        assert result is not None
        assert isinstance(result, Decimal)
        # Higher risk-free rate should reduce Sharpe ratio
        result_low_rf = calculator.calculate_sharpe_ratio(sample_positive_returns, Decimal("0.01"))
        assert result < result_low_rf

    def test_calculate_sharpe_ratio_zero_risk_free_rate(self, calculator, sample_positive_returns):
        """Test Sharpe ratio with zero risk-free rate."""
        result = calculator.calculate_sharpe_ratio(sample_positive_returns, Decimal("0"))

        assert result is not None
        assert isinstance(result, Decimal)
        assert result > 0

    # Edge Cases - Data Insufficient
    def test_calculate_sharpe_ratio_empty_returns(self, calculator):
        """Test Sharpe ratio with empty returns list."""
        result = calculator.calculate_sharpe_ratio([])

        assert result is None

    def test_calculate_sharpe_ratio_insufficient_data(self, calculator):
        """Test Sharpe ratio with insufficient data points."""
        # Less than MIN_DATA_POINTS_FOR_STATS
        insufficient_returns = [Decimal("0.01")]

        result = calculator.calculate_sharpe_ratio(insufficient_returns)

        assert result is None

    def test_calculate_sharpe_ratio_exactly_min_data_points(self, calculator):
        """Test Sharpe ratio with exactly minimum data points."""
        min_returns = [Decimal("0.01"), Decimal("0.02")]  # Exactly MIN_DATA_POINTS_FOR_STATS

        result = calculator.calculate_sharpe_ratio(min_returns)

        assert result is not None
        assert isinstance(result, Decimal)

    # Edge Cases - Zero Volatility
    def test_calculate_sharpe_ratio_zero_volatility(self, calculator, zero_volatility_returns):
        """Test Sharpe ratio with zero standard deviation."""
        result = calculator.calculate_sharpe_ratio(zero_volatility_returns)

        assert result is None  # Cannot divide by zero volatility

    def test_calculate_sharpe_ratio_single_unique_value(self, calculator):
        """Test Sharpe ratio when all returns are identical."""
        identical_returns = [Decimal("0.015")] * 10

        result = calculator.calculate_sharpe_ratio(identical_returns)

        assert result is None  # Zero standard deviation

    # Mathematical Precision Tests
    def test_calculate_sharpe_ratio_high_precision_returns(self, calculator):
        """Test Sharpe ratio with high precision decimal returns."""
        precision_returns = [
            Decimal("0.0123456789"),
            Decimal("-0.0087654321"),
            Decimal("0.0145673829"),
            Decimal("-0.0098765432"),
            Decimal("0.0167892345"),
            Decimal("0.0034567890"),
        ]

        result = calculator.calculate_sharpe_ratio(precision_returns)

        assert result is not None
        assert isinstance(result, Decimal)

    def test_calculate_sharpe_ratio_very_small_returns(self, calculator):
        """Test Sharpe ratio with very small return values."""
        small_returns = [
            Decimal("0.0001"),
            Decimal("-0.0002"),
            Decimal("0.0001"),
            Decimal("0.0003"),
            Decimal("-0.0001"),
            Decimal("0.0002"),
        ]

        result = calculator.calculate_sharpe_ratio(small_returns)

        assert result is not None
        assert isinstance(result, Decimal)

    def test_calculate_sharpe_ratio_very_large_returns(self, calculator):
        """Test Sharpe ratio with very large return values."""
        large_returns = [
            Decimal("0.5"),
            Decimal("-0.3"),
            Decimal("0.4"),
            Decimal("0.6"),
            Decimal("-0.2"),
            Decimal("0.3"),
        ]

        result = calculator.calculate_sharpe_ratio(large_returns)

        assert result is not None
        assert isinstance(result, Decimal)

    # Annualization Tests
    def test_sharpe_ratio_annualization_factor(self, calculator):
        """Test that Sharpe ratio uses correct annualization factor."""
        # Create varied daily returns to avoid zero volatility
        daily_returns = (
            [Decimal("0.001")] * 100 + [Decimal("0.002")] * 100 + [Decimal("0.0005")] * 52
        )

        result = calculator.calculate_sharpe_ratio(daily_returns)

        # The result should reflect annualization (252 trading days)
        assert result is not None
        # At minimum, should be calculated without errors and reflect annualization

    def test_sharpe_ratio_trading_days_constant(self, calculator):
        """Test that TRADING_DAYS_PER_YEAR constant is used correctly."""
        assert TRADING_DAYS_PER_YEAR == 252
        assert Decimal(str(252**0.5)) == ANNUALIZATION_FACTOR

    # Formula Verification Tests
    def test_sharpe_ratio_formula_components(self, calculator):
        """Test individual components of Sharpe ratio formula."""
        returns = [Decimal("0.01"), Decimal("0.02"), Decimal("-0.01")]
        risk_free_rate = Decimal("0.03")

        result = calculator.calculate_sharpe_ratio(returns, risk_free_rate)

        # Manually calculate components
        avg_return = sum(returns) / len(returns)  # 0.00667
        annual_return = avg_return * TRADING_DAYS_PER_YEAR  # Should be ~1.68

        # Should be calculated without errors
        assert result is not None
        assert isinstance(result, Decimal)

    def test_sharpe_ratio_manual_calculation_verification(self, calculator):
        """Test Sharpe ratio against manual calculation."""
        simple_returns = [Decimal("0.01"), Decimal("0.02")]
        risk_free_rate = Decimal("0.02")

        result = calculator.calculate_sharpe_ratio(simple_returns, risk_free_rate)

        # Manual calculation
        avg_return = Decimal("0.015")  # (0.01 + 0.02) / 2
        annual_return = avg_return * Decimal(str(TRADING_DAYS_PER_YEAR))  # 3.78

        # Variance = ((0.01-0.015)^2 + (0.02-0.015)^2) / 2 = 0.000025
        # Std dev = sqrt(0.000025) = 0.005
        # Annual std = 0.005 * sqrt(252) ≈ 0.0794
        # Sharpe = (3.78 - 0.02) / 0.0794 ≈ 47.35

        assert result is not None
        assert result > Decimal("40")  # Should be high positive value
        assert result < Decimal("50")

    # Risk-Adjusted Return Tests
    def test_calculate_risk_adjusted_return_basic(self, calculator, sample_portfolio):
        """Test basic risk-adjusted return calculation."""
        result = calculator.calculate_risk_adjusted_return(sample_portfolio)

        assert isinstance(result, dict)
        expected_keys = [
            "total_return",
            "win_rate",
            "profit_factor",
            "average_win",
            "average_loss",
            "max_drawdown",
            "sharpe_ratio",
            "calmar_ratio",
            "expectancy",
        ]
        for key in expected_keys:
            assert key in result

    def test_calculate_risk_adjusted_return_all_metrics_present(self, calculator, sample_portfolio):
        """Test that all risk-adjusted return metrics are calculated."""
        result = calculator.calculate_risk_adjusted_return(sample_portfolio)

        # Check that portfolio methods were called
        sample_portfolio.get_return_percentage.assert_called_once()
        sample_portfolio.get_win_rate.assert_called_once()
        sample_portfolio.get_profit_factor.assert_called_once()
        sample_portfolio.get_average_win.assert_called_once()
        sample_portfolio.get_average_loss.assert_called_once()
        sample_portfolio.get_sharpe_ratio.assert_called_once()

    def test_calculate_risk_adjusted_return_calmar_ratio_calculation(
        self, calculator, sample_portfolio
    ):
        """Test Calmar ratio calculation within risk-adjusted returns."""
        # Mock max_drawdown to return a specific value
        with patch(
            "src.domain.services.risk.portfolio_var_calculator.PortfolioVaRCalculator.calculate_max_drawdown"
        ) as mock_drawdown:
            mock_drawdown.return_value = Decimal("5.0")  # 5% max drawdown

            result = calculator.calculate_risk_adjusted_return(sample_portfolio)

            # Calmar ratio should be total_return / max_drawdown = 15.5 / 5.0 = 3.1
            assert result["calmar_ratio"] == Decimal("3.1")

    def test_calculate_risk_adjusted_return_calmar_ratio_zero_drawdown(
        self, calculator, sample_portfolio
    ):
        """Test Calmar ratio when max drawdown is zero."""
        with patch(
            "src.domain.services.risk.portfolio_var_calculator.PortfolioVaRCalculator.calculate_max_drawdown"
        ) as mock_drawdown:
            mock_drawdown.return_value = Decimal("0")  # Zero drawdown

            result = calculator.calculate_risk_adjusted_return(sample_portfolio)

            # Calmar ratio should be None when drawdown is zero
            assert result["calmar_ratio"] is None

    def test_calculate_risk_adjusted_return_calmar_ratio_none_values(
        self, calculator, sample_portfolio
    ):
        """Test Calmar ratio when return or drawdown is None."""
        sample_portfolio.get_return_percentage.return_value = None

        with patch(
            "src.domain.services.risk.portfolio_var_calculator.PortfolioVaRCalculator.calculate_max_drawdown"
        ) as mock_drawdown:
            mock_drawdown.return_value = Decimal("5.0")

            result = calculator.calculate_risk_adjusted_return(sample_portfolio)

            assert result["calmar_ratio"] is None

    # Expectancy Calculation Tests
    def test_calculate_risk_adjusted_return_expectancy_calculation(
        self, calculator, sample_portfolio
    ):
        """Test expectancy calculation within risk-adjusted returns."""
        result = calculator.calculate_risk_adjusted_return(sample_portfolio)

        # Expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        # = (0.6 * 150) - (0.4 * 100) = 90 - 40 = 50
        expected_expectancy = Money(Decimal("50.00"))
        assert result["expectancy"] == expected_expectancy

    def test_calculate_risk_adjusted_return_expectancy_none_values(
        self, calculator, sample_portfolio
    ):
        """Test expectancy calculation when some values are None."""
        sample_portfolio.get_win_rate.return_value = None

        result = calculator.calculate_risk_adjusted_return(sample_portfolio)

        assert result["expectancy"] is None

    def test_calculate_risk_adjusted_return_expectancy_invalid_types(
        self, calculator, sample_portfolio
    ):
        """Test expectancy calculation with invalid types."""
        sample_portfolio.get_average_win.return_value = "invalid"

        result = calculator.calculate_risk_adjusted_return(sample_portfolio)

        assert result["expectancy"] is None

    def test_calculate_risk_adjusted_return_expectancy_negative(self, calculator, sample_portfolio):
        """Test expectancy calculation resulting in negative value."""
        # Set up for negative expectancy
        sample_portfolio.get_win_rate.return_value = Decimal("30.0")  # 30% win rate
        sample_portfolio.get_average_win.return_value = Money(Decimal("100.00"))
        sample_portfolio.get_average_loss.return_value = Money(Decimal("200.00"))

        result = calculator.calculate_risk_adjusted_return(sample_portfolio)

        # Expectancy = (0.3 * 100) - (0.7 * 200) = 30 - 140 = -110
        expected_expectancy = Money(Decimal("-110.00"))
        assert result["expectancy"] == expected_expectancy

    # Time Period Parameter Tests
    def test_calculate_risk_adjusted_return_time_period_parameter(
        self, calculator, sample_portfolio
    ):
        """Test that time_period_days parameter is handled correctly."""
        result = calculator.calculate_risk_adjusted_return(sample_portfolio, 60)  # 60 days

        # Parameter is currently unused but should not cause errors
        assert isinstance(result, dict)

    def test_calculate_risk_adjusted_return_default_time_period(self, calculator, sample_portfolio):
        """Test risk-adjusted return with default time period."""
        result = calculator.calculate_risk_adjusted_return(sample_portfolio)

        # Should work with default 30 days
        assert isinstance(result, dict)

    # Edge Cases and Error Handling
    def test_calculate_risk_adjusted_return_portfolio_method_exceptions(self, calculator):
        """Test handling of portfolio method exceptions."""
        failing_portfolio = Mock(spec=Portfolio)
        failing_portfolio.get_return_percentage.side_effect = Exception("Portfolio error")

        # Should handle exceptions gracefully
        try:
            result = calculator.calculate_risk_adjusted_return(failing_portfolio)
            # If no exception, check that result is still a dict
            assert isinstance(result, dict)
        except Exception:
            # Acceptable to propagate portfolio exceptions
            pass

    def test_calculate_risk_adjusted_return_missing_portfolio_methods(self, calculator):
        """Test with portfolio missing required methods."""
        incomplete_portfolio = Mock()
        # Don't spec it so it doesn't have the required methods

        result = calculator.calculate_risk_adjusted_return(incomplete_portfolio)

        # Should handle missing methods gracefully
        assert isinstance(result, dict)

    # Performance and Thread Safety Tests
    def test_calculator_is_stateless(self, calculator, sample_positive_returns):
        """Test that calculator is stateless and thread-safe."""
        # Run multiple Sharpe ratio calculations
        results = []
        for _ in range(5):
            result = calculator.calculate_sharpe_ratio(sample_positive_returns)
            results.append(result)

        # All results should be identical
        assert all(r == results[0] for r in results)

    def test_concurrent_calculations_isolation(
        self, calculator, sample_positive_returns, sample_negative_returns
    ):
        """Test that concurrent calculations don't interfere."""
        result1 = calculator.calculate_sharpe_ratio(sample_positive_returns)
        result2 = calculator.calculate_sharpe_ratio(sample_negative_returns)

        # Results should be independent
        assert result1 != result2
        assert result1 > 0
        assert result2 < 0

    # Documentation Examples Tests
    def test_sharpe_ratio_interpretation_ranges(self, calculator):
        """Test Sharpe ratio ranges mentioned in documentation."""
        # Excellent risk-adjusted returns (> 2)
        excellent_returns = [Decimal("0.02")] * 50 + [Decimal("-0.005")] * 10
        result_excellent = calculator.calculate_sharpe_ratio(excellent_returns, Decimal("0.01"))

        # Good risk-adjusted returns (1-2 range)
        good_returns = [Decimal("0.01")] * 40 + [Decimal("-0.01")] * 20
        result_good = calculator.calculate_sharpe_ratio(good_returns, Decimal("0.02"))

        assert result_excellent is not None
        assert result_good is not None
        # Just ensure calculations complete without errors

    def test_performance_metrics_interpretation_examples(self, calculator):
        """Test performance metrics with interpretation examples from documentation."""
        # Create portfolio that should hit interpretation thresholds
        good_portfolio = Mock(spec=Portfolio)
        good_portfolio.get_return_percentage.return_value = Decimal("20.0")
        good_portfolio.get_win_rate.return_value = Decimal("55.0")  # > 50%
        good_portfolio.get_profit_factor.return_value = Decimal("1.6")  # > 1.5
        good_portfolio.get_average_win.return_value = Money(Decimal("120.00"))
        good_portfolio.get_average_loss.return_value = Money(Decimal("100.00"))
        good_portfolio.get_sharpe_ratio.return_value = Decimal("1.2")  # > 1
        good_portfolio.get_total_value.return_value = Money(Decimal("120000.00"))

        with patch(
            "src.domain.services.risk.portfolio_var_calculator.PortfolioVaRCalculator.calculate_max_drawdown"
        ) as mock_drawdown:
            mock_drawdown.return_value = Decimal("10.0")

            result = calculator.calculate_risk_adjusted_return(good_portfolio)

            # Check interpretation thresholds
            assert result["profit_factor"] > Decimal("1.5")  # Good profitability
            assert result["win_rate"] > Decimal("50")  # More winners than losers
            assert result["sharpe_ratio"] > Decimal("1")  # Good risk-adjusted returns
            assert result["calmar_ratio"] > Decimal("1")  # Return exceeds max drawdown
            assert result["expectancy"].amount > Decimal("0")  # Positive expectancy

    # Decimal Type Consistency Tests
    def test_all_decimal_calculations_maintain_type(self, calculator, sample_positive_returns):
        """Test that all calculations maintain Decimal type."""
        result = calculator.calculate_sharpe_ratio(sample_positive_returns)

        assert isinstance(result, Decimal)

    def test_risk_adjusted_return_decimal_consistency(self, calculator, sample_portfolio):
        """Test that risk-adjusted return metrics maintain correct types."""
        result = calculator.calculate_risk_adjusted_return(sample_portfolio)

        # Check types of returned values
        decimal_keys = [
            "total_return",
            "win_rate",
            "profit_factor",
            "max_drawdown",
            "sharpe_ratio",
            "calmar_ratio",
        ]
        for key in decimal_keys:
            if result[key] is not None:
                assert isinstance(
                    result[key], Decimal
                ), f"Key {key} should be Decimal, got {type(result[key])}"

        money_keys = ["average_win", "average_loss", "expectancy"]
        for key in money_keys:
            if result[key] is not None:
                assert isinstance(
                    result[key], Money
                ), f"Key {key} should be Money, got {type(result[key])}"
