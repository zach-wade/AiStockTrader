"""
Comprehensive Tests for Portfolio VaR Calculator Service
======================================================

Tests for the PortfolioVaRCalculator domain service that handles
portfolio-level Value at Risk and drawdown calculations.
"""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from src.domain.services.risk.portfolio_var_calculator import PortfolioVaRCalculator
from src.domain.value_objects import Money


class TestCalculatePortfolioVar:
    """Test Value at Risk calculation functionality."""

    def test_calculate_portfolio_var_default(self):
        """Test VaR calculation with default parameters."""
        portfolio = Mock()
        portfolio.get_total_value.return_value = Money(Decimal("100000"))

        calculator = PortfolioVaRCalculator()
        var = calculator.calculate_portfolio_var(portfolio)

        # Default: 95% confidence, 1 day horizon
        # VaR = 100000 * 0.02 * 1.65 * sqrt(1) = 3300
        expected = Money(Decimal("3300"))
        assert var == expected

    def test_calculate_portfolio_var_90_confidence(self):
        """Test VaR calculation with 90% confidence level."""
        portfolio = Mock()
        portfolio.get_total_value.return_value = Money(Decimal("500000"))

        calculator = PortfolioVaRCalculator()
        var = calculator.calculate_portfolio_var(portfolio, confidence_level=Decimal("0.90"))

        # 90%: z-score = 1.28
        # VaR = 500000 * 0.02 * 1.28 * sqrt(1) = 12800
        expected = Money(Decimal("12800"))
        assert var == expected

    def test_calculate_portfolio_var_99_confidence(self):
        """Test VaR calculation with 99% confidence level."""
        portfolio = Mock()
        portfolio.get_total_value.return_value = Money(Decimal("200000"))

        calculator = PortfolioVaRCalculator()
        var = calculator.calculate_portfolio_var(portfolio, confidence_level=Decimal("0.99"))

        # 99%: z-score = 2.33
        # VaR = 200000 * 0.02 * 2.33 * sqrt(1) = 9320
        expected = Money(Decimal("9320"))
        assert var == expected

    def test_calculate_portfolio_var_multi_day_horizon(self):
        """Test VaR calculation with multi-day time horizon."""
        portfolio = Mock()
        portfolio.get_total_value.return_value = Money(Decimal("100000"))

        calculator = PortfolioVaRCalculator()
        var = calculator.calculate_portfolio_var(portfolio, time_horizon=5)

        # 5-day horizon: sqrt(5) = 2.236...
        # VaR = 100000 * 0.02 * 1.65 * sqrt(5) ≈ 7378
        expected_amount = (
            Decimal("100000") * Decimal("0.02") * Decimal("1.65") * Decimal("2.236067977")
        )
        assert abs(var.amount - expected_amount) < Decimal("1")

    def test_calculate_portfolio_var_invalid_confidence_low(self):
        """Test VaR calculation with invalid low confidence level."""
        portfolio = Mock()
        calculator = PortfolioVaRCalculator()

        with pytest.raises(ValueError, match="Confidence level must be between 0 and 1"):
            calculator.calculate_portfolio_var(portfolio, confidence_level=Decimal("0"))

    def test_calculate_portfolio_var_invalid_confidence_high(self):
        """Test VaR calculation with invalid high confidence level."""
        portfolio = Mock()
        calculator = PortfolioVaRCalculator()

        with pytest.raises(ValueError, match="Confidence level must be between 0 and 1"):
            calculator.calculate_portfolio_var(portfolio, confidence_level=Decimal("1.5"))

    def test_calculate_portfolio_var_custom_confidence(self):
        """Test VaR calculation with custom confidence level."""
        portfolio = Mock()
        portfolio.get_total_value.return_value = Money(Decimal("100000"))

        calculator = PortfolioVaRCalculator()
        var = calculator.calculate_portfolio_var(
            portfolio,
            confidence_level=Decimal("0.85"),  # Custom level, falls back to default
        )

        # Should use default z-score of 1.65
        expected = Money(Decimal("3300"))
        assert var == expected


class TestCalculateMaxDrawdown:
    """Test maximum drawdown calculation functionality."""

    def test_calculate_max_drawdown_simple_decline(self):
        """Test max drawdown with simple decline."""
        calculator = PortfolioVaRCalculator()

        history = [
            Money(Decimal("100000")),
            Money(Decimal("90000")),
            Money(Decimal("80000")),
            Money(Decimal("85000")),
        ]

        max_dd = calculator.calculate_max_drawdown(history)

        # Max drawdown: (100000 - 80000) / 100000 * 100 = 20%
        assert max_dd == Decimal("20")

    def test_calculate_max_drawdown_multiple_peaks(self):
        """Test max drawdown with multiple peaks and troughs."""
        calculator = PortfolioVaRCalculator()

        history = [
            Money(Decimal("100000")),  # Peak 1
            Money(Decimal("90000")),
            Money(Decimal("110000")),  # New peak
            Money(Decimal("80000")),  # Larger drawdown from new peak
            Money(Decimal("85000")),
        ]

        max_dd = calculator.calculate_max_drawdown(history)

        # Max drawdown: (110000 - 80000) / 110000 * 100 ≈ 27.27%
        expected = (Decimal("30000") / Decimal("110000")) * Decimal("100")
        assert abs(max_dd - expected) < Decimal("0.01")

    def test_calculate_max_drawdown_no_decline(self):
        """Test max drawdown with portfolio only going up."""
        calculator = PortfolioVaRCalculator()

        history = [
            Money(Decimal("100000")),
            Money(Decimal("110000")),
            Money(Decimal("120000")),
            Money(Decimal("130000")),
        ]

        max_dd = calculator.calculate_max_drawdown(history)

        # No drawdown when only going up
        assert max_dd == Decimal("0")

    def test_calculate_max_drawdown_empty_list(self):
        """Test max drawdown with empty history."""
        calculator = PortfolioVaRCalculator()

        max_dd = calculator.calculate_max_drawdown([])
        assert max_dd == Decimal("0")

    def test_calculate_max_drawdown_insufficient_data(self):
        """Test max drawdown with insufficient data points."""
        calculator = PortfolioVaRCalculator()

        # Only 1 data point (MIN_DATA_POINTS_FOR_STATS = 2)
        history = [Money(Decimal("100000"))]

        max_dd = calculator.calculate_max_drawdown(history)
        assert max_dd == Decimal("0")

    def test_calculate_max_drawdown_recovery_after_drawdown(self):
        """Test max drawdown with recovery after maximum drawdown."""
        calculator = PortfolioVaRCalculator()

        history = [
            Money(Decimal("100000")),  # Start
            Money(Decimal("70000")),  # -30% drawdown
            Money(Decimal("120000")),  # Recovery above original peak
            Money(Decimal("90000")),  # New decline from new peak
        ]

        max_dd = calculator.calculate_max_drawdown(history)

        # Max drawdown should still be 30% from the first decline
        # Not the 25% from the second decline (90000 from 120000 peak)
        assert max_dd == Decimal("30")

    def test_calculate_max_drawdown_zero_value(self):
        """Test max drawdown with zero portfolio value."""
        calculator = PortfolioVaRCalculator()

        history = [
            Money(Decimal("100000")),
            Money(Decimal("0")),  # Complete loss
            Money(Decimal("10000")),  # Partial recovery
        ]

        max_dd = calculator.calculate_max_drawdown(history)

        # 100% drawdown
        assert max_dd == Decimal("100")

    def test_calculate_max_drawdown_precision(self):
        """Test max drawdown calculation maintains precision."""
        calculator = PortfolioVaRCalculator()

        history = [
            Money(Decimal("123456.789")),
            Money(Decimal("111111.111")),
        ]

        max_dd = calculator.calculate_max_drawdown(history)

        # Precise calculation
        expected = (Decimal("12345.678") / Decimal("123456.789")) * Decimal("100")
        assert abs(max_dd - expected) < Decimal("0.001")


class TestPortfolioVaRCalculatorEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_var_calculation_large_portfolio(self):
        """Test VaR calculation with extremely large portfolio."""
        portfolio = Mock()
        portfolio.get_total_value.return_value = Money(Decimal("999999999999"))

        calculator = PortfolioVaRCalculator()
        var = calculator.calculate_portfolio_var(portfolio)

        # Should handle large values without error
        expected = Decimal("999999999999") * Decimal("0.02") * Decimal("1.65")
        assert var.amount == expected

    def test_var_calculation_small_portfolio(self):
        """Test VaR calculation with very small portfolio."""
        portfolio = Mock()
        portfolio.get_total_value.return_value = Money(Decimal("100.50"))

        calculator = PortfolioVaRCalculator()
        var = calculator.calculate_portfolio_var(portfolio)

        # Should handle small values with precision
        expected = Decimal("100.50") * Decimal("0.02") * Decimal("1.65")
        assert var.amount == expected

    def test_var_calculation_zero_portfolio(self):
        """Test VaR calculation with zero portfolio value."""
        portfolio = Mock()
        portfolio.get_total_value.return_value = Money(Decimal("0"))

        calculator = PortfolioVaRCalculator()
        var = calculator.calculate_portfolio_var(portfolio)

        assert var.amount == Decimal("0")

    def test_max_drawdown_volatile_history(self):
        """Test max drawdown with highly volatile portfolio history."""
        calculator = PortfolioVaRCalculator()

        # Simulate volatile market with multiple ups and downs
        history = []
        base_value = Decimal("100000")

        # Create 20 data points with volatility
        for i in range(20):
            # Simulate some volatility pattern
            if i < 5:
                value = base_value * (Decimal("1") - Decimal("0.02") * i)  # Decline
            elif i < 10:
                value = (
                    base_value * Decimal("0.9") * (Decimal("1") + Decimal("0.01") * (i - 5))
                )  # Recovery
            elif i < 15:
                value = base_value * (Decimal("1") - Decimal("0.04") * (i - 10))  # Bigger decline
            else:
                value = (
                    base_value * Decimal("0.8") * (Decimal("1") + Decimal("0.02") * (i - 15))
                )  # Recovery

            history.append(Money(value))

        max_dd = calculator.calculate_max_drawdown(history)

        # Should find the maximum drawdown across the volatile period
        assert max_dd > Decimal("15")  # Should capture significant decline
        assert max_dd <= Decimal("100")  # Should not exceed 100%

    def test_calculator_thread_safety(self):
        """Test that calculator is stateless and thread-safe."""
        calculator = PortfolioVaRCalculator()

        # Create multiple portfolios
        portfolio1 = Mock()
        portfolio1.get_total_value.return_value = Money(Decimal("100000"))

        portfolio2 = Mock()
        portfolio2.get_total_value.return_value = Money(Decimal("200000"))

        # Calculate VaRs simultaneously (simulating concurrent access)
        var1 = calculator.calculate_portfolio_var(portfolio1)
        var2 = calculator.calculate_portfolio_var(portfolio2)

        # Verify independent calculations
        assert var1.amount == Decimal("3300")
        assert var2.amount == Decimal("6600")

        # Test drawdown calculations don't interfere
        history1 = [Money(Decimal("100000")), Money(Decimal("80000"))]
        history2 = [Money(Decimal("200000")), Money(Decimal("150000"))]

        # Need more data points for drawdown calculation
        history1.extend([Money(Decimal("85000"))] * 10)
        history2.extend([Money(Decimal("180000"))] * 10)

        dd1 = calculator.calculate_max_drawdown(history1)
        dd2 = calculator.calculate_max_drawdown(history2)

        # Independent calculations
        assert dd1 == Decimal("20")  # 20% drawdown
        assert dd2 == Decimal("25")  # 25% drawdown
