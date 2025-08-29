"""
Comprehensive Tests for Portfolio Validator Service
=================================================

Tests for the PortfolioValidator domain service that handles
all portfolio validation and risk checking logic.
"""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from src.domain.entities.portfolio import Portfolio
from src.domain.services.portfolio_validator import PortfolioValidator
from src.domain.value_objects import Money, Price, Quantity


class TestCanOpenPosition:
    """Test position opening validation."""

    def test_can_open_position_success(self):
        """Test successful validation for opening position."""
        portfolio = Portfolio(
            cash_balance=Money(Decimal("100000")),
            max_position_size=Money(Decimal("20000")),
            max_positions=10,
            max_portfolio_risk=Decimal("0.1"),  # 10%
        )

        can_open, reason = PortfolioValidator.can_open_position(
            portfolio, "AAPL", Quantity(Decimal("100")), Price(Decimal("150"))
        )

        assert can_open is True
        assert reason is None

    def test_can_open_position_already_exists(self):
        """Test validation when position already exists."""
        portfolio = Portfolio(cash_balance=Money(Decimal("100000")))

        # Add existing open position
        existing_position = Mock()
        existing_position.is_closed.return_value = False
        portfolio.positions["AAPL"] = existing_position

        can_open, reason = PortfolioValidator.can_open_position(
            portfolio, "AAPL", Quantity(Decimal("100")), Price(Decimal("150"))
        )

        assert can_open is False
        assert "Position already exists for AAPL" in reason

    def test_can_open_position_closed_exists_ok(self):
        """Test validation allows reopening closed position."""
        portfolio = Portfolio(cash_balance=Money(Decimal("100000")))

        # Add existing closed position
        closed_position = Mock()
        closed_position.is_closed.return_value = True
        portfolio.positions["AAPL"] = closed_position

        can_open, reason = PortfolioValidator.can_open_position(
            portfolio, "AAPL", Quantity(Decimal("100")), Price(Decimal("150"))
        )

        assert can_open is True
        assert reason is None

    def test_can_open_position_max_positions_reached(self):
        """Test validation when max positions limit reached."""
        portfolio = Portfolio(cash_balance=Money(Decimal("100000")), max_positions=2)

        # Add two open positions
        for symbol in ["AAPL", "MSFT"]:
            position = Mock()
            position.is_closed.return_value = False
            portfolio.positions[symbol] = position

        can_open, reason = PortfolioValidator.can_open_position(
            portfolio, "GOOGL", Quantity(Decimal("10")), Price(Decimal("2500"))
        )

        assert can_open is False
        assert "Maximum positions limit reached (2)" in reason

    def test_can_open_position_size_limit_exceeded(self):
        """Test validation when position size limit exceeded."""
        portfolio = Portfolio(
            cash_balance=Money(Decimal("100000")), max_position_size=Money(Decimal("10000"))
        )

        can_open, reason = PortfolioValidator.can_open_position(
            portfolio,
            "AAPL",
            Quantity(Decimal("100")),
            Price(Decimal("150")),  # 100 * 150 = 15000 > 10000 limit
        )

        assert can_open is False
        assert "Position size" in reason
        assert "exceeds limit" in reason

    def test_can_open_position_insufficient_cash(self):
        """Test validation with insufficient cash."""
        portfolio = Portfolio(cash_balance=Money(Decimal("10000")))

        can_open, reason = PortfolioValidator.can_open_position(
            portfolio,
            "AAPL",
            Quantity(Decimal("100")),
            Price(Decimal("150")),  # Requires 15000 > 10000 available
        )

        assert can_open is False
        assert "Insufficient cash" in reason
        assert "10000" in reason
        assert "15000" in reason

    def test_can_open_position_portfolio_risk_exceeded(self):
        """Test validation when portfolio risk limit exceeded."""
        portfolio = Portfolio(
            cash_balance=Money(Decimal("100000")),
            max_portfolio_risk=Decimal("0.05"),  # 5% max risk
        )

        can_open, reason = PortfolioValidator.can_open_position(
            portfolio,
            "AAPL",
            Quantity(Decimal("100")),
            Price(Decimal("150")),  # 15000 / 100000 = 15% > 5% limit
        )

        assert can_open is False
        assert "Position risk" in reason
        assert "exceeds portfolio limit" in reason

    def test_can_open_position_with_value_objects(self):
        """Test validation works with proper value objects."""
        portfolio = Portfolio(
            cash_balance=Money(Decimal("50000")), max_position_size=Money(Decimal("20000"))
        )

        can_open, reason = PortfolioValidator.can_open_position(
            portfolio,
            "MSFT",
            Quantity(Decimal("50")),
            Price(Decimal("300")),  # 15000 position value
        )

        assert can_open is True
        assert reason is None

    def test_can_open_position_with_raw_decimals(self):
        """Test validation handles raw Decimal values."""
        portfolio = Mock()
        portfolio.positions = {}
        portfolio.get_open_positions.return_value = []
        portfolio.max_positions = 10
        portfolio.max_position_size = Decimal("20000")
        portfolio.cash_balance = Decimal("50000")
        portfolio.max_portfolio_risk = Decimal("0.1")

        # Mock the portfolio metrics calculator
        with patch(
            "src.domain.services.portfolio_validator.PortfolioMetricsCalculator"
        ) as mock_calc:
            mock_calc.get_total_value.return_value = Decimal("50000")

            can_open, reason = PortfolioValidator.can_open_position(
                portfolio,
                "TEST",
                Decimal("100"),
                Decimal("150"),  # Raw Decimal  # Raw Decimal
            )

        assert can_open is True
        assert reason is None

    def test_can_open_position_zero_portfolio_value(self):
        """Test validation when portfolio value is zero."""
        portfolio = Portfolio(cash_balance=Money(Decimal("0")), initial_capital=Money(Decimal("0")))

        # Should pass risk check when portfolio value is 0
        can_open, reason = PortfolioValidator.can_open_position(
            portfolio, "AAPL", Quantity(Decimal("1")), Price(Decimal("1"))
        )

        # Should fail on insufficient cash, not risk
        assert can_open is False
        assert "Insufficient cash" in reason

    def test_can_open_position_edge_case_limits(self):
        """Test validation at exact limit boundaries."""
        portfolio = Portfolio(
            cash_balance=Money(Decimal("15000")),
            max_position_size=Money(Decimal("15000")),
            max_portfolio_risk=Decimal("1.0"),  # 100% risk allowed
        )

        # Should pass when exactly at limits
        can_open, reason = PortfolioValidator.can_open_position(
            portfolio,
            "AAPL",
            Quantity(Decimal("100")),
            Price(Decimal("150")),  # Exactly 15000
        )

        assert can_open is True
        assert reason is None


class TestValidatePortfolioState:
    """Test portfolio state validation."""

    def test_validate_portfolio_state_success(self):
        """Test successful portfolio state validation."""
        portfolio = Portfolio(
            initial_capital=Money(Decimal("100000")),
            cash_balance=Money(Decimal("50000")),
            max_position_size=Money(Decimal("10000")),
            max_portfolio_risk=Decimal("0.05"),
            max_positions=10,
            max_leverage=Decimal("2.0"),
        )

        # Should not raise any exception
        PortfolioValidator.validate_portfolio_state(portfolio)

    def test_validate_negative_initial_capital(self):
        """Test validation fails with negative initial capital."""
        portfolio = Portfolio(initial_capital=Money(Decimal("-1000")))

        with pytest.raises(ValueError, match="Initial capital must be positive"):
            PortfolioValidator.validate_portfolio_state(portfolio)

    def test_validate_zero_initial_capital(self):
        """Test validation fails with zero initial capital."""
        portfolio = Portfolio(initial_capital=Money(Decimal("0")))

        with pytest.raises(ValueError, match="Initial capital must be positive"):
            PortfolioValidator.validate_portfolio_state(portfolio)

    def test_validate_negative_cash_balance(self):
        """Test validation fails with negative cash balance."""
        portfolio = Portfolio(cash_balance=Money(Decimal("-5000")))

        with pytest.raises(ValueError, match="Cash balance cannot be negative"):
            PortfolioValidator.validate_portfolio_state(portfolio)

    def test_validate_zero_cash_balance_ok(self):
        """Test validation passes with zero cash balance."""
        portfolio = Portfolio(cash_balance=Money(Decimal("0")))

        # Should not raise exception
        PortfolioValidator.validate_portfolio_state(portfolio)

    def test_validate_negative_max_position_size(self):
        """Test validation fails with negative max position size."""
        portfolio = Portfolio(max_position_size=Money(Decimal("-1000")))

        with pytest.raises(ValueError, match="Max position size must be positive"):
            PortfolioValidator.validate_portfolio_state(portfolio)

    def test_validate_zero_max_position_size(self):
        """Test validation fails with zero max position size."""
        portfolio = Portfolio(max_position_size=Money(Decimal("0")))

        with pytest.raises(ValueError, match="Max position size must be positive"):
            PortfolioValidator.validate_portfolio_state(portfolio)

    def test_validate_none_max_position_size_ok(self):
        """Test validation passes with None max position size."""
        portfolio = Portfolio(max_position_size=None)

        # Should not raise exception
        PortfolioValidator.validate_portfolio_state(portfolio)

    def test_validate_negative_max_portfolio_risk(self):
        """Test validation fails with negative max portfolio risk."""
        portfolio = Portfolio(max_portfolio_risk=Decimal("-0.01"))

        with pytest.raises(ValueError, match="Max portfolio risk must be between 0 and 1"):
            PortfolioValidator.validate_portfolio_state(portfolio)

    def test_validate_zero_max_portfolio_risk(self):
        """Test validation fails with zero max portfolio risk."""
        portfolio = Portfolio(max_portfolio_risk=Decimal("0"))

        with pytest.raises(ValueError, match="Max portfolio risk must be between 0 and 1"):
            PortfolioValidator.validate_portfolio_state(portfolio)

    def test_validate_max_portfolio_risk_over_one(self):
        """Test validation fails with max portfolio risk over 100%."""
        portfolio = Portfolio(max_portfolio_risk=Decimal("1.5"))

        with pytest.raises(ValueError, match="Max portfolio risk must be between 0 and 1"):
            PortfolioValidator.validate_portfolio_state(portfolio)

    def test_validate_max_portfolio_risk_exactly_one_ok(self):
        """Test validation passes with exactly 100% max risk."""
        portfolio = Portfolio(max_portfolio_risk=Decimal("1.0"))

        # Should not raise exception
        PortfolioValidator.validate_portfolio_state(portfolio)

    def test_validate_none_max_portfolio_risk_ok(self):
        """Test validation passes with None max portfolio risk."""
        portfolio = Portfolio(max_portfolio_risk=None)

        # Should not raise exception
        PortfolioValidator.validate_portfolio_state(portfolio)

    def test_validate_negative_max_positions(self):
        """Test validation fails with negative max positions."""
        portfolio = Portfolio(max_positions=-1)

        with pytest.raises(ValueError, match="Max positions must be positive"):
            PortfolioValidator.validate_portfolio_state(portfolio)

    def test_validate_zero_max_positions(self):
        """Test validation fails with zero max positions."""
        portfolio = Portfolio(max_positions=0)

        with pytest.raises(ValueError, match="Max positions must be positive"):
            PortfolioValidator.validate_portfolio_state(portfolio)

    def test_validate_max_leverage_less_than_one(self):
        """Test validation fails with max leverage less than 1."""
        portfolio = Portfolio(max_leverage=Decimal("0.5"))

        with pytest.raises(ValueError, match="Max leverage must be at least 1.0"):
            PortfolioValidator.validate_portfolio_state(portfolio)

    def test_validate_max_leverage_exactly_one_ok(self):
        """Test validation passes with exactly 1.0 leverage."""
        portfolio = Portfolio(max_leverage=Decimal("1.0"))

        # Should not raise exception
        PortfolioValidator.validate_portfolio_state(portfolio)

    def test_validate_with_raw_decimal_values(self):
        """Test validation works with raw Decimal values."""
        portfolio = Mock()
        portfolio.initial_capital = Decimal("100000")
        portfolio.cash_balance = Decimal("50000")
        portfolio.max_position_size = Decimal("10000")
        portfolio.max_portfolio_risk = Decimal("0.1")
        portfolio.max_positions = 5
        portfolio.max_leverage = Decimal("2.0")

        # Should not raise exception
        PortfolioValidator.validate_portfolio_state(portfolio)

    def test_validate_with_money_objects(self):
        """Test validation works with Money objects."""
        portfolio = Portfolio(
            initial_capital=Money(Decimal("100000")),
            cash_balance=Money(Decimal("75000")),
            max_position_size=Money(Decimal("15000")),
        )

        # Should not raise exception
        PortfolioValidator.validate_portfolio_state(portfolio)


class TestValidationEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_validation_extremely_large_values(self):
        """Test validation with extremely large values."""
        large_value = Decimal("999999999999.99")

        portfolio = Portfolio(
            initial_capital=Money(large_value),
            cash_balance=Money(large_value),
            max_position_size=Money(large_value),
        )

        # Should handle large values without error
        PortfolioValidator.validate_portfolio_state(portfolio)

        can_open, reason = PortfolioValidator.can_open_position(
            portfolio, "AAPL", Quantity(Decimal("1")), Price(Decimal("100"))
        )

        assert can_open is True

    def test_validation_high_precision_values(self):
        """Test validation with high precision decimal values."""
        precise_value = Decimal("12345.6789123456789")

        portfolio = Portfolio(
            initial_capital=Money(precise_value),
            cash_balance=Money(precise_value),
            max_position_size=Money(precise_value),
        )

        PortfolioValidator.validate_portfolio_state(portfolio)

        can_open, reason = PortfolioValidator.can_open_position(
            portfolio, "AAPL", Quantity(Decimal("1.23456789")), Price(Decimal("9876.54321"))
        )

        assert can_open is True

    def test_validation_mixed_attribute_types(self):
        """Test validation handles mixed Money/Decimal attributes."""
        portfolio = Mock()
        portfolio.initial_capital = Money(Decimal("100000"))  # Money object
        portfolio.cash_balance = Decimal("50000")  # Raw Decimal
        portfolio.max_position_size = Money(Decimal("20000"))  # Money object
        portfolio.max_portfolio_risk = Decimal("0.1")
        portfolio.max_positions = 10
        portfolio.max_leverage = Decimal("1.0")

        # Should handle mixed types correctly
        PortfolioValidator.validate_portfolio_state(portfolio)

    def test_can_open_position_complex_scenario(self):
        """Test complex validation scenario with multiple constraints."""
        portfolio = Portfolio(
            cash_balance=Money(Decimal("50000")),
            max_position_size=Money(Decimal("15000")),
            max_positions=3,
            max_portfolio_risk=Decimal("0.2"),  # 20%
        )

        # Add two existing positions
        for i, symbol in enumerate(["AAPL", "MSFT"]):
            position = Mock()
            position.is_closed.return_value = False
            portfolio.positions[symbol] = position

        # Try to open position that passes individual checks but may fail combined
        can_open, reason = PortfolioValidator.can_open_position(
            portfolio,
            "GOOGL",
            Quantity(Decimal("5")),
            Price(Decimal("2500")),  # 12500 position value
        )

        # Should pass all constraints
        assert can_open is True
        assert reason is None

    def test_risk_calculation_accuracy(self):
        """Test risk calculation accuracy with realistic portfolio."""
        portfolio = Portfolio(
            cash_balance=Money(Decimal("80000")),
            max_portfolio_risk=Decimal("0.15"),  # 15%
        )

        # Add existing position to increase portfolio value
        existing_position = Mock()
        existing_position.is_closed.return_value = False
        existing_position.get_position_value.return_value = Money(Decimal("20000"))
        portfolio.positions["AAPL"] = existing_position

        # Portfolio value now: 80000 (cash) + 20000 (position) = 100000
        # New position value: 12000
        # Risk ratio: 12000 / 100000 = 12% < 15% limit
        can_open, reason = PortfolioValidator.can_open_position(
            portfolio,
            "MSFT",
            Quantity(Decimal("40")),
            Price(Decimal("300")),  # 12000 position value
        )

        assert can_open is True
        assert reason is None
