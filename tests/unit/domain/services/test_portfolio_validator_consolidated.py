"""Comprehensive tests for PortfolioValidator consolidated service."""

from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from src.domain.entities.portfolio import Portfolio, PositionRequest
from src.domain.entities.position import Position
from src.domain.services.portfolio_validator_consolidated import PortfolioValidator
from src.domain.value_objects import Money, Price, Quantity


class TestPortfolioValidator:
    """Test suite for PortfolioValidator service."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = PortfolioValidator()

        # Create mock portfolio with standard attributes
        self.portfolio = Mock(spec=Portfolio)
        self.portfolio.cash_balance = Money(Decimal("10000"))
        self.portfolio.initial_capital = Money(Decimal("100000"))
        self.portfolio.total_realized_pnl = Money(Decimal("0"))
        self.portfolio.max_positions = 10
        self.portfolio.max_position_size = Money(Decimal("5000"))
        self.portfolio.max_portfolio_risk = Decimal("0.5")
        self.portfolio.max_leverage = Decimal("2.0")
        self.portfolio.trades_count = 0
        self.portfolio.total_commission_paid = Money(Decimal("0"))
        self.portfolio.winning_trades = 0
        self.portfolio.losing_trades = 0

        # Mock methods
        self.portfolio.has_position = Mock(return_value=False)
        self.portfolio.is_position_limit_reached = Mock(return_value=False)
        self.portfolio.get_open_positions = Mock(return_value=[])
        self.portfolio.get_position = Mock(return_value=None)
        self.portfolio.get_position_count = Mock(return_value=0)

    # --- Position Opening Validation Tests ---

    def test_can_open_position_success(self):
        """Test successful position opening validation."""
        can_open, reason = self.validator.can_open_position(
            self.portfolio, "AAPL", Quantity(Decimal("10")), Price(Decimal("150"))
        )

        assert can_open is True
        assert reason is None

    def test_can_open_position_already_exists(self):
        """Test validation when position already exists."""
        self.portfolio.has_position.return_value = True

        can_open, reason = self.validator.can_open_position(
            self.portfolio, "AAPL", Quantity(Decimal("10")), Price(Decimal("150"))
        )

        assert can_open is False
        assert "Position already exists for AAPL" in reason

    def test_can_open_position_limit_reached(self):
        """Test validation when position limit is reached."""
        self.portfolio.is_position_limit_reached.return_value = True

        can_open, reason = self.validator.can_open_position(
            self.portfolio, "AAPL", Quantity(Decimal("10")), Price(Decimal("150"))
        )

        assert can_open is False
        assert "Maximum position limit (10) reached" in reason

    def test_can_open_position_insufficient_cash(self):
        """Test validation with insufficient cash."""
        self.portfolio.cash_balance = Money(Decimal("1000"))

        can_open, reason = self.validator.can_open_position(
            self.portfolio,
            "AAPL",
            Quantity(Decimal("20")),
            Price(Decimal("100")),  # Total: 2000 + commission, under max_position_size but > cash
        )

        assert can_open is False
        assert "Insufficient cash" in reason

    def test_can_open_position_exceeds_max_size(self):
        """Test validation when position size exceeds maximum."""
        can_open, reason = self.validator.can_open_position(
            self.portfolio,
            "AAPL",
            Quantity(Decimal("100")),
            Price(Decimal("100")),  # Total: 10000 > max 5000
        )

        assert can_open is False
        assert "Position size" in reason and "exceeds maximum" in reason

    def test_can_open_position_exceeds_risk_limit(self):
        """Test validation when position risk exceeds portfolio risk limit."""
        # Set low risk limit
        self.portfolio.max_portfolio_risk = Decimal("0.1")

        # Try to open position that would be 15% of portfolio
        can_open, reason = self.validator.can_open_position(
            self.portfolio,
            "AAPL",
            Quantity(Decimal("10")),
            Price(Decimal("150")),  # 1500 / 10000 = 15%
        )

        assert can_open is False
        assert "Position risk" in reason and "exceeds maximum" in reason

    def test_can_open_position_exceeds_leverage(self):
        """Test validation when leverage limit is exceeded."""
        # Create existing positions
        position1 = Mock(spec=Position)
        position1.get_position_value.return_value = Money(Decimal("8000"))

        self.portfolio.get_open_positions.return_value = [position1]

        # Try to open position that would exceed leverage
        # Current: 8000 in positions, 10000 cash = 18000 total
        # New position: 3000, would make 11000 in positions
        # Leverage: 11000 / 18000 = 0.61, but we'll set max to 0.5
        self.portfolio.max_leverage = Decimal("0.5")

        can_open, reason = self.validator.can_open_position(
            self.portfolio,
            "AAPL",
            Quantity(Decimal("20")),
            Price(Decimal("150")),  # 3000
        )

        assert can_open is False
        assert "Leverage" in reason and "exceeds maximum" in reason

    def test_can_open_position_with_existing_positions(self):
        """Test validation with existing positions in portfolio."""
        # Create existing positions
        position1 = Mock(spec=Position)
        position1.get_position_value.return_value = Money(Decimal("3000"))
        position1.symbol = "MSFT"

        position2 = Mock(spec=Position)
        position2.get_position_value.return_value = Money(Decimal("2000"))
        position2.symbol = "GOOGL"

        self.portfolio.get_open_positions.return_value = [position1, position2]

        # Should still be able to open new position
        can_open, reason = self.validator.can_open_position(
            self.portfolio,
            "AAPL",
            Quantity(Decimal("10")),
            Price(Decimal("100")),  # 1000
        )

        assert can_open is True
        assert reason is None

    # --- Position Request Validation Tests ---

    def test_validate_position_request_success(self):
        """Test successful position request validation."""
        request = Mock(spec=PositionRequest)
        request.symbol = "AAPL"
        request.quantity = Quantity(Decimal("10"))
        request.entry_price = Price(Decimal("150"))
        request.commission = Money(Decimal("5"))

        # Should not raise any exception
        self.validator.validate_position_request(self.portfolio, request)

    def test_validate_position_request_no_symbol(self):
        """Test validation with missing symbol."""
        request = Mock(spec=PositionRequest)
        request.symbol = ""
        request.quantity = Quantity(Decimal("10"))
        request.entry_price = Price(Decimal("150"))
        request.commission = Money(Decimal("5"))

        with pytest.raises(ValueError, match="Symbol is required"):
            self.validator.validate_position_request(self.portfolio, request)

    def test_validate_position_request_zero_quantity(self):
        """Test validation with zero quantity."""
        request = Mock(spec=PositionRequest)
        request.symbol = "AAPL"
        request.quantity = Quantity(Decimal("0"))
        request.entry_price = Price(Decimal("150"))
        request.commission = Money(Decimal("5"))

        with pytest.raises(ValueError, match="Quantity must be positive"):
            self.validator.validate_position_request(self.portfolio, request)

    def test_validate_position_request_negative_quantity(self):
        """Test validation with negative quantity."""
        request = Mock(spec=PositionRequest)
        request.symbol = "AAPL"
        request.quantity = Mock(value=Decimal("-10"))
        request.entry_price = Price(Decimal("150"))
        request.commission = Money(Decimal("5"))

        with pytest.raises(ValueError, match="Quantity must be positive"):
            self.validator.validate_position_request(self.portfolio, request)

    def test_validate_position_request_zero_price(self):
        """Test validation with zero entry price."""
        request = Mock(spec=PositionRequest)
        request.symbol = "AAPL"
        request.quantity = Quantity(Decimal("10"))
        # Mock a Price object with value 0 to bypass Price's own validation
        mock_price = Mock(spec=Price)
        mock_price.value = Decimal("0")
        request.entry_price = mock_price
        request.commission = Money(Decimal("5"))

        with pytest.raises(ValueError, match="Entry price must be positive"):
            self.validator.validate_position_request(self.portfolio, request)

    def test_validate_position_request_negative_commission(self):
        """Test validation with negative commission."""
        request = Mock(spec=PositionRequest)
        request.symbol = "AAPL"
        request.quantity = Quantity(Decimal("10"))
        request.entry_price = Price(Decimal("150"))
        request.commission = Mock(amount=Decimal("-5"))

        with pytest.raises(ValueError, match="Commission cannot be negative"):
            self.validator.validate_position_request(self.portfolio, request)

    def test_validate_position_request_cannot_open(self):
        """Test validation when position cannot be opened."""
        self.portfolio.has_position.return_value = True

        request = Mock(spec=PositionRequest)
        request.symbol = "AAPL"
        request.quantity = Quantity(Decimal("10"))
        request.entry_price = Price(Decimal("150"))
        request.commission = Money(Decimal("5"))

        with pytest.raises(ValueError, match="Cannot open position"):
            self.validator.validate_position_request(self.portfolio, request)

    # --- Position Closing Validation Tests ---

    def test_can_close_position_success(self):
        """Test successful position closing validation."""
        position = Mock(spec=Position)
        position.is_closed.return_value = False
        position.quantity = Quantity(Decimal("10"))

        self.portfolio.get_position.return_value = position

        can_close, reason = self.validator.can_close_position(self.portfolio, "AAPL")

        assert can_close is True
        assert reason is None

    def test_can_close_position_not_found(self):
        """Test validation when position doesn't exist."""
        self.portfolio.get_position.return_value = None

        can_close, reason = self.validator.can_close_position(self.portfolio, "AAPL")

        assert can_close is False
        assert "No position found for AAPL" in reason

    def test_can_close_position_already_closed(self):
        """Test validation when position is already closed."""
        position = Mock(spec=Position)
        position.is_closed.return_value = True

        self.portfolio.get_position.return_value = position

        can_close, reason = self.validator.can_close_position(self.portfolio, "AAPL")

        assert can_close is False
        assert "Position for AAPL is already closed" in reason

    def test_can_close_position_partial_valid(self):
        """Test validation for valid partial close."""
        position = Mock(spec=Position)
        position.is_closed.return_value = False
        position.quantity = Quantity(Decimal("10"))

        self.portfolio.get_position.return_value = position

        can_close, reason = self.validator.can_close_position(
            self.portfolio, "AAPL", Quantity(Decimal("5"))
        )

        assert can_close is True
        assert reason is None

    def test_can_close_position_partial_zero_quantity(self):
        """Test validation for partial close with zero quantity."""
        position = Mock(spec=Position)
        position.is_closed.return_value = False
        position.quantity = Quantity(Decimal("10"))

        self.portfolio.get_position.return_value = position

        can_close, reason = self.validator.can_close_position(
            self.portfolio, "AAPL", Quantity(Decimal("0"))
        )

        assert can_close is False
        assert "Quantity must be positive" in reason

    def test_can_close_position_partial_exceeds_available(self):
        """Test validation for partial close exceeding available quantity."""
        position = Mock(spec=Position)
        position.is_closed.return_value = False
        position.quantity = Quantity(Decimal("10"))

        self.portfolio.get_position.return_value = position

        can_close, reason = self.validator.can_close_position(
            self.portfolio, "AAPL", Quantity(Decimal("15"))
        )

        assert can_close is False
        assert "Cannot close" in reason and "only" in reason and "available" in reason

    # --- Cash Management Validation Tests ---

    def test_validate_cash_addition_success(self):
        """Test successful cash addition validation."""
        # Should not raise exception
        self.validator.validate_cash_addition(self.portfolio, Money(Decimal("1000")))

    def test_validate_cash_addition_zero_amount(self):
        """Test validation with zero cash addition."""
        with pytest.raises(ValueError, match="Cash amount must be positive"):
            self.validator.validate_cash_addition(self.portfolio, Money(Decimal("0")))

    def test_validate_cash_addition_negative_amount(self):
        """Test validation with negative cash addition."""
        with pytest.raises(ValueError, match="Cash amount must be positive"):
            self.validator.validate_cash_addition(self.portfolio, Money(Decimal("-1000")))

    def test_validate_cash_addition_overflow(self):
        """Test validation when cash addition would cause overflow."""
        self.portfolio.cash_balance = Money(Decimal("999999999"))

        with pytest.raises(ValueError, match="Cash balance would exceed maximum"):
            self.validator.validate_cash_addition(self.portfolio, Money(Decimal("2000000")))

    def test_validate_cash_deduction_success(self):
        """Test successful cash deduction validation."""
        # Should not raise exception
        self.validator.validate_cash_deduction(self.portfolio, Money(Decimal("1000")))

    def test_validate_cash_deduction_zero_amount(self):
        """Test validation with zero cash deduction."""
        with pytest.raises(ValueError, match="Cash amount must be positive"):
            self.validator.validate_cash_deduction(self.portfolio, Money(Decimal("0")))

    def test_validate_cash_deduction_negative_amount(self):
        """Test validation with negative cash deduction."""
        with pytest.raises(ValueError, match="Cash amount must be positive"):
            self.validator.validate_cash_deduction(self.portfolio, Money(Decimal("-1000")))

    def test_validate_cash_deduction_insufficient_cash(self):
        """Test validation with insufficient cash for deduction."""
        with pytest.raises(ValueError, match="Insufficient cash"):
            self.validator.validate_cash_deduction(
                self.portfolio,
                Money(Decimal("15000")),  # More than 10000 available
            )

    # --- Risk Validation Tests ---

    def test_validate_portfolio_risk_no_warnings(self):
        """Test risk validation with healthy portfolio."""
        warnings = self.validator.validate_portfolio_risk(self.portfolio)

        assert len(warnings) == 0

    def test_validate_portfolio_risk_high_concentration(self):
        """Test risk validation with high position concentration."""
        # Create position with 30% of portfolio value
        position = Mock(spec=Position)
        position.symbol = "AAPL"
        position.get_position_value.return_value = Money(Decimal("4500"))  # 30% of 15000
        position.get_unrealized_pnl.return_value = Money(Decimal("100"))  # Small profit

        self.portfolio.get_open_positions.return_value = [position]

        warnings = self.validator.validate_portfolio_risk(self.portfolio)

        assert len(warnings) > 0
        assert any("High concentration risk" in w for w in warnings)
        assert any("AAPL" in w for w in warnings)

    def test_validate_portfolio_risk_low_cash_ratio(self):
        """Test risk validation with low cash ratio."""
        # Create positions that use most of the capital
        position1 = Mock(spec=Position)
        position1.symbol = "AAPL"
        position1.get_position_value.return_value = Money(Decimal("50000"))
        position1.get_unrealized_pnl.return_value = Money(Decimal("0"))

        position2 = Mock(spec=Position)
        position2.symbol = "MSFT"
        position2.get_position_value.return_value = Money(Decimal("45000"))
        position2.get_unrealized_pnl.return_value = Money(Decimal("0"))

        self.portfolio.get_open_positions.return_value = [position1, position2]
        self.portfolio.cash_balance = Money(Decimal("1000"))  # Only 1% cash

        warnings = self.validator.validate_portfolio_risk(self.portfolio)

        assert any("Low cash ratio" in w for w in warnings)

    def test_validate_portfolio_risk_near_position_limit(self):
        """Test risk validation when near position limit."""
        self.portfolio.get_position_count.return_value = 9
        self.portfolio.max_positions = 10

        warnings = self.validator.validate_portfolio_risk(self.portfolio)

        assert any("Near position limit" in w for w in warnings)

    def test_validate_portfolio_risk_significant_drawdown(self):
        """Test risk validation with significant drawdown."""
        # Set negative P&L
        self.portfolio.total_realized_pnl = Money(Decimal("-12000"))  # 12% drawdown

        position = Mock(spec=Position)
        position.get_position_value.return_value = Money(Decimal("5000"))
        position.get_unrealized_pnl.return_value = Money(Decimal("0"))
        position.symbol = "AAPL"
        self.portfolio.get_open_positions.return_value = [position]
        self.portfolio.get_position_count.return_value = 1

        warnings = self.validator.validate_portfolio_risk(self.portfolio)

        assert any("Significant drawdown" in w for w in warnings)

    def test_validate_portfolio_risk_multiple_warnings(self):
        """Test risk validation with multiple issues."""
        # High concentration position
        position1 = Mock(spec=Position)
        position1.symbol = "AAPL"
        position1.get_position_value.return_value = Money(Decimal("30000"))

        # Low cash
        self.portfolio.cash_balance = Money(Decimal("1000"))

        # Near position limit
        self.portfolio.get_position_count.return_value = 9

        # Drawdown
        self.portfolio.total_realized_pnl = Money(Decimal("-15000"))
        position1.get_unrealized_pnl.return_value = Money(Decimal("0"))

        self.portfolio.get_open_positions.return_value = [position1]

        warnings = self.validator.validate_portfolio_risk(self.portfolio)

        assert len(warnings) >= 3

    # --- Advanced Risk Metrics Tests ---

    @patch("src.domain.services.portfolio_calculator.PortfolioCalculator")
    def test_validate_advanced_risk_metrics_var_threshold(self, mock_calculator_class):
        """Test advanced risk validation with VaR threshold."""
        mock_calculator = mock_calculator_class.return_value
        mock_calculator_class.calculate_value_at_risk.return_value = Money(Decimal("6000"))

        warnings = self.validator.validate_advanced_risk_metrics(
            self.portfolio, var_threshold=Money(Decimal("5000"))
        )

        assert any("VaR exceeds threshold" in w for w in warnings)

    def test_validate_advanced_risk_metrics_sector_concentration(self):
        """Test advanced risk validation with sector concentration."""
        # Create multiple positions in same "sector" (first 2 chars)
        positions = []
        for i in range(5):
            position = Mock(spec=Position)
            position.symbol = f"AA{i}"  # Same sector prefix
            position.get_position_value.return_value = Money(Decimal("1000"))
            positions.append(position)

        self.portfolio.get_open_positions.return_value = positions

        warnings = self.validator.validate_advanced_risk_metrics(self.portfolio)

        assert any("High sector concentration" in w for w in warnings)

    def test_validate_advanced_risk_metrics_no_warnings(self):
        """Test advanced risk validation with no issues."""
        # Diversified positions
        position1 = Mock(spec=Position)
        position1.symbol = "AAPL"

        position2 = Mock(spec=Position)
        position2.symbol = "MSFT"

        self.portfolio.get_open_positions.return_value = [position1, position2]

        warnings = self.validator.validate_advanced_risk_metrics(self.portfolio)

        assert len(warnings) == 0

    # --- Regulatory Compliance Tests ---

    def test_validate_regulatory_compliance_success(self):
        """Test regulatory compliance with valid portfolio."""
        self.portfolio.trades_count = 3
        self.portfolio.cash_balance = Money(Decimal("30000"))
        self.portfolio.get_position_count.return_value = 5
        self.portfolio.max_leverage = Decimal("2.0")

        # Should not raise exception
        self.validator.validate_regulatory_compliance(self.portfolio)

    def test_validate_regulatory_compliance_pdt_rule_violation(self):
        """Test Pattern Day Trading rule violation."""
        self.portfolio.trades_count = 5  # More than 4 day trades
        self.portfolio.cash_balance = Money(Decimal("20000"))  # Less than $25k
        self.portfolio.get_open_positions.return_value = []

        with pytest.raises(ValueError, match="Pattern Day Trading Rule"):
            self.validator.validate_regulatory_compliance(self.portfolio)

    def test_validate_regulatory_compliance_pdt_rule_compliant(self):
        """Test Pattern Day Trading rule compliance with sufficient capital."""
        self.portfolio.trades_count = 5
        self.portfolio.cash_balance = Money(Decimal("26000"))
        self.portfolio.get_open_positions.return_value = []

        # Should not raise exception
        self.validator.validate_regulatory_compliance(self.portfolio)

    def test_validate_regulatory_compliance_position_limit_exceeded(self):
        """Test regulatory position limit exceeded."""
        self.portfolio.get_position_count.return_value = 201

        with pytest.raises(ValueError, match="Exceeds regulatory position limit"):
            self.validator.validate_regulatory_compliance(self.portfolio)

    def test_validate_regulatory_compliance_leverage_exceeded(self):
        """Test regulatory leverage limit exceeded."""
        self.portfolio.max_leverage = Decimal("5.0")  # Exceeds 4:1 limit

        with pytest.raises(ValueError, match="Leverage exceeds regulatory limit"):
            self.validator.validate_regulatory_compliance(self.portfolio)

    # --- Order Validation Tests ---

    def test_validate_order_for_portfolio_buy(self):
        """Test order validation for buy order."""
        can_execute, reason = self.validator.validate_order_for_portfolio(
            self.portfolio, "AAPL", Quantity(Decimal("10")), Price(Decimal("150")), "BUY"
        )

        assert can_execute is True
        assert reason is None

    def test_validate_order_for_portfolio_buy_to_open(self):
        """Test order validation for buy to open order."""
        can_execute, reason = self.validator.validate_order_for_portfolio(
            self.portfolio, "AAPL", Quantity(Decimal("10")), Price(Decimal("150")), "BUY_TO_OPEN"
        )

        assert can_execute is True
        assert reason is None

    def test_validate_order_for_portfolio_sell(self):
        """Test order validation for sell order."""
        position = Mock(spec=Position)
        position.is_closed.return_value = False
        position.quantity = Quantity(Decimal("10"))

        self.portfolio.get_position.return_value = position

        can_execute, reason = self.validator.validate_order_for_portfolio(
            self.portfolio, "AAPL", Quantity(Decimal("5")), Price(Decimal("160")), "SELL"
        )

        assert can_execute is True
        assert reason is None

    def test_validate_order_for_portfolio_sell_to_close(self):
        """Test order validation for sell to close order."""
        position = Mock(spec=Position)
        position.is_closed.return_value = False
        position.quantity = Quantity(Decimal("10"))

        self.portfolio.get_position.return_value = position

        can_execute, reason = self.validator.validate_order_for_portfolio(
            self.portfolio, "AAPL", Quantity(Decimal("10")), Price(Decimal("160")), "SELL_TO_CLOSE"
        )

        assert can_execute is True
        assert reason is None

    def test_validate_order_for_portfolio_unknown_type(self):
        """Test order validation with unknown order type."""
        can_execute, reason = self.validator.validate_order_for_portfolio(
            self.portfolio, "AAPL", Quantity(Decimal("10")), Price(Decimal("150")), "INVALID_TYPE"
        )

        assert can_execute is False
        assert "Unknown order type" in reason

    def test_validate_order_for_portfolio_buy_insufficient_funds(self):
        """Test buy order validation with insufficient funds."""
        self.portfolio.cash_balance = Money(Decimal("100"))

        can_execute, reason = self.validator.validate_order_for_portfolio(
            self.portfolio, "AAPL", Quantity(Decimal("10")), Price(Decimal("150")), "BUY"
        )

        assert can_execute is False
        assert "Insufficient cash" in reason

    def test_validate_order_for_portfolio_sell_no_position(self):
        """Test sell order validation with no position."""
        self.portfolio.get_position.return_value = None

        can_execute, reason = self.validator.validate_order_for_portfolio(
            self.portfolio, "AAPL", Quantity(Decimal("10")), Price(Decimal("160")), "SELL"
        )

        assert can_execute is False
        assert "No position found" in reason

    # --- Edge Cases and Financial Precision Tests ---

    def test_financial_precision_calculations(self):
        """Test financial calculations maintain precision."""
        # Test with precise decimal values
        can_open, _ = self.validator.can_open_position(
            self.portfolio, "AAPL", Quantity(Decimal("3.14159")), Price(Decimal("299.99"))
        )

        assert can_open is True

    def test_very_small_position_values(self):
        """Test validation with very small position values."""
        can_open, _ = self.validator.can_open_position(
            self.portfolio,
            "PENNY",
            Quantity(Decimal("1000")),
            Price(Decimal("0.001")),  # Penny stock
        )

        assert can_open is True

    def test_very_large_position_values(self):
        """Test validation with very large position values."""
        self.portfolio.cash_balance = Money(Decimal("1000000"))
        self.portfolio.max_position_size = Money(Decimal("500000"))

        can_open, reason = self.validator.can_open_position(
            self.portfolio,
            "BRK.A",
            Quantity(Decimal("2")),
            Price(Decimal("300000")),  # Expensive stock
        )

        assert can_open is False
        assert "Position size" in reason and "exceeds maximum" in reason

    def test_fractional_shares(self):
        """Test validation with fractional shares."""
        can_open, _ = self.validator.can_open_position(
            self.portfolio,
            "AAPL",
            Quantity(Decimal("0.5")),
            Price(Decimal("150")),  # Half share
        )

        assert can_open is True

    def test_zero_commission_handling(self):
        """Test validation with zero commission trades."""
        request = Mock(spec=PositionRequest)
        request.symbol = "AAPL"
        request.quantity = Quantity(Decimal("10"))
        request.entry_price = Price(Decimal("150"))
        request.commission = Money(Decimal("0"))

        # Should not raise exception
        self.validator.validate_position_request(self.portfolio, request)
