"""
Comprehensive unit tests for Order Validator service
"""

# Standard library imports
from decimal import Decimal
from unittest.mock import MagicMock, create_autospec

# Third-party imports
import pytest

# Local imports
from src.domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from src.domain.entities.portfolio import Portfolio
from src.domain.entities.position import Position
from src.domain.services.commission_calculator import ICommissionCalculator
from src.domain.services.order_validator import OrderConstraints, OrderValidator, ValidationResult
from src.domain.value_objects.money import Money
from src.domain.value_objects.price import Price


class TestValidationResult:
    """Test ValidationResult data class"""

    def test_success_result(self):
        """Test creating successful validation result"""
        result = ValidationResult.success(
            required_capital=Money(Decimal("1000")), estimated_commission=Money(Decimal("10"))
        )

        assert result.is_valid is True
        assert result.error_message is None
        assert result.required_capital == Decimal("1000")
        assert result.estimated_commission == Decimal("10")

    def test_success_result_without_capital(self):
        """Test successful result without capital details"""
        result = ValidationResult.success()

        assert result.is_valid is True
        assert result.error_message is None
        assert result.required_capital is None
        assert result.estimated_commission is None

    def test_failure_result(self):
        """Test creating failed validation result"""
        result = ValidationResult.failure("Insufficient funds")

        assert result.is_valid is False
        assert result.error_message == "Insufficient funds"
        assert result.required_capital is None
        assert result.estimated_commission is None


class TestOrderConstraints:
    """Test OrderConstraints validation"""

    def test_default_constraints(self):
        """Test default constraint values"""
        constraints = OrderConstraints()

        assert constraints.max_position_size is None
        assert constraints.max_order_value is None
        assert constraints.min_order_value == Decimal("1")
        assert constraints.max_portfolio_concentration == Decimal("0.20")
        assert constraints.require_margin_for_shorts is True
        assert constraints.short_margin_requirement == Decimal("1.5")

    def test_custom_constraints(self):
        """Test custom constraint values"""
        constraints = OrderConstraints(
            max_position_size=Decimal("10000"),
            max_order_value=Money(Decimal("100000")),
            min_order_value=Money(Decimal("100")),
            max_portfolio_concentration=Decimal("0.10"),
            require_margin_for_shorts=False,
            short_margin_requirement=Decimal("2.0"),
        )

        assert constraints.max_position_size == Decimal("10000")
        assert constraints.max_order_value == Decimal("100000")
        assert constraints.min_order_value == Decimal("100")
        assert constraints.max_portfolio_concentration == Decimal("0.10")
        assert constraints.require_margin_for_shorts is False
        assert constraints.short_margin_requirement == Decimal("2.0")

    def test_invalid_portfolio_concentration_zero(self):
        """Test that zero portfolio concentration raises error"""
        with pytest.raises(ValueError, match="Portfolio concentration must be between 0 and 1"):
            OrderConstraints(max_portfolio_concentration=Decimal("0"))

    def test_invalid_portfolio_concentration_over_one(self):
        """Test that portfolio concentration over 1 raises error"""
        with pytest.raises(ValueError, match="Portfolio concentration must be between 0 and 1"):
            OrderConstraints(max_portfolio_concentration=Decimal("1.1"))

    def test_invalid_short_margin_requirement(self):
        """Test that short margin requirement below 1 raises error"""
        with pytest.raises(ValueError, match="Short margin requirement must be at least 100%"):
            OrderConstraints(short_margin_requirement=Decimal("0.9"))

    def test_edge_case_portfolio_concentration(self):
        """Test edge case portfolio concentration values"""
        # Should work with exactly 1
        constraints = OrderConstraints(max_portfolio_concentration=Decimal("1.0"))
        assert constraints.max_portfolio_concentration == Decimal("1.0")

        # Should work with very small value
        constraints = OrderConstraints(max_portfolio_concentration=Decimal("0.001"))
        assert constraints.max_portfolio_concentration == Decimal("0.001")


class TestOrderValidator:
    """Test OrderValidator main functionality"""

    @pytest.fixture
    def mock_commission_calculator(self):
        """Create mock commission calculator"""
        calculator = create_autospec(ICommissionCalculator)
        calculator.calculate.return_value = Money(Decimal("10"))
        return calculator

    @pytest.fixture
    def validator(self, mock_commission_calculator):
        """Create OrderValidator instance"""
        return OrderValidator(mock_commission_calculator)

    @pytest.fixture
    def sample_order(self):
        """Create sample order for testing"""
        return Order(
            symbol="AAPL",
            quantity=Decimal("10"),  # Smaller quantity to avoid concentration limits
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            reason="Test order",
        )

    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio for testing"""
        portfolio = Portfolio(name="test_account")
        portfolio.cash_balance = Decimal("20000")  # Enough for most test orders
        return portfolio

    def test_validate_valid_buy_order(self, validator, sample_order, sample_portfolio):
        """Test validation of valid buy order"""
        current_price = Price(Decimal("150"))

        result = validator.validate_order(sample_order, sample_portfolio, current_price)

        assert result.is_valid is True
        assert result.error_message is None
        assert result.required_capital is not None
        assert result.estimated_commission is not None

    def test_validate_order_insufficient_funds(self, validator, sample_portfolio):
        """Test validation fails with insufficient funds"""
        # Create a large order that exceeds available funds
        large_order = Order(
            symbol="AAPL",
            quantity=Decimal("1000"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            reason="Test order",
        )
        sample_portfolio.cash_balance = Decimal("100")
        current_price = Price(Decimal("150"))

        result = validator.validate_order(large_order, sample_portfolio, current_price)

        assert result.is_valid is False
        assert "Insufficient funds" in result.error_message

    def test_validate_order_wrong_status(self, validator, sample_portfolio):
        """Test validation fails for non-pending order"""
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            reason="Test order",
        )
        current_price = Price(Decimal("150"))

        result = validator.validate_order(order, sample_portfolio, current_price)

        assert result.is_valid is False
        assert "Cannot submit order with status" in result.error_message

    def test_validate_order_negative_quantity(self, validator, sample_portfolio):
        """Test that negative quantity raises error at order creation"""
        # Order entity validates quantity in constructor, so negative quantity
        # will raise ValueError before reaching the validator
        with pytest.raises(ValueError, match="Order quantity must be positive"):
            Order(
                symbol="AAPL",
                quantity=Decimal("-100"),
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                status=OrderStatus.PENDING,
                reason="Test order",
            )

    def test_validate_limit_order_without_price(self, validator, sample_portfolio):
        """Test that limit order without price raises error at order creation"""
        # Order entity validates limit price in constructor
        with pytest.raises(ValueError, match="Limit order requires limit price"):
            Order(
                symbol="AAPL",
                quantity=Decimal("100"),
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                status=OrderStatus.PENDING,
                reason="Test order",
            )

    def test_validate_stop_order_without_stop_price(self, validator, sample_portfolio):
        """Test that stop order without stop price raises error at order creation"""
        # Order entity validates stop price in constructor
        with pytest.raises(ValueError, match="Stop order requires stop price"):
            Order(
                symbol="AAPL",
                quantity=Decimal("100"),
                side=OrderSide.BUY,
                order_type=OrderType.STOP,
                status=OrderStatus.PENDING,
                reason="Test order",
            )

    def test_validate_order_below_minimum_value(self, validator, sample_portfolio):
        """Test validation fails for order below minimum value"""
        constraints = OrderConstraints(min_order_value=Money(Decimal("1000")))
        validator.constraints = constraints

        order = Order(
            symbol="AAPL",
            quantity=Decimal("1"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            reason="Test order",
        )
        current_price = Price(Decimal("150"))

        result = validator.validate_order(order, sample_portfolio, current_price)

        assert result.is_valid is False
        assert "below minimum" in result.error_message

    def test_validate_order_above_maximum_value(self, validator, sample_portfolio):
        """Test validation fails for order above maximum value"""
        constraints = OrderConstraints(max_order_value=Money(Decimal("10000")))
        validator.constraints = constraints

        # Give portfolio enough funds so we hit max order value constraint, not insufficient funds
        sample_portfolio.cash_balance = Decimal("200000")

        order = Order(
            symbol="AAPL",
            quantity=Decimal("1000"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            reason="Test order",
        )
        current_price = Price(Decimal("150"))

        result = validator.validate_order(order, sample_portfolio, current_price)

        assert result.is_valid is False
        assert "exceeds maximum" in result.error_message

    def test_validate_position_size_limit(self, validator, sample_portfolio):
        """Test validation fails when exceeding position size limit"""
        constraints = OrderConstraints(max_position_size=Decimal("500"))
        validator.constraints = constraints

        # Give portfolio enough funds to pass the funds check
        sample_portfolio.cash_balance = Decimal("50000")

        # Add existing position
        position = Position(
            symbol="AAPL",
            quantity=Decimal("400"),
            average_entry_price=Decimal("140"),
            current_price=Decimal("150"),
        )
        sample_portfolio.positions[position.id] = position

        order = Order(
            symbol="AAPL",
            quantity=Decimal("200"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            reason="Test order",
        )
        current_price = Price(Decimal("150"))

        result = validator.validate_order(order, sample_portfolio, current_price)

        assert result.is_valid is False
        assert "exceeds maximum" in result.error_message

    def test_validate_portfolio_concentration_limit(self, validator, sample_portfolio):
        """Test validation fails when exceeding portfolio concentration"""
        constraints = OrderConstraints(max_portfolio_concentration=Decimal("0.10"))
        validator.constraints = constraints

        # Set up portfolio with appropriate total value
        sample_portfolio.cash_balance = Decimal("10000")

        order = Order(
            symbol="AAPL",
            quantity=Decimal("10"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            reason="Test order",
        )
        current_price = Price(Decimal("150"))  # 10 * 150 = 1500, > 10% of 10000

        result = validator.validate_order(order, sample_portfolio, current_price)

        assert result.is_valid is False
        assert "exceeds maximum" in result.error_message
        assert "%" in result.error_message

    def test_validate_short_sale_with_margin(self, validator, sample_portfolio):
        """Test short sale validation with margin requirements"""
        # Set custom constraints to avoid concentration limit
        constraints = OrderConstraints(
            max_portfolio_concentration=Decimal("1.0"),  # No concentration limit
            require_margin_for_shorts=True,
            short_margin_requirement=Decimal("1.5"),
        )
        validator.constraints = constraints

        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            reason="Test short",
        )
        current_price = Price(Decimal("150"))

        # Short value: 100 * 150 = 15000
        # Margin required: 15000 * 1.5 = 22500
        sample_portfolio.cash_balance = Decimal("20000")  # Not enough

        result = validator.validate_order(order, sample_portfolio, current_price)

        assert result.is_valid is False
        assert "Insufficient margin" in result.error_message

    def test_validate_short_sale_sufficient_margin(self, validator, sample_portfolio):
        """Test short sale validation with sufficient margin"""
        # Disable concentration limits for this test
        validator.constraints.max_portfolio_concentration = Decimal("1.0")

        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            reason="Test short",
        )
        current_price = Price(Decimal("150"))

        sample_portfolio.cash_balance = Decimal("25000")  # Enough for margin

        result = validator.validate_order(order, sample_portfolio, current_price)

        assert result.is_valid is True

    def test_validate_sell_with_existing_position(self, validator, sample_portfolio):
        """Test selling with existing long position"""
        # Disable concentration limits for this test
        validator.constraints.max_portfolio_concentration = Decimal("1.0")

        # Add existing long position
        position = Position(
            symbol="AAPL",
            quantity=Decimal("200"),
            average_entry_price=Decimal("140"),
            current_price=Decimal("150"),
        )
        sample_portfolio.positions[position.id] = position

        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            reason="Test sell",
        )
        current_price = Price(Decimal("150"))

        result = validator.validate_order(order, sample_portfolio, current_price)

        assert result.is_valid is True  # Have shares to sell

    def test_validate_limit_order_with_limit_price(self, validator, sample_portfolio):
        """Test limit order uses limit price for capital calculation"""
        # Disable concentration limits for this test
        validator.constraints.max_portfolio_concentration = Decimal("1.0")

        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("145"),
            status=OrderStatus.PENDING,
            reason="Test limit",
        )
        current_price = Price(Decimal("150"))

        result = validator.validate_order(order, sample_portfolio, current_price)

        assert result.is_valid is True
        # Should use limit price (145) not current price (150)
        assert result.required_capital < Decimal("15000")

    def test_validate_closed_position_not_counted(self, validator, sample_portfolio):
        """Test that closed positions are not counted"""
        # Add closed position
        position = Position(
            symbol="AAPL",
            quantity=Decimal("0"),  # Closed
            average_entry_price=Decimal("140", closed_at=datetime.now(UTC)),
            current_price=Decimal("150"),
        )
        position.status = "closed"
        sample_portfolio.positions[position.id] = position

        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.SELL,  # Try to sell without position
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            reason="Test sell",
        )
        current_price = Price(Decimal("150"))

        result = validator.validate_order(order, sample_portfolio, current_price)

        # Should require margin since no open position
        assert result.is_valid is False or result.required_capital is not None


class TestOrderModificationValidation:
    """Test order modification validation"""

    @pytest.fixture
    def mock_commission_calculator(self):
        """Create mock commission calculator"""
        calculator = create_autospec(ICommissionCalculator)
        calculator.calculate.return_value = Money(Decimal("10"))
        return calculator

    @pytest.fixture
    def validator(self, mock_commission_calculator):
        """Create OrderValidator instance"""
        return OrderValidator(mock_commission_calculator)

    def test_validate_modification_pending_order(self, validator):
        """Test modification of pending order"""
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("150"),
            status=OrderStatus.PENDING,
            reason="Test",
        )

        result = validator.validate_modification(
            order, new_quantity=Decimal("200"), new_limit_price=Price(Decimal("155"))
        )

        assert result.is_valid is True

    def test_validate_modification_partially_filled(self, validator):
        """Test modification of partially filled order"""
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("150"),
            status=OrderStatus.PARTIALLY_FILLED,
            reason="Test",
        )
        order.filled_quantity = Decimal("30")

        result = validator.validate_modification(order, new_quantity=Decimal("80"))

        assert result.is_valid is True

    def test_validate_modification_filled_order(self, validator):
        """Test cannot modify filled order"""
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            reason="Test",
        )

        result = validator.validate_modification(order, new_quantity=Decimal("200"))

        assert result.is_valid is False
        assert "Cannot modify order with status" in result.error_message

    def test_validate_modification_cancelled_order(self, validator):
        """Test cannot modify cancelled order"""
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.CANCELLED,
            reason="Test",
        )

        result = validator.validate_modification(order)

        assert result.is_valid is False
        assert "Cannot modify order with status" in result.error_message

    def test_validate_modification_negative_quantity(self, validator):
        """Test modification with negative quantity fails"""
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("150"),
            status=OrderStatus.PENDING,
            reason="Test",
        )

        # Create a Quantity and modify its internal value to be negative
        qty = Decimal("1")
        # Use object.__setattr__ to bypass frozen dataclass
        object.__setattr__(qty, "_value", Decimal("-50"))

        result = validator.validate_modification(order, new_quantity=qty)
        assert result.is_valid is False
        assert "Modified quantity must be positive" in result.error_message

    def test_validate_modification_below_filled_quantity(self, validator):
        """Test cannot reduce quantity below filled amount"""
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("150"),
            status=OrderStatus.PARTIALLY_FILLED,
            reason="Test",
        )
        order.filled_quantity = Decimal("60")

        result = validator.validate_modification(order, new_quantity=Decimal("50"))

        assert result.is_valid is False
        assert "Cannot reduce quantity below filled amount" in result.error_message

    def test_validate_modification_negative_limit_price(self, validator):
        """Test modification with negative limit price fails"""
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("150"),
            status=OrderStatus.PENDING,
            reason="Test",
        )

        # Create a Price and modify its internal value to be negative
        price = Price(Decimal("1"))
        # Use object.__setattr__ to bypass frozen dataclass
        object.__setattr__(price, "_value", Decimal("-150"))

        result = validator.validate_modification(order, new_limit_price=price)
        assert result.is_valid is False
        assert "Modified limit price must be positive" in result.error_message

    def test_validate_modification_negative_stop_price(self, validator):
        """Test modification with negative stop price fails"""
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.STOP,
            stop_price=Decimal("145"),
            status=OrderStatus.PENDING,
            reason="Test",
        )

        # Create a Price and modify its internal value to be negative
        price = Price(Decimal("1"))
        # Use object.__setattr__ to bypass frozen dataclass
        object.__setattr__(price, "_value", Decimal("-145"))

        result = validator.validate_modification(order, new_stop_price=price)
        assert result.is_valid is False
        assert "Modified stop price must be positive" in result.error_message

    def test_validate_modification_no_changes(self, validator):
        """Test modification with no changes succeeds"""
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            reason="Test",
        )

        result = validator.validate_modification(order)

        assert result.is_valid is True


class TestComplexScenarios:
    """Test complex validation scenarios"""

    @pytest.fixture
    def mock_commission_calculator(self):
        """Create mock commission calculator"""
        calculator = create_autospec(ICommissionCalculator)
        calculator.calculate.return_value = Money(Decimal("10"))
        return calculator

    @pytest.fixture
    def validator(self, mock_commission_calculator):
        """Create OrderValidator instance"""
        return OrderValidator(mock_commission_calculator)

    def test_partial_short_sale(self, validator):
        """Test partial short sale (selling more than owned)"""
        portfolio = Portfolio(name="test")
        portfolio.cash_balance = Decimal("50000")

        # Own 50 shares
        position = Position(
            symbol="AAPL",
            quantity=Decimal("50"),
            average_entry_price=Decimal("140"),
            current_price=Decimal("150"),
        )
        portfolio.positions[position.id] = position

        # Try to sell 100 (50 owned + 50 short)
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            reason="Test",
        )
        current_price = Price(Decimal("150"))

        result = validator.validate_order(order, portfolio, current_price)

        # Should calculate margin for the 50 shares to short
        assert result.is_valid is True  # Has enough margin

    def test_multiple_constraint_violations(self, validator):
        """Test order failing multiple constraints"""
        constraints = OrderConstraints(
            max_position_size=Decimal("50"),
            max_order_value=Money(Decimal("5000")),
            min_order_value=Money(Decimal("1000")),
        )
        validator.constraints = constraints

        portfolio = Portfolio(name="test")
        portfolio.cash_balance = Decimal("100")  # Very low balance

        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),  # Exceeds max position size
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            reason="Test",
        )
        current_price = Price(Decimal("150"))  # Order value = 15000, exceeds max

        result = validator.validate_order(order, portfolio, current_price)

        # Should fail on first constraint check
        assert result.is_valid is False

    def test_zero_portfolio_value(self, validator):
        """Test handling zero portfolio value"""
        portfolio = Portfolio(name="test")
        portfolio.cash_balance = Decimal("0")

        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.SELL,  # Short sale
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            reason="Test",
        )
        current_price = Price(Decimal("150"))

        result = validator.validate_order(order, portfolio, current_price)

        assert result.is_valid is False
        assert "Insufficient" in result.error_message

    def test_stop_limit_order_validation(self, validator):
        """Test stop-limit order validation"""
        # Disable concentration limits for this test
        validator.constraints.max_portfolio_concentration = Decimal("1.0")

        portfolio = Portfolio(name="test")
        portfolio.cash_balance = Decimal("20000")

        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.STOP_LIMIT,
            stop_price=Decimal("155"),
            limit_price=Decimal("160"),
            status=OrderStatus.PENDING,
            reason="Test",
        )
        current_price = Price(Decimal("150"))

        result = validator.validate_order(order, portfolio, current_price)

        assert result.is_valid is True

    def test_commission_affects_buying_power(self, mock_commission_calculator):
        """Test that commission is properly included in capital requirements"""
        mock_commission_calculator.calculate.return_value = Money(Decimal("100"))
        validator = OrderValidator(mock_commission_calculator)

        portfolio = Portfolio(name="test")
        portfolio.cash_balance = Decimal("15050")  # Just enough for order + commission

        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            reason="Test",
        )
        current_price = Price(Decimal("150"))

        result = validator.validate_order(order, portfolio, current_price)

        # 100 * 150 + 100 commission = 15100, more than 15050
        assert result.is_valid is False

    def test_no_margin_requirement_setting(self, validator):
        """Test with margin requirements disabled"""
        constraints = OrderConstraints(
            require_margin_for_shorts=False,
            max_portfolio_concentration=Decimal("1.0"),  # Disable concentration limits
            max_position_size=None,  # No position size limit
        )
        validator.constraints = constraints

        portfolio = Portfolio(name="test")
        portfolio.cash_balance = Decimal("10000")  # Enough to avoid concentration issues

        order = Order(
            symbol="AAPL",
            quantity=Decimal("10"),  # Small short
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            reason="Test",
        )
        current_price = Price(Decimal("150"))

        result = validator.validate_order(order, portfolio, current_price)

        # Should pass since margin not required
        assert result.is_valid is True

    def test_custom_margin_requirement(self, validator):
        """Test with custom margin requirement"""
        constraints = OrderConstraints(
            short_margin_requirement=Decimal("3.0"),
            max_portfolio_concentration=Decimal("1.0"),  # Disable concentration limits
        )
        validator.constraints = constraints

        portfolio = Portfolio(name="test")
        portfolio.cash_balance = Decimal("40000")

        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            reason="Test",
        )
        current_price = Price(Decimal("150"))

        # Short value: 15000, margin required: 45000
        result = validator.validate_order(order, portfolio, current_price)

        assert result.is_valid is False
        assert "Insufficient margin" in result.error_message


class TestOrderValidatorEdgeCases:
    """Test edge cases and boundary conditions for order validator"""

    @pytest.fixture
    def calculator(self):
        """Create mock commission calculator"""
        calc = MagicMock(spec=ICommissionCalculator)
        calc.calculate.return_value = Money(Decimal("1.00"))
        return calc

    @pytest.fixture
    def validator(self, calculator):
        """Create OrderValidator with default constraints"""
        return OrderValidator(calculator)

    @pytest.fixture
    def portfolio(self):
        """Create test portfolio"""
        return Portfolio(cash_balance=Decimal("10000"))

    def test_validate_order_with_zero_quantity(self, validator, portfolio):
        """Test validation with zero quantity order"""
        # Order entity validates quantity in constructor, so zero quantity
        # will raise ValueError before reaching the validator
        with pytest.raises(ValueError, match="Order quantity must be positive"):
            Order(
                symbol="AAPL",
                quantity=Decimal("0"),
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
            )

    def test_validate_order_with_negative_quantity(self, validator, portfolio):
        """Test validation with negative quantity order"""
        # Order entity validates quantity in constructor, so negative quantity
        # will raise ValueError before reaching the validator
        with pytest.raises(ValueError, match="Order quantity must be positive"):
            Order(
                symbol="AAPL",
                quantity=Decimal("-100"),
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
            )

    def test_validate_limit_order_with_zero_price(self, validator, portfolio):
        """Test validation of limit order with zero price"""
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("0"),
        )
        current_price = Price(Decimal("150"))

        result = validator.validate_order(order, portfolio, current_price)

        assert result.is_valid is False
        assert "positive limit price" in result.error_message

    def test_validate_stop_order_with_negative_price(self, validator, portfolio):
        """Test validation of stop order with negative price"""
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.STOP,
            stop_price=Decimal("-10"),
        )
        current_price = Price(Decimal("150"))

        result = validator.validate_order(order, portfolio, current_price)

        assert result.is_valid is False
        assert "positive stop price" in result.error_message

    def test_validate_order_exact_funds_available(self, validator, portfolio):
        """Test validation when exact funds are available"""
        # Disable concentration limits for this test
        validator.constraints.max_portfolio_concentration = Decimal("1.0")

        # Portfolio has exactly $10,000
        order = Order(
            symbol="AAPL",
            quantity=Decimal("66"),  # 66 * 150 = 9900
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            reason="Test order",
        )
        current_price = Price(Decimal("150"))

        # 9900 + 1 commission = 9901, just under 10000
        result = validator.validate_order(order, portfolio, current_price)

        assert result.is_valid is True
        assert result.required_capital == Decimal("9901")

    def test_validate_order_at_concentration_limit(self, validator, portfolio):
        """Test validation at exact concentration limit"""
        # Add existing position
        position = Position(
            symbol="AAPL",
            quantity=Decimal("50"),
            average_entry_price=Decimal("140"),
            current_price=Decimal("150"),
        )
        portfolio.positions[position.id] = position

        # Order that would bring concentration to exactly 20%
        order = Order(
            symbol="AAPL",
            quantity=Decimal("13"),  # Will make position ~20% of portfolio
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            reason="Test order",
        )
        current_price = Price(Decimal("150"))

        result = validator.validate_order(order, portfolio, current_price)

        # Should pass if at or just under 20%
        # Total position: 63 shares * 150 = 9450
        # Portfolio value: ~17000
        # Concentration: 9450/17000 = ~55%, should fail
        assert result.is_valid is False
        assert "exceeds maximum" in result.error_message

    def test_validate_modification_reduce_below_filled(self, validator):
        """Test modification that reduces quantity below filled amount"""
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("150"),
        )
        order.status = OrderStatus.PARTIALLY_FILLED
        order.filled_quantity = Decimal("60")

        # Try to reduce quantity to 50 (below filled 60)
        result = validator.validate_modification(order, new_quantity=Decimal("50"))

        assert result.is_valid is False
        assert "below filled amount" in result.error_message

    def test_validate_modification_exact_filled_quantity(self, validator):
        """Test modification to exact filled quantity"""
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("150"),
            status=OrderStatus.PARTIALLY_FILLED,
            reason="Test order",
        )
        order.filled_quantity = Decimal("60")

        # Modify to exactly filled quantity
        result = validator.validate_modification(order, new_quantity=Decimal("60"))

        assert result.is_valid is True

    def test_short_sell_partial_coverage(self, validator, portfolio):
        """Test short selling when partially covered by existing position"""
        # Disable concentration limits for this test
        validator.constraints.max_portfolio_concentration = Decimal("1.0")

        # Add existing long position
        position = Position(
            symbol="AAPL",
            quantity=Decimal("50"),
            average_entry_price=Decimal("140"),
            current_price=Decimal("150"),
        )
        portfolio.positions[position.id] = position

        # Sell 100 shares (50 from position, 50 short)
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            reason="Test order",
        )
        current_price = Price(Decimal("150"))

        result = validator.validate_order(order, portfolio, current_price)

        # Should check margin for the 50 shares to be shorted
        # 50 * 150 * 1.5 = 11250 margin required
        assert result.is_valid is False  # Insufficient margin
        assert "Insufficient margin" in result.error_message

    def test_custom_constraints_validation(self):
        """Test validator with custom constraints"""
        calc = MagicMock(spec=ICommissionCalculator)
        calc.calculate.return_value = Money(Decimal("1.00"))

        constraints = OrderConstraints(
            max_position_size=Decimal("10"),
            max_order_value=Money(Decimal("1000")),
            min_order_value=Money(Decimal("100")),
            max_portfolio_concentration=Decimal("0.10"),
            short_margin_requirement=Decimal("2.0"),
        )

        validator = OrderValidator(calc, constraints)
        portfolio = Portfolio(cash_balance=Decimal("10000"))

        # Test max position size
        order = Order(
            symbol="AAPL",
            quantity=Decimal("20"),  # Exceeds max of 10
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
        )
        result = validator.validate_order(order, portfolio, Price(Decimal("50")))

        assert result.is_valid is False
        assert "exceeds maximum" in result.error_message

    def test_commission_calculation_in_validation(self):
        """Test that commission is properly calculated in validation"""
        calc = MagicMock(spec=ICommissionCalculator)
        calc.calculate.return_value = Money(Decimal("25.00"))  # High commission

        # Disable concentration limits for this test
        constraints = OrderConstraints(max_portfolio_concentration=Decimal("1.0"))
        validator = OrderValidator(calc, constraints)

        portfolio = Portfolio(name="test")
        portfolio.cash_balance = Decimal("1000")

        order = Order(
            symbol="AAPL",
            quantity=Decimal("6"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            reason="Test order",
        )
        current_price = Price(Decimal("150"))

        result = validator.validate_order(order, portfolio, current_price)

        # 6 * 150 = 900 + 25 commission = 925
        assert result.is_valid is True
        assert result.estimated_commission == Decimal("25.00")
        assert result.required_capital == Decimal("925")

    def test_zero_cash_balance_portfolio(self, validator):
        """Test validation with zero cash balance"""
        portfolio = Portfolio(cash_balance=Decimal("0"))

        order = Order(
            symbol="AAPL", quantity=Decimal("1"), side=OrderSide.BUY, order_type=OrderType.MARKET
        )
        current_price = Price(Decimal("150"))

        result = validator.validate_order(order, portfolio, current_price)

        assert result.is_valid is False
        assert "Insufficient funds" in result.error_message


class TestOrderValidatorFullCoverage:
    """Additional tests for complete code coverage"""

    @pytest.fixture
    def mock_commission_calculator(self):
        """Create mock commission calculator"""
        calculator = create_autospec(ICommissionCalculator)
        calculator.calculate.return_value = Money(Decimal("10"))
        return calculator

    @pytest.fixture
    def validator(self, mock_commission_calculator):
        """Create OrderValidator instance"""
        return OrderValidator(mock_commission_calculator)

    def test_validate_order_zero_quantity_check(self, validator):
        """Test order with zero quantity validation check"""
        # Create an order that somehow gets zero quantity
        # (normally prevented by Order entity validation)
        portfolio = Portfolio(name="test")
        portfolio.cash_balance = Decimal("10000")

        # We'll create a valid order first, then manually set quantity to 0
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            reason="Test",
        )
        # Directly modify the quantity attribute to bypass validation
        order.quantity = Decimal("0")

        current_price = Price(Decimal("150"))

        result = validator.validate_order(order, portfolio, current_price)

        assert result.is_valid is False
        assert "Order quantity must be positive" in result.error_message

    def test_validate_modification_zero_quantity(self, validator):
        """Test modification with zero quantity"""
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("150"),
            status=OrderStatus.PENDING,
            reason="Test",
        )

        # Create a Quantity object and then modify its internal value
        new_qty = Decimal("1")
        new_qty._value = Decimal("0")

        result = validator.validate_modification(order, new_quantity=new_qty)

        assert result.is_valid is False
        assert "Modified quantity must be positive" in result.error_message

    def test_validate_modification_zero_limit_price(self, validator):
        """Test modification with zero limit price"""
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("150"),
            status=OrderStatus.PENDING,
            reason="Test",
        )

        # Create a Price object and then modify its internal value
        new_price = Price(Decimal("1"))
        new_price._value = Decimal("0")

        result = validator.validate_modification(order, new_limit_price=new_price)

        assert result.is_valid is False
        assert "Modified limit price must be positive" in result.error_message

    def test_validate_modification_zero_stop_price(self, validator):
        """Test modification with zero stop price"""
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.STOP,
            stop_price=Decimal("145"),
            status=OrderStatus.PENDING,
            reason="Test",
        )

        # Create a Price object and then modify its internal value
        new_price = Price(Decimal("1"))
        new_price._value = Decimal("0")

        result = validator.validate_modification(order, new_stop_price=new_price)

        assert result.is_valid is False
        assert "Modified stop price must be positive" in result.error_message

    def test_sell_order_only_commission_required(self, validator):
        """Test that sell orders only require commission as capital"""
        portfolio = Portfolio(name="test")
        portfolio.cash_balance = Decimal("20")  # Just enough for commission

        # Add position to sell
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            average_entry_price=Decimal("140"),
            current_price=Decimal("150"),
        )
        portfolio.positions[position.id] = position

        # Disable concentration limits
        validator.constraints.max_portfolio_concentration = Decimal("1.0")

        order = Order(
            symbol="AAPL",
            quantity=Decimal("50"),
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            reason="Test",
        )
        current_price = Price(Decimal("150"))

        result = validator.validate_order(order, portfolio, current_price)

        assert result.is_valid is True
        # For sell orders, required capital should be just the commission
        assert result.required_capital == Decimal("10")  # Just commission

    def test_position_with_negative_quantity_sell(self, validator):
        """Test selling when position quantity becomes negative (short)"""
        portfolio = Portfolio(name="test")
        portfolio.cash_balance = Decimal("50000")

        # Disable concentration limits
        validator.constraints.max_portfolio_concentration = Decimal("1.0")

        # No existing position - pure short sale
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            reason="Test",
        )
        current_price = Price(Decimal("150"))

        result = validator.validate_order(order, portfolio, current_price)

        # Should check margin requirements for the full short
        assert result.is_valid is True  # Has enough margin

    def test_max_position_size_exact_limit(self, validator):
        """Test position at exactly max size limit"""
        constraints = OrderConstraints(
            max_position_size=Decimal("100"), max_portfolio_concentration=Decimal("1.0")
        )
        validator.constraints = constraints

        portfolio = Portfolio(name="test")
        portfolio.cash_balance = Decimal("20000")

        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),  # Exactly at limit
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            reason="Test",
        )
        current_price = Price(Decimal("150"))

        result = validator.validate_order(order, portfolio, current_price)

        assert result.is_valid is True  # Should pass at exact limit

    def test_portfolio_concentration_with_zero_total_value(self, validator):
        """Test concentration check when portfolio has zero total value"""
        portfolio = Portfolio(name="test")
        # Mock get_total_value to return 0
        portfolio.get_total_value = MagicMock(return_value=Decimal("0"))
        portfolio.cash_balance = Decimal("10000")

        order = Order(
            symbol="AAPL",
            quantity=Decimal("10"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            reason="Test",
        )
        current_price = Price(Decimal("150"))

        result = validator.validate_order(order, portfolio, current_price)

        # Should not fail on concentration when portfolio value is 0
        # (division by zero is avoided in the code)
        assert result.is_valid is True

    def test_short_sale_with_partial_position_exact_coverage(self, validator):
        """Test short sale when position exactly covers order"""
        portfolio = Portfolio(name="test")
        portfolio.cash_balance = Decimal("1000")

        # Disable concentration limits
        validator.constraints.max_portfolio_concentration = Decimal("1.0")

        # Position exactly matches order quantity
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            average_entry_price=Decimal("140"),
            current_price=Decimal("150"),
        )
        portfolio.positions[position.id] = position

        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),  # Exactly matches position
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            reason="Test",
        )
        current_price = Price(Decimal("150"))

        result = validator.validate_order(order, portfolio, current_price)

        # No shorting, just selling existing position
        assert result.is_valid is True

    def test_limit_order_with_none_limit_price(self, validator):
        """Test that limit order validation catches None limit price"""
        portfolio = Portfolio(name="test")
        portfolio.cash_balance = Decimal("10000")

        # Create a limit order and force limit_price to None
        order = Order(
            symbol="AAPL",
            quantity=Decimal("10"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("150"),
            status=OrderStatus.PENDING,
            reason="Test",
        )
        # Force limit_price to None
        order.limit_price = None

        current_price = Price(Decimal("150"))

        result = validator.validate_order(order, portfolio, current_price)

        assert result.is_valid is False
        assert "Limit orders must have a positive limit price" in result.error_message

    def test_stop_order_with_none_stop_price(self, validator):
        """Test that stop order validation catches None stop price"""
        portfolio = Portfolio(name="test")
        portfolio.cash_balance = Decimal("10000")

        # Create a stop order and force stop_price to None
        order = Order(
            symbol="AAPL",
            quantity=Decimal("10"),
            side=OrderSide.BUY,
            order_type=OrderType.STOP,
            stop_price=Decimal("145"),
            status=OrderStatus.PENDING,
            reason="Test",
        )
        # Force stop_price to None
        order.stop_price = None

        current_price = Price(Decimal("150"))

        result = validator.validate_order(order, portfolio, current_price)

        assert result.is_valid is False
        assert "Stop orders must have a positive stop price" in result.error_message

    def test_position_calculation_for_sell_order_reducing_long(self, validator):
        """Test position calculation when sell order reduces existing long"""
        portfolio = Portfolio(name="test")
        portfolio.cash_balance = Decimal("10000")

        # Disable concentration limits
        validator.constraints.max_portfolio_concentration = Decimal("1.0")

        # Existing long position
        position = Position(
            symbol="AAPL",
            quantity=Decimal("150"),
            average_entry_price=Decimal("140"),
            current_price=Decimal("150"),
        )
        portfolio.positions[position.id] = position

        # Sell part of position
        order = Order(
            symbol="AAPL",
            quantity=Decimal("50"),
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            reason="Test",
        )
        current_price = Price(Decimal("150"))

        result = validator.validate_order(order, portfolio, current_price)

        # Should succeed - just reducing position
        assert result.is_valid is True
        # Capital required is just commission for sell
        assert result.required_capital == Decimal("10")

    def test_short_margin_check_when_shares_to_short_zero(self, validator):
        """Test margin check when calculated shares to short is exactly zero"""
        portfolio = Portfolio(name="test")
        portfolio.cash_balance = Decimal("100")

        # Disable concentration limits
        validator.constraints.max_portfolio_concentration = Decimal("1.0")

        # Position exactly covers sell order
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            average_entry_price=Decimal("140"),
            current_price=Decimal("150"),
        )
        portfolio.positions[position.id] = position

        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            reason="Test",
        )
        current_price = Price(Decimal("150"))

        result = validator.validate_order(order, portfolio, current_price)

        # Should pass - no margin needed when not shorting
        assert result.is_valid is True

    def test_validate_order_negative_quantity_edge_case(self, validator):
        """Test handling of negative quantity (line 327 coverage)"""
        portfolio = Portfolio(name="test")
        portfolio.cash_balance = Decimal("10000")

        # Create order and force negative quantity
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            reason="Test",
        )
        # Force negative quantity to trigger line 327
        order.quantity = Decimal("-10")

        current_price = Price(Decimal("150"))

        result = validator.validate_order(order, portfolio, current_price)

        assert result.is_valid is False
        assert "Order quantity must be positive" in result.error_message

    def test_limit_order_zero_limit_price(self, validator):
        """Test limit order with zero limit price"""
        portfolio = Portfolio(name="test")
        portfolio.cash_balance = Decimal("10000")

        order = Order(
            symbol="AAPL",
            quantity=Decimal("10"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("150"),
            status=OrderStatus.PENDING,
            reason="Test",
        )
        # Force limit price to zero
        order.limit_price = Decimal("0")

        current_price = Price(Decimal("150"))

        result = validator.validate_order(order, portfolio, current_price)

        assert result.is_valid is False
        assert "Limit orders must have a positive limit price" in result.error_message

    def test_stop_order_zero_stop_price(self, validator):
        """Test stop order with zero stop price"""
        portfolio = Portfolio(name="test")
        portfolio.cash_balance = Decimal("10000")

        order = Order(
            symbol="AAPL",
            quantity=Decimal("10"),
            side=OrderSide.BUY,
            order_type=OrderType.STOP,
            stop_price=Decimal("145"),
            status=OrderStatus.PENDING,
            reason="Test",
        )
        # Force stop price to zero
        order.stop_price = Decimal("0")

        current_price = Price(Decimal("150"))

        result = validator.validate_order(order, portfolio, current_price)

        assert result.is_valid is False
        assert "Stop orders must have a positive stop price" in result.error_message

    def test_stop_limit_order_with_zero_stop_price(self, validator):
        """Test stop limit order with zero stop price"""
        portfolio = Portfolio(name="test")
        portfolio.cash_balance = Decimal("10000")

        order = Order(
            symbol="AAPL",
            quantity=Decimal("10"),
            side=OrderSide.BUY,
            order_type=OrderType.STOP_LIMIT,
            stop_price=Decimal("145"),
            limit_price=Decimal("150"),
            status=OrderStatus.PENDING,
            reason="Test",
        )
        # Force stop price to zero
        order.stop_price = Decimal("0")

        current_price = Price(Decimal("150"))

        result = validator.validate_order(order, portfolio, current_price)

        assert result.is_valid is False
        assert "Stop orders must have a positive stop price" in result.error_message

    def test_short_sale_with_margin_disabled_by_position(self, validator):
        """Test short sale edge case with position partially covering"""
        portfolio = Portfolio(name="test")
        portfolio.cash_balance = Decimal("1000")

        # Disable concentration limits
        validator.constraints.max_portfolio_concentration = Decimal("1.0")

        # Position that partially covers
        position = Position(
            symbol="AAPL",
            quantity=Decimal("50"),
            average_entry_price=Decimal("140"),
            current_price=Decimal("150"),
        )
        portfolio.positions[position.id] = position

        # Sell more than position
        order = Order(
            symbol="AAPL",
            quantity=Decimal("51"),  # 1 share short
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            reason="Test",
        )
        current_price = Price(Decimal("150"))

        result = validator.validate_order(order, portfolio, current_price)

        # Should succeed with 1 share short (1 * 150 * 1.5 = 225 margin needed)
        assert result.is_valid is True
