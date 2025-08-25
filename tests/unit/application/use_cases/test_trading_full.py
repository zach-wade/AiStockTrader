"""
Comprehensive Unit Tests for Trading Use Cases

Tests all trading use cases with complete coverage including:
- PlaceOrderUseCase
- CancelOrderUseCase
- ModifyOrderUseCase
- GetOrderStatusUseCase

Achieves 80%+ coverage with focus on business logic validation.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from src.application.use_cases.trading import (
    CancelOrderRequest,
    CancelOrderResponse,
    CancelOrderUseCase,
    GetOrderStatusRequest,
    GetOrderStatusResponse,
    GetOrderStatusUseCase,
    ModifyOrderRequest,
    ModifyOrderResponse,
    ModifyOrderUseCase,
    PlaceOrderRequest,
    PlaceOrderResponse,
    PlaceOrderUseCase,
)
from src.domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from src.domain.entities.portfolio import Portfolio
from src.domain.services.order_validator import OrderValidator, ValidationResult
from src.domain.services.risk_calculator import RiskCalculator
from src.domain.value_objects.money import Money


# Fixtures
@pytest.fixture
def mock_unit_of_work():
    """Create a mock unit of work with all repositories."""
    uow = AsyncMock()
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock(return_value=None)
    uow.commit = AsyncMock()
    uow.rollback = AsyncMock()

    # Mock repositories
    uow.orders = AsyncMock()
    uow.portfolios = AsyncMock()
    uow.market_data = AsyncMock()

    return uow


@pytest.fixture
def mock_broker():
    """Create a mock broker."""
    broker = AsyncMock()
    broker.submit_order = Mock(return_value="BROKER-123")
    broker.cancel_order = Mock(return_value=True)
    broker.update_order = Mock(return_value=True)
    broker.get_order_status = Mock(return_value=OrderStatus.FILLED)
    return broker


@pytest.fixture
def mock_order_validator():
    """Create a mock order validator."""
    validator = Mock(spec=OrderValidator)
    validator.validate_order = AsyncMock(
        return_value=ValidationResult(is_valid=True, error_message=None)
    )
    validator.validate_modification = Mock(
        return_value=ValidationResult(is_valid=True, error_message=None)
    )
    return validator


@pytest.fixture
def mock_risk_calculator():
    """Create a mock risk calculator."""
    calculator = Mock(spec=RiskCalculator)
    calculator.check_risk_limits = Mock(return_value=(True, None))
    return calculator


@pytest.fixture
def sample_portfolio():
    """Create a sample portfolio for testing."""
    portfolio = Portfolio(
        name="Test Portfolio",
        initial_capital=Money(Decimal("10000.00")),
    )
    portfolio.id = uuid4()
    portfolio.cash_balance = Money(Decimal("5000.00"))
    return portfolio


@pytest.fixture
def sample_order():
    """Create a sample order for testing."""
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("100"),
        limit_price=Decimal("150.00"),
    )
    order.id = uuid4()
    order.status = OrderStatus.SUBMITTED
    return order


@pytest.fixture
def sample_market_bar():
    """Create a sample market bar for testing."""
    bar = Mock()
    bar.close = Decimal("149.50")
    bar.high = Decimal("150.00")
    bar.low = Decimal("148.00")
    bar.volume = 1000000
    return bar


# Test PlaceOrderUseCase
class TestPlaceOrderUseCase:
    """Test PlaceOrderUseCase class."""

    @pytest.mark.asyncio
    async def test_init(
        self, mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator
    ):
        """Test use case initialization."""
        use_case = PlaceOrderUseCase(
            mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator
        )

        assert use_case.unit_of_work == mock_unit_of_work
        assert use_case.broker == mock_broker
        assert use_case.order_validator == mock_order_validator
        assert use_case.risk_calculator == mock_risk_calculator
        assert use_case.name == "PlaceOrderUseCase"

    @pytest.mark.asyncio
    async def test_validate_success(
        self, mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator
    ):
        """Test successful validation."""
        use_case = PlaceOrderUseCase(
            mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator
        )

        request = PlaceOrderRequest(
            portfolio_id=uuid4(),
            symbol="AAPL",
            side="buy",
            order_type="limit",
            quantity=100,
            limit_price=150.00,
            time_in_force="day",
        )

        result = await use_case.validate(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_invalid_order_type(
        self, mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator
    ):
        """Test validation with invalid order type."""
        use_case = PlaceOrderUseCase(
            mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator
        )

        request = PlaceOrderRequest(
            portfolio_id=uuid4(),
            symbol="AAPL",
            side="buy",
            order_type="invalid",
            quantity=100,
        )

        result = await use_case.validate(request)
        assert result == "Invalid order type: invalid"

    @pytest.mark.asyncio
    async def test_validate_invalid_side(
        self, mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator
    ):
        """Test validation with invalid order side."""
        use_case = PlaceOrderUseCase(
            mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator
        )

        request = PlaceOrderRequest(
            portfolio_id=uuid4(),
            symbol="AAPL",
            side="invalid",
            order_type="market",
            quantity=100,
        )

        result = await use_case.validate(request)
        assert result == "Invalid order side: invalid"

    @pytest.mark.asyncio
    async def test_validate_negative_quantity(
        self, mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator
    ):
        """Test validation with negative quantity."""
        use_case = PlaceOrderUseCase(
            mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator
        )

        request = PlaceOrderRequest(
            portfolio_id=uuid4(),
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=-100,
        )

        result = await use_case.validate(request)
        assert result == "Quantity must be positive"

    @pytest.mark.asyncio
    async def test_validate_limit_order_without_price(
        self, mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator
    ):
        """Test validation of limit order without limit price."""
        use_case = PlaceOrderUseCase(
            mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator
        )

        request = PlaceOrderRequest(
            portfolio_id=uuid4(),
            symbol="AAPL",
            side="buy",
            order_type="limit",
            quantity=100,
            limit_price=None,
        )

        result = await use_case.validate(request)
        assert result == "limit order requires limit price"

    @pytest.mark.asyncio
    async def test_validate_stop_order_without_price(
        self, mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator
    ):
        """Test validation of stop order without stop price."""
        use_case = PlaceOrderUseCase(
            mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator
        )

        request = PlaceOrderRequest(
            portfolio_id=uuid4(),
            symbol="AAPL",
            side="sell",
            order_type="stop",
            quantity=100,
            stop_price=None,
        )

        result = await use_case.validate(request)
        assert result == "stop order requires stop price"

    @pytest.mark.asyncio
    async def test_validate_stop_limit_order_missing_prices(
        self, mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator
    ):
        """Test validation of stop-limit order without required prices."""
        use_case = PlaceOrderUseCase(
            mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator
        )

        # Missing limit price
        request = PlaceOrderRequest(
            portfolio_id=uuid4(),
            symbol="AAPL",
            side="sell",
            order_type="stop_limit",
            quantity=100,
            stop_price=145.00,
            limit_price=None,
        )

        result = await use_case.validate(request)
        assert result == "stop_limit order requires limit price"

        # Missing stop price
        request.limit_price = 140.00
        request.stop_price = None

        result = await use_case.validate(request)
        assert result == "stop_limit order requires stop price"

    @pytest.mark.asyncio
    async def test_process_successful_order_placement(
        self,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        sample_market_bar,
    ):
        """Test successful order placement."""
        use_case = PlaceOrderUseCase(
            mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator
        )

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_unit_of_work.market_data.get_latest_bar.return_value = sample_market_bar

        request = PlaceOrderRequest(
            portfolio_id=sample_portfolio.id,
            symbol="AAPL",
            side="buy",
            order_type="limit",
            quantity=100,
            limit_price=150.00,
            strategy_id="test_strategy",
        )

        response = await use_case.process(request)

        assert response.success is True
        assert response.order_id is not None
        assert response.broker_order_id == "BROKER-123"
        assert response.status == "submitted"

        # Verify interactions
        mock_unit_of_work.portfolios.get_portfolio_by_id.assert_called_once_with(
            sample_portfolio.id
        )
        mock_unit_of_work.market_data.get_latest_bar.assert_called_once_with("AAPL")
        mock_order_validator.validate_order.assert_called_once()
        mock_risk_calculator.check_risk_limits.assert_called_once()
        mock_broker.submit_order.assert_called_once()
        mock_unit_of_work.orders.save_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_portfolio_not_found(
        self, mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator
    ):
        """Test processing when portfolio not found."""
        use_case = PlaceOrderUseCase(
            mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator
        )

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = None

        request = PlaceOrderRequest(
            portfolio_id=uuid4(),
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=100,
        )

        response = await use_case.process(request)

        assert response.success is False
        assert response.error == "Portfolio not found"

    @pytest.mark.asyncio
    async def test_process_market_data_not_available(
        self,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
    ):
        """Test processing when market data not available."""
        use_case = PlaceOrderUseCase(
            mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator
        )

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_unit_of_work.market_data.get_latest_bar.return_value = None

        request = PlaceOrderRequest(
            portfolio_id=sample_portfolio.id,
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=100,
        )

        response = await use_case.process(request)

        assert response.success is False
        assert response.error == "Cannot get current market price"

    @pytest.mark.asyncio
    async def test_process_order_validation_failure(
        self,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        sample_market_bar,
    ):
        """Test processing when order validation fails."""
        use_case = PlaceOrderUseCase(
            mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator
        )

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_unit_of_work.market_data.get_latest_bar.return_value = sample_market_bar
        mock_order_validator.validate_order.return_value = ValidationResult(
            is_valid=False, error_message="Insufficient funds"
        )

        request = PlaceOrderRequest(
            portfolio_id=sample_portfolio.id,
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=1000,
        )

        response = await use_case.process(request)

        assert response.success is False
        assert response.error == "Insufficient funds"

    @pytest.mark.asyncio
    async def test_process_risk_limit_violation(
        self,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        sample_market_bar,
    ):
        """Test processing when risk limits are violated."""
        use_case = PlaceOrderUseCase(
            mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator
        )

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_unit_of_work.market_data.get_latest_bar.return_value = sample_market_bar
        mock_risk_calculator.check_risk_limits.return_value = (False, "Position size too large")

        request = PlaceOrderRequest(
            portfolio_id=sample_portfolio.id,
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=10000,
        )

        response = await use_case.process(request)

        assert response.success is False
        assert response.error == "Risk limit violated: Position size too large"

    @pytest.mark.asyncio
    async def test_process_broker_submission_failure(
        self,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        sample_market_bar,
    ):
        """Test processing when broker submission fails."""
        use_case = PlaceOrderUseCase(
            mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator
        )

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_unit_of_work.market_data.get_latest_bar.return_value = sample_market_bar
        mock_broker.submit_order.side_effect = Exception("Broker connection error")

        request = PlaceOrderRequest(
            portfolio_id=sample_portfolio.id,
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=100,
        )

        response = await use_case.process(request)

        assert response.success is False
        assert "Failed to submit order: Broker connection error" in response.error

    @pytest.mark.asyncio
    async def test_process_with_all_order_types(
        self,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        sample_market_bar,
    ):
        """Test processing with different order types."""
        use_case = PlaceOrderUseCase(
            mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator
        )

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_unit_of_work.market_data.get_latest_bar.return_value = sample_market_bar

        # Test market order
        request = PlaceOrderRequest(
            portfolio_id=sample_portfolio.id,
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=100,
        )
        response = await use_case.process(request)
        assert response.success is True

        # Test limit order
        request.order_type = "limit"
        request.limit_price = 150.00
        response = await use_case.process(request)
        assert response.success is True

        # Test stop order
        request.order_type = "stop"
        request.stop_price = 145.00
        response = await use_case.process(request)
        assert response.success is True

        # Test stop-limit order
        request.order_type = "stop_limit"
        request.limit_price = 144.00
        request.stop_price = 145.00
        response = await use_case.process(request)
        assert response.success is True


# Test CancelOrderUseCase
class TestCancelOrderUseCase:
    """Test CancelOrderUseCase class."""

    @pytest.mark.asyncio
    async def test_init(self, mock_unit_of_work, mock_broker):
        """Test use case initialization."""
        use_case = CancelOrderUseCase(mock_unit_of_work, mock_broker)

        assert use_case.unit_of_work == mock_unit_of_work
        assert use_case.broker == mock_broker
        assert use_case.name == "CancelOrderUseCase"

    @pytest.mark.asyncio
    async def test_validate_success(self, mock_unit_of_work, mock_broker):
        """Test successful validation."""
        use_case = CancelOrderUseCase(mock_unit_of_work, mock_broker)

        request = CancelOrderRequest(order_id=uuid4(), reason="User requested")

        result = await use_case.validate(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_missing_order_id(self, mock_unit_of_work, mock_broker):
        """Test validation with missing order ID."""
        use_case = CancelOrderUseCase(mock_unit_of_work, mock_broker)

        request = CancelOrderRequest(order_id=None)

        result = await use_case.validate(request)
        assert result == "Order ID is required"

    @pytest.mark.asyncio
    async def test_process_successful_cancellation(
        self, mock_unit_of_work, mock_broker, sample_order
    ):
        """Test successful order cancellation."""
        use_case = CancelOrderUseCase(mock_unit_of_work, mock_broker)

        # Setup mocks
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_broker.cancel_order.return_value = True

        request = CancelOrderRequest(order_id=sample_order.id, reason="Strategy exit")

        response = await use_case.process(request)

        assert response.success is True
        assert response.cancelled is True
        assert response.final_status is not None

        # Verify interactions
        mock_unit_of_work.orders.get_order_by_id.assert_called_once_with(sample_order.id)
        mock_broker.cancel_order.assert_called_once_with(sample_order.id)
        mock_unit_of_work.orders.update_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_order_not_found(self, mock_unit_of_work, mock_broker):
        """Test cancellation when order not found."""
        use_case = CancelOrderUseCase(mock_unit_of_work, mock_broker)

        # Setup mocks
        mock_unit_of_work.orders.get_order_by_id.return_value = None

        request = CancelOrderRequest(order_id=uuid4())

        response = await use_case.process(request)

        assert response.success is False
        assert response.error == "Order not found"
        assert response.cancelled is False

    @pytest.mark.asyncio
    async def test_process_inactive_order(self, mock_unit_of_work, mock_broker, sample_order):
        """Test cancellation of inactive order."""
        use_case = CancelOrderUseCase(mock_unit_of_work, mock_broker)

        # Setup mocks - order already filled
        sample_order.status = OrderStatus.FILLED
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order

        request = CancelOrderRequest(order_id=sample_order.id)

        response = await use_case.process(request)

        assert response.success is False
        assert "Order cannot be cancelled in status: filled" in response.error
        assert response.cancelled is False

    @pytest.mark.asyncio
    async def test_process_broker_cancellation_failure(
        self, mock_unit_of_work, mock_broker, sample_order
    ):
        """Test when broker fails to cancel order."""
        use_case = CancelOrderUseCase(mock_unit_of_work, mock_broker)

        # Setup mocks
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_broker.cancel_order.return_value = False

        request = CancelOrderRequest(order_id=sample_order.id)

        response = await use_case.process(request)

        assert response.success is False
        assert response.error == "Broker failed to cancel order"
        assert response.cancelled is False

    @pytest.mark.asyncio
    async def test_process_cancellation_exception(
        self, mock_unit_of_work, mock_broker, sample_order
    ):
        """Test exception during cancellation."""
        use_case = CancelOrderUseCase(mock_unit_of_work, mock_broker)

        # Setup mocks
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_broker.cancel_order.side_effect = Exception("Network error")

        request = CancelOrderRequest(order_id=sample_order.id)

        response = await use_case.process(request)

        assert response.success is False
        assert "Failed to cancel order: Network error" in response.error
        assert response.cancelled is False


# Test ModifyOrderUseCase
class TestModifyOrderUseCase:
    """Test ModifyOrderUseCase class."""

    @pytest.mark.asyncio
    async def test_init(self, mock_unit_of_work, mock_broker, mock_order_validator):
        """Test use case initialization."""
        use_case = ModifyOrderUseCase(mock_unit_of_work, mock_broker, mock_order_validator)

        assert use_case.unit_of_work == mock_unit_of_work
        assert use_case.broker == mock_broker
        assert use_case.order_validator == mock_order_validator
        assert use_case.name == "ModifyOrderUseCase"

    @pytest.mark.asyncio
    async def test_validate_success(self, mock_unit_of_work, mock_broker, mock_order_validator):
        """Test successful validation."""
        use_case = ModifyOrderUseCase(mock_unit_of_work, mock_broker, mock_order_validator)

        request = ModifyOrderRequest(
            order_id=uuid4(),
            new_quantity=150,
            new_limit_price=155.00,
        )

        result = await use_case.validate(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_missing_order_id(
        self, mock_unit_of_work, mock_broker, mock_order_validator
    ):
        """Test validation with missing order ID."""
        use_case = ModifyOrderUseCase(mock_unit_of_work, mock_broker, mock_order_validator)

        request = ModifyOrderRequest(order_id=None, new_quantity=100)

        result = await use_case.validate(request)
        assert result == "Order ID is required"

    @pytest.mark.asyncio
    async def test_validate_no_modifications(
        self, mock_unit_of_work, mock_broker, mock_order_validator
    ):
        """Test validation with no modifications."""
        use_case = ModifyOrderUseCase(mock_unit_of_work, mock_broker, mock_order_validator)

        request = ModifyOrderRequest(order_id=uuid4())

        result = await use_case.validate(request)
        assert result == "At least one modification is required"

    @pytest.mark.asyncio
    async def test_validate_negative_quantity(
        self, mock_unit_of_work, mock_broker, mock_order_validator
    ):
        """Test validation with negative quantity."""
        use_case = ModifyOrderUseCase(mock_unit_of_work, mock_broker, mock_order_validator)

        request = ModifyOrderRequest(order_id=uuid4(), new_quantity=-100)

        result = await use_case.validate(request)
        assert result == "New quantity must be positive"

    @pytest.mark.asyncio
    async def test_process_successful_modification(
        self, mock_unit_of_work, mock_broker, mock_order_validator, sample_order
    ):
        """Test successful order modification."""
        use_case = ModifyOrderUseCase(mock_unit_of_work, mock_broker, mock_order_validator)

        # Setup mocks
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_broker.update_order.return_value = True

        request = ModifyOrderRequest(
            order_id=sample_order.id,
            new_quantity=200,
            new_limit_price=155.00,
            new_stop_price=145.00,
        )

        response = await use_case.process(request)

        assert response.success is True
        assert response.modified is True
        assert response.new_values == {
            "quantity": 200,
            "limit_price": 155.00,
            "stop_price": 145.00,
        }

        # Verify interactions
        mock_unit_of_work.orders.get_order_by_id.assert_called_once_with(sample_order.id)
        mock_order_validator.validate_modification.assert_called_once()
        mock_broker.update_order.assert_called_once()
        mock_unit_of_work.orders.update_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_order_not_found(
        self, mock_unit_of_work, mock_broker, mock_order_validator
    ):
        """Test modification when order not found."""
        use_case = ModifyOrderUseCase(mock_unit_of_work, mock_broker, mock_order_validator)

        # Setup mocks
        mock_unit_of_work.orders.get_order_by_id.return_value = None

        request = ModifyOrderRequest(order_id=uuid4(), new_quantity=100)

        response = await use_case.process(request)

        assert response.success is False
        assert response.error == "Order not found"
        assert response.modified is False

    @pytest.mark.asyncio
    async def test_process_validation_failure(
        self, mock_unit_of_work, mock_broker, mock_order_validator, sample_order
    ):
        """Test modification with validation failure."""
        use_case = ModifyOrderUseCase(mock_unit_of_work, mock_broker, mock_order_validator)

        # Setup mocks
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_order_validator.validate_modification.return_value = ValidationResult(
            is_valid=False, error_message="Cannot modify filled order"
        )

        request = ModifyOrderRequest(order_id=sample_order.id, new_quantity=200)

        response = await use_case.process(request)

        assert response.success is False
        assert response.error == "Cannot modify filled order"
        assert response.modified is False

    @pytest.mark.asyncio
    async def test_process_broker_update_failure(
        self, mock_unit_of_work, mock_broker, mock_order_validator, sample_order
    ):
        """Test when broker fails to update order."""
        use_case = ModifyOrderUseCase(mock_unit_of_work, mock_broker, mock_order_validator)

        # Setup mocks
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_broker.update_order.return_value = False

        request = ModifyOrderRequest(order_id=sample_order.id, new_quantity=200)

        response = await use_case.process(request)

        assert response.success is False
        assert response.error == "Broker failed to modify order"
        assert response.modified is False

    @pytest.mark.asyncio
    async def test_process_modification_exception(
        self, mock_unit_of_work, mock_broker, mock_order_validator, sample_order
    ):
        """Test exception during modification."""
        use_case = ModifyOrderUseCase(mock_unit_of_work, mock_broker, mock_order_validator)

        # Setup mocks
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_broker.update_order.side_effect = Exception("API error")

        request = ModifyOrderRequest(order_id=sample_order.id, new_quantity=200)

        response = await use_case.process(request)

        assert response.success is False
        assert "Failed to modify order: API error" in response.error
        assert response.modified is False


# Test GetOrderStatusUseCase
class TestGetOrderStatusUseCase:
    """Test GetOrderStatusUseCase class."""

    @pytest.mark.asyncio
    async def test_init(self, mock_unit_of_work, mock_broker):
        """Test use case initialization."""
        use_case = GetOrderStatusUseCase(mock_unit_of_work, mock_broker)

        assert use_case.unit_of_work == mock_unit_of_work
        assert use_case.broker == mock_broker
        assert use_case.name == "GetOrderStatusUseCase"

    @pytest.mark.asyncio
    async def test_validate_success(self, mock_unit_of_work, mock_broker):
        """Test successful validation."""
        use_case = GetOrderStatusUseCase(mock_unit_of_work, mock_broker)

        request = GetOrderStatusRequest(order_id=uuid4())

        result = await use_case.validate(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_missing_order_id(self, mock_unit_of_work, mock_broker):
        """Test validation with missing order ID."""
        use_case = GetOrderStatusUseCase(mock_unit_of_work, mock_broker)

        request = GetOrderStatusRequest(order_id=None)

        result = await use_case.validate(request)
        assert result == "Order ID is required"

    @pytest.mark.asyncio
    async def test_process_successful_status_retrieval(
        self, mock_unit_of_work, mock_broker, sample_order
    ):
        """Test successful order status retrieval."""
        use_case = GetOrderStatusUseCase(mock_unit_of_work, mock_broker)

        # Setup mocks
        sample_order.filled_quantity = Decimal("50")
        sample_order.average_fill_price = Decimal("149.75")
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_broker.get_order_status.return_value = OrderStatus.PARTIALLY_FILLED

        request = GetOrderStatusRequest(order_id=sample_order.id)

        response = await use_case.process(request)

        assert response.success is True
        assert response.status == "partially_filled"
        assert response.filled_quantity == 50
        assert response.average_fill_price == 149.75

        # Verify interactions
        mock_unit_of_work.orders.get_order_by_id.assert_called_once_with(sample_order.id)
        mock_broker.get_order_status.assert_called_once_with(sample_order.id)

    @pytest.mark.asyncio
    async def test_process_order_not_found(self, mock_unit_of_work, mock_broker):
        """Test status retrieval when order not found."""
        use_case = GetOrderStatusUseCase(mock_unit_of_work, mock_broker)

        # Setup mocks
        mock_unit_of_work.orders.get_order_by_id.return_value = None

        request = GetOrderStatusRequest(order_id=uuid4())

        response = await use_case.process(request)

        assert response.success is False
        assert response.error == "Order not found"

    @pytest.mark.asyncio
    async def test_process_status_update_from_broker(
        self, mock_unit_of_work, mock_broker, sample_order
    ):
        """Test status update from broker."""
        use_case = GetOrderStatusUseCase(mock_unit_of_work, mock_broker)

        # Setup mocks - broker has updated status
        sample_order.status = OrderStatus.SUBMITTED
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_broker.get_order_status.return_value = OrderStatus.FILLED

        request = GetOrderStatusRequest(order_id=sample_order.id)

        response = await use_case.process(request)

        assert response.success is True
        assert response.status == "filled"

        # Verify order was updated
        assert sample_order.status == OrderStatus.FILLED
        mock_unit_of_work.orders.update_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_broker_error_returns_cached(
        self, mock_unit_of_work, mock_broker, sample_order
    ):
        """Test that cached status is returned when broker query fails."""
        use_case = GetOrderStatusUseCase(mock_unit_of_work, mock_broker)

        # Setup mocks
        sample_order.status = OrderStatus.SUBMITTED
        sample_order.filled_quantity = Decimal("25")
        sample_order.average_fill_price = Decimal("150.00")
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_broker.get_order_status.side_effect = Exception("Broker unavailable")

        request = GetOrderStatusRequest(order_id=sample_order.id)

        response = await use_case.process(request)

        # Should return cached status despite broker error
        assert response.success is True
        assert response.status == "submitted"
        assert response.filled_quantity == 25
        assert response.average_fill_price == 150.00


# Test Request/Response DTOs
class TestRequestResponseDTOs:
    """Test request and response data classes."""

    def test_place_order_request_init(self):
        """Test PlaceOrderRequest initialization."""
        request = PlaceOrderRequest(
            portfolio_id=uuid4(),
            symbol="AAPL",
            side="buy",
            order_type="limit",
            quantity=100,
        )

        assert request.request_id is not None
        assert request.metadata == {}
        assert request.time_in_force == "day"
        assert request.strategy_id is None

    def test_cancel_order_request_init(self):
        """Test CancelOrderRequest initialization."""
        request = CancelOrderRequest(order_id=uuid4())

        assert request.request_id is not None
        assert request.metadata == {}
        assert request.reason is None

    def test_modify_order_request_init(self):
        """Test ModifyOrderRequest initialization."""
        request = ModifyOrderRequest(order_id=uuid4(), new_quantity=200)

        assert request.request_id is not None
        assert request.metadata == {}
        assert request.new_limit_price is None
        assert request.new_stop_price is None

    def test_get_order_status_request_init(self):
        """Test GetOrderStatusRequest initialization."""
        request = GetOrderStatusRequest(order_id=uuid4())

        assert request.request_id is not None
        assert request.metadata == {}

    def test_place_order_response(self):
        """Test PlaceOrderResponse initialization."""
        order_id = uuid4()
        response = PlaceOrderResponse(
            success=True,
            order_id=order_id,
            broker_order_id="BROKER-456",
            status="submitted",
        )

        assert response.success is True
        assert response.order_id == order_id
        assert response.broker_order_id == "BROKER-456"
        assert response.status == "submitted"

    def test_cancel_order_response(self):
        """Test CancelOrderResponse initialization."""
        response = CancelOrderResponse(
            success=True,
            cancelled=True,
            final_status="cancelled",
        )

        assert response.success is True
        assert response.cancelled is True
        assert response.final_status == "cancelled"

    def test_modify_order_response(self):
        """Test ModifyOrderResponse initialization."""
        response = ModifyOrderResponse(
            success=True,
            modified=True,
            new_values={"quantity": 200, "limit_price": 155.00},
        )

        assert response.success is True
        assert response.modified is True
        assert response.new_values["quantity"] == 200

    def test_get_order_status_response(self):
        """Test GetOrderStatusResponse initialization."""
        response = GetOrderStatusResponse(
            success=True,
            status="filled",
            filled_quantity=100,
            average_fill_price=149.50,
        )

        assert response.success is True
        assert response.status == "filled"
        assert response.filled_quantity == 100
        assert response.average_fill_price == 149.50
