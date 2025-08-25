"""
Comprehensive tests for trading use cases with full coverage.

Tests all trading-related use cases including:
- PlaceOrderUseCase
- CancelOrderUseCase
- ModifyOrderUseCase
- GetOrderStatusUseCase

Covers all scenarios including success, failure, validation, and edge cases.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from src.application.interfaces.broker import IBroker
from src.application.interfaces.repositories import (
    IMarketDataRepository,
    IOrderRepository,
    IPortfolioRepository,
)
from src.application.interfaces.unit_of_work import IUnitOfWork
from src.application.use_cases.trading import (
    CancelOrderRequest,
    CancelOrderUseCase,
    GetOrderStatusRequest,
    GetOrderStatusResponse,
    GetOrderStatusUseCase,
    ModifyOrderRequest,
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


@pytest.fixture
def mock_unit_of_work():
    """Create a mock unit of work."""
    uow = Mock(spec=IUnitOfWork)

    # Setup repository mocks
    uow.orders = AsyncMock(spec=IOrderRepository)
    uow.portfolios = AsyncMock(spec=IPortfolioRepository)
    uow.positions = AsyncMock()
    uow.market_data = AsyncMock(spec=IMarketDataRepository)

    # Setup transaction methods
    uow.begin = AsyncMock()
    uow.commit = AsyncMock()
    uow.rollback = AsyncMock()

    # Make it work as async context manager
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock(return_value=None)

    return uow


@pytest.fixture
def mock_broker():
    """Create a mock broker."""
    broker = Mock(spec=IBroker)
    broker.submit_order = Mock(return_value="BROKER123")
    broker.cancel_order = Mock(return_value=True)
    broker.update_order = Mock()
    broker.get_order_status = Mock(return_value=OrderStatus.PENDING)
    return broker


@pytest.fixture
def mock_order_validator():
    """Create a mock order validator."""
    validator = Mock(spec=OrderValidator)
    validator.validate_order = AsyncMock(return_value=ValidationResult(is_valid=True))
    validator.validate_modification = Mock(return_value=ValidationResult(is_valid=True))
    return validator


@pytest.fixture
def mock_risk_calculator():
    """Create a mock risk calculator."""
    calculator = Mock(spec=RiskCalculator)
    calculator.check_risk_limits = Mock(return_value=(True, None))
    calculator.calculate_portfolio_var = Mock(return_value=Money(Decimal("1000")))
    calculator.calculate_sharpe_ratio = Mock(return_value=Decimal("1.5"))
    calculator.calculate_max_drawdown = Mock(return_value=Decimal("0.1"))
    return calculator


@pytest.fixture
def sample_portfolio():
    """Create a sample portfolio for testing."""
    portfolio = Portfolio(
        id=uuid4(),
        name="Test Portfolio",
        initial_capital=Decimal("100000"),
        cash_balance=Decimal("50000"),
    )
    portfolio.max_position_size = Decimal("10000")
    portfolio.max_positions = 10
    portfolio.max_leverage = Decimal("2.0")
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
    order.portfolio_id = uuid4()
    return order


@pytest.fixture
def sample_market_bar():
    """Create a sample market bar."""
    bar = Mock()
    bar.close = Decimal("150.00")
    bar.high = Decimal("151.00")
    bar.low = Decimal("149.00")
    bar.volume = 1000000
    return bar


class TestPlaceOrderUseCase:
    """Test PlaceOrderUseCase."""

    @pytest.mark.asyncio
    async def test_place_market_order_success(
        self,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        sample_market_bar,
    ):
        """Test successful market order placement."""
        # Setup
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_unit_of_work.market_data.get_latest_bar.return_value = sample_market_bar

        use_case = PlaceOrderUseCase(
            unit_of_work=mock_unit_of_work,
            broker=mock_broker,
            order_validator=mock_order_validator,
            risk_calculator=mock_risk_calculator,
        )

        request = PlaceOrderRequest(
            portfolio_id=sample_portfolio.id,
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=100,
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.broker_order_id == "BROKER123"
        assert response.status == OrderStatus.SUBMITTED.value
        assert response.order_id is not None

        # Verify interactions
        mock_broker.submit_order.assert_called_once()
        mock_unit_of_work.orders.save_order.assert_called_once()
        mock_unit_of_work.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_place_limit_order_success(
        self,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        sample_market_bar,
    ):
        """Test successful limit order placement."""
        # Setup
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_unit_of_work.market_data.get_latest_bar.return_value = sample_market_bar

        use_case = PlaceOrderUseCase(
            unit_of_work=mock_unit_of_work,
            broker=mock_broker,
            order_validator=mock_order_validator,
            risk_calculator=mock_risk_calculator,
        )

        request = PlaceOrderRequest(
            portfolio_id=sample_portfolio.id,
            symbol="AAPL",
            side="buy",
            order_type="limit",
            quantity=100,
            limit_price=149.50,
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.broker_order_id == "BROKER123"

        # Verify order was created with limit price
        call_args = mock_broker.submit_order.call_args[0][0]
        assert call_args.limit_price == Decimal("149.50")

    @pytest.mark.asyncio
    async def test_place_stop_limit_order_success(
        self,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        sample_market_bar,
    ):
        """Test successful stop-limit order placement."""
        # Setup
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_unit_of_work.market_data.get_latest_bar.return_value = sample_market_bar

        use_case = PlaceOrderUseCase(
            unit_of_work=mock_unit_of_work,
            broker=mock_broker,
            order_validator=mock_order_validator,
            risk_calculator=mock_risk_calculator,
        )

        request = PlaceOrderRequest(
            portfolio_id=sample_portfolio.id,
            symbol="AAPL",
            side="sell",
            order_type="stop_limit",
            quantity=100,
            limit_price=148.00,
            stop_price=149.00,
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True

        # Verify order was created with both prices
        call_args = mock_broker.submit_order.call_args[0][0]
        assert call_args.limit_price == Decimal("148.00")
        assert call_args.stop_price == Decimal("149.00")

    @pytest.mark.asyncio
    async def test_place_order_portfolio_not_found(
        self, mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator
    ):
        """Test order placement when portfolio doesn't exist."""
        # Setup
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = None

        use_case = PlaceOrderUseCase(
            unit_of_work=mock_unit_of_work,
            broker=mock_broker,
            order_validator=mock_order_validator,
            risk_calculator=mock_risk_calculator,
        )

        request = PlaceOrderRequest(
            portfolio_id=uuid4(), symbol="AAPL", side="buy", order_type="market", quantity=100
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Portfolio not found" in response.error
        mock_broker.submit_order.assert_not_called()
        mock_unit_of_work.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_place_order_no_market_data(
        self,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
    ):
        """Test order placement when market data is unavailable."""
        # Setup
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_unit_of_work.market_data.get_latest_bar.return_value = None

        use_case = PlaceOrderUseCase(
            unit_of_work=mock_unit_of_work,
            broker=mock_broker,
            order_validator=mock_order_validator,
            risk_calculator=mock_risk_calculator,
        )

        request = PlaceOrderRequest(
            portfolio_id=sample_portfolio.id,
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=100,
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Cannot get current market price" in response.error
        mock_broker.submit_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_place_order_validation_failure(
        self,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        sample_market_bar,
    ):
        """Test order placement when validation fails."""
        # Setup
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_unit_of_work.market_data.get_latest_bar.return_value = sample_market_bar
        mock_order_validator.validate_order.return_value = ValidationResult(
            is_valid=False, error_message="Insufficient funds"
        )

        use_case = PlaceOrderUseCase(
            unit_of_work=mock_unit_of_work,
            broker=mock_broker,
            order_validator=mock_order_validator,
            risk_calculator=mock_risk_calculator,
        )

        request = PlaceOrderRequest(
            portfolio_id=sample_portfolio.id,
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=100,
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Insufficient funds" in response.error
        mock_broker.submit_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_place_order_risk_limit_violation(
        self,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        sample_market_bar,
    ):
        """Test order placement when risk limits are violated."""
        # Setup
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_unit_of_work.market_data.get_latest_bar.return_value = sample_market_bar
        mock_risk_calculator.check_risk_limits.return_value = (
            False,
            "Position size exceeds maximum",
        )

        use_case = PlaceOrderUseCase(
            unit_of_work=mock_unit_of_work,
            broker=mock_broker,
            order_validator=mock_order_validator,
            risk_calculator=mock_risk_calculator,
        )

        request = PlaceOrderRequest(
            portfolio_id=sample_portfolio.id,
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=100,
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Risk limit violated" in response.error
        assert "Position size exceeds maximum" in response.error
        mock_broker.submit_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_place_order_broker_submission_failure(
        self,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        sample_market_bar,
    ):
        """Test order placement when broker submission fails."""
        # Setup
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_unit_of_work.market_data.get_latest_bar.return_value = sample_market_bar
        mock_broker.submit_order.side_effect = Exception("Broker connection failed")

        use_case = PlaceOrderUseCase(
            unit_of_work=mock_unit_of_work,
            broker=mock_broker,
            order_validator=mock_order_validator,
            risk_calculator=mock_risk_calculator,
        )

        request = PlaceOrderRequest(
            portfolio_id=sample_portfolio.id,
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=100,
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Failed to submit order" in response.error
        assert "Broker connection failed" in response.error
        mock_unit_of_work.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_invalid_order_type(
        self, mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator
    ):
        """Test validation with invalid order type."""
        use_case = PlaceOrderUseCase(
            unit_of_work=mock_unit_of_work,
            broker=mock_broker,
            order_validator=mock_order_validator,
            risk_calculator=mock_risk_calculator,
        )

        request = PlaceOrderRequest(
            portfolio_id=uuid4(), symbol="AAPL", side="buy", order_type="invalid_type", quantity=100
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Invalid order type" in response.error

    @pytest.mark.asyncio
    async def test_validate_invalid_side(
        self, mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator
    ):
        """Test validation with invalid order side."""
        use_case = PlaceOrderUseCase(
            unit_of_work=mock_unit_of_work,
            broker=mock_broker,
            order_validator=mock_order_validator,
            risk_calculator=mock_risk_calculator,
        )

        request = PlaceOrderRequest(
            portfolio_id=uuid4(),
            symbol="AAPL",
            side="invalid_side",
            order_type="market",
            quantity=100,
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Invalid order side" in response.error

    @pytest.mark.asyncio
    async def test_validate_negative_quantity(
        self, mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator
    ):
        """Test validation with negative quantity."""
        use_case = PlaceOrderUseCase(
            unit_of_work=mock_unit_of_work,
            broker=mock_broker,
            order_validator=mock_order_validator,
            risk_calculator=mock_risk_calculator,
        )

        request = PlaceOrderRequest(
            portfolio_id=uuid4(), symbol="AAPL", side="buy", order_type="market", quantity=-100
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Quantity must be positive" in response.error

    @pytest.mark.asyncio
    async def test_validate_limit_order_without_price(
        self, mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator
    ):
        """Test validation of limit order without limit price."""
        use_case = PlaceOrderUseCase(
            unit_of_work=mock_unit_of_work,
            broker=mock_broker,
            order_validator=mock_order_validator,
            risk_calculator=mock_risk_calculator,
        )

        request = PlaceOrderRequest(
            portfolio_id=uuid4(),
            symbol="AAPL",
            side="buy",
            order_type="limit",
            quantity=100,
            limit_price=None,
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "limit order requires limit price" in response.error

    @pytest.mark.asyncio
    async def test_validate_stop_order_without_price(
        self, mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator
    ):
        """Test validation of stop order without stop price."""
        use_case = PlaceOrderUseCase(
            unit_of_work=mock_unit_of_work,
            broker=mock_broker,
            order_validator=mock_order_validator,
            risk_calculator=mock_risk_calculator,
        )

        request = PlaceOrderRequest(
            portfolio_id=uuid4(),
            symbol="AAPL",
            side="sell",
            order_type="stop",
            quantity=100,
            stop_price=None,
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "stop order requires stop price" in response.error


class TestCancelOrderUseCase:
    """Test CancelOrderUseCase."""

    @pytest.mark.asyncio
    async def test_cancel_order_success(self, mock_unit_of_work, mock_broker, sample_order):
        """Test successful order cancellation."""
        # Setup
        sample_order.status = OrderStatus.PENDING
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_broker.cancel_order.return_value = True

        use_case = CancelOrderUseCase(unit_of_work=mock_unit_of_work, broker=mock_broker)

        request = CancelOrderRequest(order_id=sample_order.id, reason="User requested")

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.cancelled is True
        assert response.final_status is not None

        # Verify interactions
        mock_broker.cancel_order.assert_called_once_with(sample_order.id)
        mock_unit_of_work.orders.update_order.assert_called_once()
        mock_unit_of_work.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_order_not_found(self, mock_unit_of_work, mock_broker):
        """Test cancellation when order doesn't exist."""
        # Setup
        mock_unit_of_work.orders.get_order_by_id.return_value = None

        use_case = CancelOrderUseCase(unit_of_work=mock_unit_of_work, broker=mock_broker)

        request = CancelOrderRequest(order_id=uuid4())

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Order not found" in response.error
        mock_broker.cancel_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_inactive_order(self, mock_unit_of_work, mock_broker, sample_order):
        """Test cancellation of already cancelled/filled order."""
        # Setup
        sample_order.status = OrderStatus.FILLED
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order

        use_case = CancelOrderUseCase(unit_of_work=mock_unit_of_work, broker=mock_broker)

        request = CancelOrderRequest(order_id=sample_order.id)

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Order cannot be cancelled" in response.error
        mock_broker.cancel_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_order_broker_failure(self, mock_unit_of_work, mock_broker, sample_order):
        """Test cancellation when broker fails to cancel."""
        # Setup
        sample_order.status = OrderStatus.PENDING
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_broker.cancel_order.return_value = False

        use_case = CancelOrderUseCase(unit_of_work=mock_unit_of_work, broker=mock_broker)

        request = CancelOrderRequest(order_id=sample_order.id)

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Broker failed to cancel order" in response.error
        mock_unit_of_work.orders.update_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_order_broker_exception(
        self, mock_unit_of_work, mock_broker, sample_order
    ):
        """Test cancellation when broker throws exception."""
        # Setup
        sample_order.status = OrderStatus.PENDING
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_broker.cancel_order.side_effect = Exception("Network error")

        use_case = CancelOrderUseCase(unit_of_work=mock_unit_of_work, broker=mock_broker)

        request = CancelOrderRequest(order_id=sample_order.id)

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Failed to cancel order" in response.error
        assert "Network error" in response.error
        mock_unit_of_work.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_missing_order_id(self, mock_unit_of_work, mock_broker):
        """Test validation when order ID is missing."""
        use_case = CancelOrderUseCase(unit_of_work=mock_unit_of_work, broker=mock_broker)

        request = CancelOrderRequest(order_id=None)

        # Execute validation directly
        error = await use_case.validate(request)

        # Assert
        assert error == "Order ID is required"


class TestModifyOrderUseCase:
    """Test ModifyOrderUseCase."""

    @pytest.mark.asyncio
    async def test_modify_order_quantity_success(
        self, mock_unit_of_work, mock_broker, mock_order_validator, sample_order
    ):
        """Test successful order quantity modification."""
        # Setup
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_broker.update_order.return_value = sample_order

        use_case = ModifyOrderUseCase(
            unit_of_work=mock_unit_of_work, broker=mock_broker, order_validator=mock_order_validator
        )

        request = ModifyOrderRequest(order_id=sample_order.id, new_quantity=200)

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.modified is True
        assert response.new_values["quantity"] == 200

        # Verify interactions
        mock_broker.update_order.assert_called_once()
        mock_unit_of_work.orders.update_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_modify_order_prices_success(
        self, mock_unit_of_work, mock_broker, mock_order_validator, sample_order
    ):
        """Test successful order price modification."""
        # Setup
        sample_order.order_type = OrderType.STOP_LIMIT
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_broker.update_order.return_value = sample_order

        use_case = ModifyOrderUseCase(
            unit_of_work=mock_unit_of_work, broker=mock_broker, order_validator=mock_order_validator
        )

        request = ModifyOrderRequest(
            order_id=sample_order.id, new_limit_price=155.00, new_stop_price=154.00
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.modified is True
        assert response.new_values["limit_price"] == 155.00
        assert response.new_values["stop_price"] == 154.00

    @pytest.mark.asyncio
    async def test_modify_order_not_found(
        self, mock_unit_of_work, mock_broker, mock_order_validator
    ):
        """Test modification when order doesn't exist."""
        # Setup
        mock_unit_of_work.orders.get_order_by_id.return_value = None

        use_case = ModifyOrderUseCase(
            unit_of_work=mock_unit_of_work, broker=mock_broker, order_validator=mock_order_validator
        )

        request = ModifyOrderRequest(order_id=uuid4(), new_quantity=200)

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Order not found" in response.error
        mock_broker.update_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_modify_order_validation_failure(
        self, mock_unit_of_work, mock_broker, mock_order_validator, sample_order
    ):
        """Test modification when validation fails."""
        # Setup
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_order_validator.validate_modification.return_value = ValidationResult(
            is_valid=False, error_message="Cannot modify filled order"
        )

        use_case = ModifyOrderUseCase(
            unit_of_work=mock_unit_of_work, broker=mock_broker, order_validator=mock_order_validator
        )

        request = ModifyOrderRequest(order_id=sample_order.id, new_quantity=200)

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Cannot modify filled order" in response.error
        mock_broker.update_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_modify_order_broker_failure(
        self, mock_unit_of_work, mock_broker, mock_order_validator, sample_order
    ):
        """Test modification when broker fails to update."""
        # Setup
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_broker.update_order.return_value = None

        use_case = ModifyOrderUseCase(
            unit_of_work=mock_unit_of_work, broker=mock_broker, order_validator=mock_order_validator
        )

        request = ModifyOrderRequest(order_id=sample_order.id, new_quantity=200)

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Broker failed to modify order" in response.error
        mock_unit_of_work.orders.update_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_validate_no_modifications(
        self, mock_unit_of_work, mock_broker, mock_order_validator
    ):
        """Test validation when no modifications are specified."""
        use_case = ModifyOrderUseCase(
            unit_of_work=mock_unit_of_work, broker=mock_broker, order_validator=mock_order_validator
        )

        request = ModifyOrderRequest(order_id=uuid4())

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "At least one modification is required" in response.error

    @pytest.mark.asyncio
    async def test_validate_negative_quantity(
        self, mock_unit_of_work, mock_broker, mock_order_validator
    ):
        """Test validation with negative quantity."""
        use_case = ModifyOrderUseCase(
            unit_of_work=mock_unit_of_work, broker=mock_broker, order_validator=mock_order_validator
        )

        request = ModifyOrderRequest(order_id=uuid4(), new_quantity=-100)

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "New quantity must be positive" in response.error


class TestGetOrderStatusUseCase:
    """Test GetOrderStatusUseCase."""

    @pytest.mark.asyncio
    async def test_get_order_status_success(self, mock_unit_of_work, mock_broker, sample_order):
        """Test successful order status retrieval."""
        # Setup
        sample_order.status = OrderStatus.PENDING
        sample_order.filled_quantity = Decimal("50")
        sample_order.average_fill_price = Decimal("150.25")
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_broker.get_order_status.return_value = OrderStatus.PARTIALLY_FILLED

        use_case = GetOrderStatusUseCase(unit_of_work=mock_unit_of_work, broker=mock_broker)

        request = GetOrderStatusRequest(order_id=sample_order.id)

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.status == OrderStatus.PARTIALLY_FILLED.value
        assert response.filled_quantity == 50
        assert response.average_fill_price == 150.25

        # Verify status was updated
        assert sample_order.status == OrderStatus.PARTIALLY_FILLED
        mock_unit_of_work.orders.update_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_order_status_not_found(self, mock_unit_of_work, mock_broker):
        """Test status retrieval when order doesn't exist."""
        # Setup
        mock_unit_of_work.orders.get_order_by_id.return_value = None

        use_case = GetOrderStatusUseCase(unit_of_work=mock_unit_of_work, broker=mock_broker)

        request = GetOrderStatusRequest(order_id=uuid4())

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Order not found" in response.error
        mock_broker.get_order_status.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_order_status_broker_failure(
        self, mock_unit_of_work, mock_broker, sample_order
    ):
        """Test status retrieval when broker query fails."""
        # Setup
        sample_order.status = OrderStatus.PENDING
        sample_order.filled_quantity = Decimal("0")
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_broker.get_order_status.side_effect = Exception("Connection timeout")

        use_case = GetOrderStatusUseCase(unit_of_work=mock_unit_of_work, broker=mock_broker)

        request = GetOrderStatusRequest(order_id=sample_order.id)

        # Execute
        response = await use_case.execute(request)

        # Assert - Should return cached status
        assert response.success is True
        assert response.status == OrderStatus.PENDING.value
        assert response.filled_quantity == 0
        mock_unit_of_work.orders.update_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_order_status_no_change(self, mock_unit_of_work, mock_broker, sample_order):
        """Test status retrieval when status hasn't changed."""
        # Setup
        sample_order.status = OrderStatus.PENDING
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_broker.get_order_status.return_value = OrderStatus.PENDING

        use_case = GetOrderStatusUseCase(unit_of_work=mock_unit_of_work, broker=mock_broker)

        request = GetOrderStatusRequest(order_id=sample_order.id)

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.status == OrderStatus.PENDING.value
        # Update should still be called even if status unchanged
        mock_unit_of_work.orders.update_order.assert_called_once()


class TestRequestResponseDTOs:
    """Test request and response DTOs."""

    def test_place_order_request_defaults(self):
        """Test PlaceOrderRequest default values."""
        request = PlaceOrderRequest(
            portfolio_id=uuid4(), symbol="AAPL", side="buy", order_type="market", quantity=100
        )

        assert request.time_in_force == "day"
        assert request.request_id is not None
        assert request.metadata == {}
        assert request.limit_price is None
        assert request.stop_price is None
        assert request.strategy_id is None

    def test_place_order_response_creation(self):
        """Test PlaceOrderResponse creation."""
        order_id = uuid4()
        request_id = uuid4()

        response = PlaceOrderResponse(
            success=True,
            order_id=order_id,
            broker_order_id="BROKER123",
            status="pending",
            request_id=request_id,
        )

        assert response.success is True
        assert response.order_id == order_id
        assert response.broker_order_id == "BROKER123"
        assert response.status == "pending"
        assert response.request_id == request_id

    def test_cancel_order_request_defaults(self):
        """Test CancelOrderRequest default values."""
        order_id = uuid4()
        request = CancelOrderRequest(order_id=order_id)

        assert request.order_id == order_id
        assert request.reason is None
        assert request.request_id is not None
        assert request.metadata == {}

    def test_modify_order_request_partial_update(self):
        """Test ModifyOrderRequest with partial updates."""
        order_id = uuid4()
        request = ModifyOrderRequest(order_id=order_id, new_quantity=200)

        assert request.order_id == order_id
        assert request.new_quantity == 200
        assert request.new_limit_price is None
        assert request.new_stop_price is None

    def test_get_order_status_response_with_fills(self):
        """Test GetOrderStatusResponse with fill information."""
        response = GetOrderStatusResponse(
            success=True,
            status="partially_filled",
            filled_quantity=50,
            average_fill_price=150.25,
            request_id=uuid4(),
        )

        assert response.success is True
        assert response.status == "partially_filled"
        assert response.filled_quantity == 50
        assert response.average_fill_price == 150.25
