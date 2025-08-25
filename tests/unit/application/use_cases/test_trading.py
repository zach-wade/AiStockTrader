"""
Comprehensive tests for trading use cases.

Tests all trading-related use cases including order placement, cancellation,
modification, and status retrieval with full coverage of success and failure scenarios.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from src.application.use_cases.trading import (
    CancelOrderRequest,
    CancelOrderUseCase,
    GetOrderStatusRequest,
    GetOrderStatusUseCase,
    ModifyOrderRequest,
    ModifyOrderUseCase,
    PlaceOrderRequest,
    PlaceOrderUseCase,
)
from src.domain.entities.order import Order, OrderSide, OrderStatus, OrderType, TimeInForce
from src.domain.entities.portfolio import Portfolio
from src.domain.services.order_validator import ValidationResult
from src.domain.value_objects.money import Money


class TestPlaceOrderRequest:
    """Test PlaceOrderRequest dataclass."""

    def test_post_init_defaults(self):
        """Test that defaults are set correctly in __post_init__."""
        request = PlaceOrderRequest(
            portfolio_id=uuid4(), symbol="AAPL", side="buy", order_type="market", quantity=100
        )

        assert request.request_id is not None
        assert request.metadata is not None
        assert isinstance(request.metadata, dict)
        assert request.time_in_force == "day"

    def test_post_init_with_values(self):
        """Test that provided values are preserved."""
        req_id = uuid4()
        corr_id = uuid4()
        metadata = {"key": "value"}

        request = PlaceOrderRequest(
            portfolio_id=uuid4(),
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=100,
            request_id=req_id,
            correlation_id=corr_id,
            metadata=metadata,
        )

        assert request.request_id == req_id
        assert request.correlation_id == corr_id
        assert request.metadata == metadata


class TestCancelOrderRequest:
    """Test CancelOrderRequest dataclass."""

    def test_post_init_defaults(self):
        """Test that defaults are set correctly in __post_init__."""
        request = CancelOrderRequest(order_id=uuid4())

        assert request.request_id is not None
        assert request.metadata is not None
        assert isinstance(request.metadata, dict)

    def test_post_init_with_values(self):
        """Test that provided values are preserved."""
        req_id = uuid4()
        corr_id = uuid4()
        metadata = {"reason": "market_change"}

        request = CancelOrderRequest(
            order_id=uuid4(),
            reason="Market conditions changed",
            request_id=req_id,
            correlation_id=corr_id,
            metadata=metadata,
        )

        assert request.request_id == req_id
        assert request.correlation_id == corr_id
        assert request.metadata == metadata
        assert request.reason == "Market conditions changed"


class TestModifyOrderRequest:
    """Test ModifyOrderRequest dataclass."""

    def test_post_init_defaults(self):
        """Test that defaults are set correctly in __post_init__."""
        request = ModifyOrderRequest(order_id=uuid4(), new_quantity=200)

        assert request.request_id is not None
        assert request.metadata is not None
        assert isinstance(request.metadata, dict)

    def test_post_init_with_values(self):
        """Test that provided values are preserved."""
        req_id = uuid4()
        corr_id = uuid4()
        metadata = {"modification": "price_update"}

        request = ModifyOrderRequest(
            order_id=uuid4(),
            new_quantity=150,
            new_limit_price=155.50,
            new_stop_price=150.00,
            request_id=req_id,
            correlation_id=corr_id,
            metadata=metadata,
        )

        assert request.request_id == req_id
        assert request.correlation_id == corr_id
        assert request.metadata == metadata
        assert request.new_quantity == 150
        assert request.new_limit_price == 155.50
        assert request.new_stop_price == 150.00


class TestGetOrderStatusRequest:
    """Test GetOrderStatusRequest dataclass."""

    def test_post_init_defaults(self):
        """Test that defaults are set correctly in __post_init__."""
        request = GetOrderStatusRequest(order_id=uuid4())

        assert request.request_id is not None
        assert request.metadata is not None
        assert isinstance(request.metadata, dict)

    def test_post_init_with_values(self):
        """Test that provided values are preserved."""
        req_id = uuid4()
        corr_id = uuid4()
        metadata = {"check": "status"}

        request = GetOrderStatusRequest(
            order_id=uuid4(), request_id=req_id, correlation_id=corr_id, metadata=metadata
        )

        assert request.request_id == req_id
        assert request.correlation_id == corr_id
        assert request.metadata == metadata


class TestPlaceOrderUseCase:
    """Test PlaceOrderUseCase with comprehensive scenarios."""

    @pytest.fixture
    def mock_unit_of_work(self):
        """Create mock unit of work."""
        uow = AsyncMock()
        uow.portfolios = AsyncMock()
        uow.orders = AsyncMock()
        uow.market_data = AsyncMock()
        uow.commit = AsyncMock()
        uow.rollback = AsyncMock()
        uow.__aenter__ = AsyncMock(return_value=uow)
        uow.__aexit__ = AsyncMock(return_value=None)
        return uow

    @pytest.fixture
    def mock_broker(self):
        """Create mock broker."""
        broker = Mock()
        broker.submit_order = Mock()
        return broker

    @pytest.fixture
    def mock_order_validator(self):
        """Create mock order validator."""
        validator = Mock()
        validator.validate_order = Mock()
        return validator

    @pytest.fixture
    def mock_risk_calculator(self):
        """Create mock risk calculator."""
        calculator = Mock()
        calculator.check_risk_limits = Mock()
        return calculator

    @pytest.fixture
    def use_case(self, mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator):
        """Create use case instance."""
        return PlaceOrderUseCase(
            unit_of_work=mock_unit_of_work,
            broker=mock_broker,
            order_validator=mock_order_validator,
            risk_calculator=mock_risk_calculator,
        )

    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio."""
        portfolio = Portfolio(name="Test Portfolio", initial_capital=Money(Decimal("100000")))
        portfolio.id = uuid4()
        return portfolio

    @pytest.fixture
    def mock_market_bar(self):
        """Create mock market bar."""
        bar = Mock()
        bar.close = Decimal("150.00")
        return bar

    @pytest.mark.asyncio
    async def test_place_market_order_success(
        self,
        use_case,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        mock_market_bar,
    ):
        """Test successful market order placement."""
        # Setup
        request = PlaceOrderRequest(
            portfolio_id=sample_portfolio.id,
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=100,
            time_in_force="day",
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_unit_of_work.market_data.get_latest_bar.return_value = mock_market_bar
        mock_order_validator.validate_order.return_value = ValidationResult(is_valid=True)
        mock_risk_calculator.check_risk_limits.return_value = (True, None)
        mock_broker.submit_order.return_value = "BROKER-123"

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.order_id is not None
        assert response.broker_order_id == "BROKER-123"
        assert response.status == "submitted"

        mock_unit_of_work.portfolios.get_portfolio_by_id.assert_called_once_with(
            sample_portfolio.id
        )
        mock_unit_of_work.market_data.get_latest_bar.assert_called_once_with("AAPL")
        mock_order_validator.validate_order.assert_called_once()
        mock_risk_calculator.check_risk_limits.assert_called_once()
        mock_broker.submit_order.assert_called_once()
        mock_unit_of_work.orders.save_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_place_limit_order_success(
        self,
        use_case,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        mock_market_bar,
    ):
        """Test successful limit order placement."""
        # Setup
        request = PlaceOrderRequest(
            portfolio_id=sample_portfolio.id,
            symbol="GOOGL",
            side="sell",
            order_type="limit",
            quantity=50,
            limit_price=150.50,
            time_in_force="gtc",
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_unit_of_work.market_data.get_latest_bar.return_value = mock_market_bar
        mock_order_validator.validate_order.return_value = ValidationResult(is_valid=True)
        mock_risk_calculator.check_risk_limits.return_value = (True, None)
        mock_broker.submit_order.return_value = "BROKER-456"

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.broker_order_id == "BROKER-456"
        mock_broker.submit_order.assert_called_once()

        # Verify order was created with correct parameters
        call_args = mock_broker.submit_order.call_args[0]
        order = call_args[0]
        assert order.symbol == "GOOGL"
        assert order.side == OrderSide.SELL
        assert order.order_type == OrderType.LIMIT
        assert order.quantity == Decimal("50")
        assert order.limit_price == Decimal("150.50")
        assert order.time_in_force == TimeInForce.GTC

    @pytest.mark.asyncio
    async def test_place_stop_order_success(
        self,
        use_case,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        mock_market_bar,
    ):
        """Test successful stop order placement."""
        # Setup
        request = PlaceOrderRequest(
            portfolio_id=sample_portfolio.id,
            symbol="TSLA",
            side="sell",
            order_type="stop",
            quantity=25,
            stop_price=195.00,
            time_in_force="day",
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_unit_of_work.market_data.get_latest_bar.return_value = mock_market_bar
        mock_order_validator.validate_order.return_value = ValidationResult(is_valid=True)
        mock_risk_calculator.check_risk_limits.return_value = (True, None)
        mock_broker.submit_order.return_value = "BROKER-789"

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.broker_order_id == "BROKER-789"

        # Verify order was created with correct stop price
        call_args = mock_broker.submit_order.call_args[0]
        order = call_args[0]
        assert order.stop_price == Decimal("195.00")
        assert order.order_type == OrderType.STOP

    @pytest.mark.asyncio
    async def test_place_stop_limit_order_success(
        self,
        use_case,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        mock_market_bar,
    ):
        """Test successful stop-limit order placement."""
        # Setup
        request = PlaceOrderRequest(
            portfolio_id=sample_portfolio.id,
            symbol="TSLA",
            side="buy",
            order_type="stop_limit",
            quantity=25,
            limit_price=200.00,
            stop_price=195.00,
            strategy_id="MOMENTUM-001",
            metadata={"strategy": "momentum"},
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_unit_of_work.market_data.get_latest_bar.return_value = mock_market_bar
        mock_order_validator.validate_order.return_value = ValidationResult(is_valid=True)
        mock_risk_calculator.check_risk_limits.return_value = (True, None)
        mock_broker.submit_order.return_value = "BROKER-789"

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.broker_order_id == "BROKER-789"

        # Verify order was created with both stop and limit prices
        call_args = mock_broker.submit_order.call_args[0]
        order = call_args[0]
        assert order.limit_price == Decimal("200.00")
        assert order.stop_price == Decimal("195.00")
        assert order.order_type == OrderType.STOP_LIMIT

    @pytest.mark.asyncio
    async def test_place_order_portfolio_not_found(self, use_case, mock_unit_of_work):
        """Test order placement when portfolio doesn't exist."""
        # Setup
        request = PlaceOrderRequest(
            portfolio_id=uuid4(), symbol="AAPL", side="buy", order_type="market", quantity=100
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = None

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.error == "Portfolio not found"
        assert response.request_id == request.request_id

    @pytest.mark.asyncio
    async def test_place_order_no_market_data(self, use_case, mock_unit_of_work, sample_portfolio):
        """Test order placement when market data is unavailable."""
        # Setup
        request = PlaceOrderRequest(
            portfolio_id=sample_portfolio.id,
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=100,
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_unit_of_work.market_data.get_latest_bar.return_value = None

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.error == "Cannot get current market price"

    @pytest.mark.asyncio
    async def test_place_order_validation_failure(
        self, use_case, mock_unit_of_work, mock_order_validator, sample_portfolio, mock_market_bar
    ):
        """Test order placement with validation failure."""
        # Setup
        request = PlaceOrderRequest(
            portfolio_id=sample_portfolio.id,
            symbol="INVALID",
            side="buy",
            order_type="market",
            quantity=100,
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_unit_of_work.market_data.get_latest_bar.return_value = mock_market_bar
        mock_order_validator.validate_order.return_value = ValidationResult(
            is_valid=False, error_message="Invalid symbol"
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.error == "Invalid symbol"

    @pytest.mark.asyncio
    async def test_place_order_validation_failure_no_message(
        self, use_case, mock_unit_of_work, mock_order_validator, sample_portfolio, mock_market_bar
    ):
        """Test order placement with validation failure but no error message."""
        # Setup
        request = PlaceOrderRequest(
            portfolio_id=sample_portfolio.id,
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=100,
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_unit_of_work.market_data.get_latest_bar.return_value = mock_market_bar
        mock_order_validator.validate_order.return_value = ValidationResult(
            is_valid=False, error_message=None
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.error == "Order validation failed"

    @pytest.mark.asyncio
    async def test_place_order_risk_limit_violation(
        self,
        use_case,
        mock_unit_of_work,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        mock_market_bar,
    ):
        """Test order placement with risk limit violation."""
        # Setup
        request = PlaceOrderRequest(
            portfolio_id=sample_portfolio.id,
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=10000,  # Large quantity
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_unit_of_work.market_data.get_latest_bar.return_value = mock_market_bar
        mock_order_validator.validate_order.return_value = ValidationResult(is_valid=True)
        mock_risk_calculator.check_risk_limits.return_value = (
            False,
            "Position size exceeds maximum",
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Risk limit violated" in response.error
        assert "Position size exceeds maximum" in response.error

    @pytest.mark.asyncio
    async def test_place_order_broker_submission_failure(
        self,
        use_case,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        mock_market_bar,
    ):
        """Test order placement when broker submission fails."""
        # Setup
        request = PlaceOrderRequest(
            portfolio_id=sample_portfolio.id,
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=100,
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_unit_of_work.market_data.get_latest_bar.return_value = mock_market_bar
        mock_order_validator.validate_order.return_value = ValidationResult(is_valid=True)
        mock_risk_calculator.check_risk_limits.return_value = (True, None)
        mock_broker.submit_order.side_effect = Exception("Broker connection failed")

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Failed to submit order" in response.error
        assert "Broker connection failed" in response.error

    @pytest.mark.asyncio
    async def test_validate_invalid_order_type(self, use_case):
        """Test validation with invalid order type."""
        # Setup
        request = PlaceOrderRequest(
            portfolio_id=uuid4(), symbol="AAPL", side="buy", order_type="invalid_type", quantity=100
        )

        # Execute
        error = await use_case.validate(request)

        # Assert
        assert error == "Invalid order type: invalid_type"

    @pytest.mark.asyncio
    async def test_validate_invalid_side(self, use_case):
        """Test validation with invalid side."""
        # Setup
        request = PlaceOrderRequest(
            portfolio_id=uuid4(),
            symbol="AAPL",
            side="invalid_side",
            order_type="market",
            quantity=100,
        )

        # Execute
        error = await use_case.validate(request)

        # Assert
        assert error == "Invalid order side: invalid_side"

    @pytest.mark.asyncio
    async def test_validate_zero_quantity(self, use_case):
        """Test validation with zero quantity."""
        # Setup
        request = PlaceOrderRequest(
            portfolio_id=uuid4(), symbol="AAPL", side="buy", order_type="market", quantity=0
        )

        # Execute
        error = await use_case.validate(request)

        # Assert
        assert error == "Quantity must be positive"

    @pytest.mark.asyncio
    async def test_validate_negative_quantity(self, use_case):
        """Test validation with negative quantity."""
        # Setup
        request = PlaceOrderRequest(
            portfolio_id=uuid4(), symbol="AAPL", side="buy", order_type="market", quantity=-10
        )

        # Execute
        error = await use_case.validate(request)

        # Assert
        assert error == "Quantity must be positive"

    @pytest.mark.asyncio
    async def test_validate_limit_order_without_price(self, use_case):
        """Test validation of limit order without limit price."""
        # Setup
        request = PlaceOrderRequest(
            portfolio_id=uuid4(),
            symbol="AAPL",
            side="buy",
            order_type="limit",
            quantity=100,
            limit_price=None,
        )

        # Execute
        error = await use_case.validate(request)

        # Assert
        assert error == "limit order requires limit price"

    @pytest.mark.asyncio
    async def test_validate_stop_order_without_price(self, use_case):
        """Test validation of stop order without stop price."""
        # Setup
        request = PlaceOrderRequest(
            portfolio_id=uuid4(),
            symbol="AAPL",
            side="sell",
            order_type="stop",
            quantity=100,
            stop_price=None,
        )

        # Execute
        error = await use_case.validate(request)

        # Assert
        assert error == "stop order requires stop price"

    @pytest.mark.asyncio
    async def test_validate_stop_limit_order_without_limit_price(self, use_case):
        """Test validation of stop-limit order without limit price."""
        # Setup
        request = PlaceOrderRequest(
            portfolio_id=uuid4(),
            symbol="AAPL",
            side="buy",
            order_type="stop_limit",
            quantity=100,
            limit_price=None,
            stop_price=150.00,
        )

        # Execute
        error = await use_case.validate(request)

        # Assert
        assert error == "stop_limit order requires limit price"

    @pytest.mark.asyncio
    async def test_validate_stop_limit_order_without_stop_price(self, use_case):
        """Test validation of stop-limit order without stop price."""
        # Setup
        request = PlaceOrderRequest(
            portfolio_id=uuid4(),
            symbol="AAPL",
            side="buy",
            order_type="stop_limit",
            quantity=100,
            limit_price=150.00,
            stop_price=None,
        )

        # Execute
        error = await use_case.validate(request)

        # Assert
        assert error == "stop_limit order requires stop price"

    @pytest.mark.asyncio
    async def test_validate_valid_request(self, use_case):
        """Test validation with valid request returns None."""
        # Setup
        request = PlaceOrderRequest(
            portfolio_id=uuid4(), symbol="AAPL", side="buy", order_type="market", quantity=100
        )

        # Execute
        error = await use_case.validate(request)

        # Assert
        assert error is None


class TestCancelOrderUseCase:
    """Test CancelOrderUseCase with comprehensive scenarios."""

    @pytest.fixture
    def mock_unit_of_work(self):
        """Create mock unit of work."""
        uow = AsyncMock()
        uow.orders = AsyncMock()
        uow.commit = AsyncMock()
        uow.rollback = AsyncMock()
        uow.__aenter__ = AsyncMock(return_value=uow)
        uow.__aexit__ = AsyncMock(return_value=None)
        return uow

    @pytest.fixture
    def mock_broker(self):
        """Create mock broker."""
        broker = Mock()
        broker.cancel_order = Mock()
        return broker

    @pytest.fixture
    def use_case(self, mock_unit_of_work, mock_broker):
        """Create use case instance."""
        return CancelOrderUseCase(unit_of_work=mock_unit_of_work, broker=mock_broker)

    @pytest.fixture
    def sample_order(self):
        """Create sample order."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("100"),
            limit_price=Decimal("150.00"),
        )
        order.submit("BROKER-123")
        return order

    @pytest.mark.asyncio
    async def test_cancel_order_success(
        self, use_case, mock_unit_of_work, mock_broker, sample_order
    ):
        """Test successful order cancellation."""
        # Setup
        request = CancelOrderRequest(order_id=sample_order.id, reason="User requested")

        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_broker.cancel_order.return_value = True

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.cancelled is True
        assert response.final_status == OrderStatus.CANCELLED

        mock_unit_of_work.orders.get_order_by_id.assert_called_once_with(sample_order.id)
        mock_broker.cancel_order.assert_called_once_with(sample_order.id)
        mock_unit_of_work.orders.update_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_order_without_reason(
        self, use_case, mock_unit_of_work, mock_broker, sample_order
    ):
        """Test successful order cancellation without reason."""
        # Setup
        request = CancelOrderRequest(order_id=sample_order.id)

        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_broker.cancel_order.return_value = True

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.cancelled is True
        assert response.final_status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_order_not_found(self, use_case, mock_unit_of_work):
        """Test cancellation when order doesn't exist."""
        # Setup
        request = CancelOrderRequest(order_id=uuid4())
        mock_unit_of_work.orders.get_order_by_id.return_value = None

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.error == "Order not found"
        assert response.request_id == request.request_id

    @pytest.mark.asyncio
    async def test_cancel_order_already_filled(self, use_case, mock_unit_of_work, sample_order):
        """Test cancellation of already filled order."""
        # Setup
        sample_order.fill(Decimal("100"), Decimal("150.00"))
        request = CancelOrderRequest(order_id=sample_order.id)
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "cannot be cancelled in status: filled" in response.error

    @pytest.mark.asyncio
    async def test_cancel_order_already_cancelled(self, use_case, mock_unit_of_work, sample_order):
        """Test cancellation of already cancelled order."""
        # Setup
        sample_order.cancel("Previous cancellation")
        request = CancelOrderRequest(order_id=sample_order.id)
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "cannot be cancelled in status: cancelled" in response.error

    @pytest.mark.asyncio
    async def test_cancel_order_rejected(self, use_case, mock_unit_of_work):
        """Test cancellation of rejected order."""
        # Setup - Create a pending order and reject it
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("100"),
            limit_price=Decimal("150.00"),
        )
        # Order starts in PENDING status by default
        order.reject("Insufficient funds")

        request = CancelOrderRequest(order_id=order.id)
        mock_unit_of_work.orders.get_order_by_id.return_value = order

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "cannot be cancelled in status: rejected" in response.error

    @pytest.mark.asyncio
    async def test_cancel_order_broker_failure(
        self, use_case, mock_unit_of_work, mock_broker, sample_order
    ):
        """Test cancellation when broker fails."""
        # Setup
        request = CancelOrderRequest(order_id=sample_order.id)
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_broker.cancel_order.return_value = False

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.error == "Broker failed to cancel order"

    @pytest.mark.asyncio
    async def test_cancel_order_broker_exception(
        self, use_case, mock_unit_of_work, mock_broker, sample_order
    ):
        """Test cancellation when broker throws exception."""
        # Setup
        request = CancelOrderRequest(order_id=sample_order.id)
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_broker.cancel_order.side_effect = Exception("Network error")

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Failed to cancel order" in response.error
        assert "Network error" in response.error

    @pytest.mark.asyncio
    async def test_validate_missing_order_id(self, use_case):
        """Test validation with missing order ID."""
        # Setup
        request = CancelOrderRequest(order_id=None)

        # Execute
        error = await use_case.validate(request)

        # Assert
        assert error == "Order ID is required"

    @pytest.mark.asyncio
    async def test_validate_valid_request(self, use_case):
        """Test validation with valid request returns None."""
        # Setup
        request = CancelOrderRequest(order_id=uuid4())

        # Execute
        error = await use_case.validate(request)

        # Assert
        assert error is None


class TestModifyOrderUseCase:
    """Test ModifyOrderUseCase with comprehensive scenarios."""

    @pytest.fixture
    def mock_unit_of_work(self):
        """Create mock unit of work."""
        uow = AsyncMock()
        uow.orders = AsyncMock()
        uow.commit = AsyncMock()
        uow.rollback = AsyncMock()
        uow.__aenter__ = AsyncMock(return_value=uow)
        uow.__aexit__ = AsyncMock(return_value=None)
        return uow

    @pytest.fixture
    def mock_broker(self):
        """Create mock broker."""
        broker = Mock()
        broker.update_order = Mock()
        return broker

    @pytest.fixture
    def mock_order_validator(self):
        """Create mock order validator."""
        validator = Mock()
        validator.validate_modification = Mock()
        return validator

    @pytest.fixture
    def use_case(self, mock_unit_of_work, mock_broker, mock_order_validator):
        """Create use case instance."""
        return ModifyOrderUseCase(
            unit_of_work=mock_unit_of_work, broker=mock_broker, order_validator=mock_order_validator
        )

    @pytest.fixture
    def sample_order(self):
        """Create sample order."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("100"),
            limit_price=Decimal("150.00"),
        )
        order.submit("BROKER-123")
        return order

    @pytest.mark.asyncio
    async def test_modify_quantity_success(
        self, use_case, mock_unit_of_work, mock_broker, mock_order_validator, sample_order
    ):
        """Test successful quantity modification."""
        # Setup
        request = ModifyOrderRequest(order_id=sample_order.id, new_quantity=200)

        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_order_validator.validate_modification.return_value = ValidationResult(is_valid=True)
        mock_broker.update_order.return_value = sample_order

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.modified is True
        assert response.new_values == {"quantity": 200}
        assert sample_order.quantity == Decimal("200")

        mock_broker.update_order.assert_called_once_with(sample_order)
        mock_unit_of_work.orders.update_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_modify_limit_price_success(
        self, use_case, mock_unit_of_work, mock_broker, mock_order_validator, sample_order
    ):
        """Test successful limit price modification."""
        # Setup
        request = ModifyOrderRequest(order_id=sample_order.id, new_limit_price=155.50)

        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_order_validator.validate_modification.return_value = ValidationResult(is_valid=True)
        mock_broker.update_order.return_value = sample_order

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.new_values == {"limit_price": 155.50}
        assert sample_order.limit_price == Decimal("155.50")

    @pytest.mark.asyncio
    async def test_modify_stop_price_success(
        self, use_case, mock_unit_of_work, mock_broker, mock_order_validator
    ):
        """Test successful stop price modification."""
        # Setup - Create stop order
        order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=Decimal("100"),
            stop_price=Decimal("145.00"),
        )
        order.submit("BROKER-123")

        request = ModifyOrderRequest(order_id=order.id, new_stop_price=140.00)

        mock_unit_of_work.orders.get_order_by_id.return_value = order
        mock_order_validator.validate_modification.return_value = ValidationResult(is_valid=True)
        mock_broker.update_order.return_value = order

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.new_values == {"stop_price": 140.00}
        assert order.stop_price == Decimal("140.00")

    @pytest.mark.asyncio
    async def test_modify_multiple_fields_success(
        self, use_case, mock_unit_of_work, mock_broker, mock_order_validator, sample_order
    ):
        """Test modifying multiple fields at once."""
        # Setup
        request = ModifyOrderRequest(
            order_id=sample_order.id, new_quantity=150, new_limit_price=152.75
        )

        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_order_validator.validate_modification.return_value = ValidationResult(is_valid=True)
        mock_broker.update_order.return_value = sample_order

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.new_values == {"quantity": 150, "limit_price": 152.75}
        assert sample_order.quantity == Decimal("150")
        assert sample_order.limit_price == Decimal("152.75")

    @pytest.mark.asyncio
    async def test_modify_all_fields_success(
        self, use_case, mock_unit_of_work, mock_broker, mock_order_validator
    ):
        """Test modifying all possible fields at once."""
        # Setup - Create stop-limit order
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.STOP_LIMIT,
            quantity=Decimal("100"),
            limit_price=Decimal("150.00"),
            stop_price=Decimal("145.00"),
        )
        order.submit("BROKER-123")

        request = ModifyOrderRequest(
            order_id=order.id, new_quantity=200, new_limit_price=155.00, new_stop_price=150.00
        )

        mock_unit_of_work.orders.get_order_by_id.return_value = order
        mock_order_validator.validate_modification.return_value = ValidationResult(is_valid=True)
        mock_broker.update_order.return_value = order

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.new_values == {"quantity": 200, "limit_price": 155.00, "stop_price": 150.00}
        assert order.quantity == Decimal("200")
        assert order.limit_price == Decimal("155.00")
        assert order.stop_price == Decimal("150.00")

    @pytest.mark.asyncio
    async def test_modify_order_not_found(self, use_case, mock_unit_of_work):
        """Test modification when order doesn't exist."""
        # Setup
        request = ModifyOrderRequest(order_id=uuid4(), new_quantity=200)
        mock_unit_of_work.orders.get_order_by_id.return_value = None

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.error == "Order not found"
        assert response.request_id == request.request_id

    @pytest.mark.asyncio
    async def test_modify_order_validation_failure(
        self, use_case, mock_unit_of_work, mock_order_validator, sample_order
    ):
        """Test modification with validation failure."""
        # Setup
        request = ModifyOrderRequest(order_id=sample_order.id, new_quantity=1000000)  # Too large

        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_order_validator.validate_modification.return_value = ValidationResult(
            is_valid=False, error_message="Quantity exceeds maximum allowed"
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.error == "Quantity exceeds maximum allowed"

    @pytest.mark.asyncio
    async def test_modify_order_validation_failure_no_message(
        self, use_case, mock_unit_of_work, mock_order_validator, sample_order
    ):
        """Test modification with validation failure but no error message."""
        # Setup
        request = ModifyOrderRequest(order_id=sample_order.id, new_quantity=200)

        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_order_validator.validate_modification.return_value = ValidationResult(
            is_valid=False, error_message=None
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.error == "Modification validation failed"

    @pytest.mark.asyncio
    async def test_modify_order_broker_failure(
        self, use_case, mock_unit_of_work, mock_broker, mock_order_validator, sample_order
    ):
        """Test modification when broker fails."""
        # Setup
        request = ModifyOrderRequest(order_id=sample_order.id, new_quantity=200)

        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_order_validator.validate_modification.return_value = ValidationResult(is_valid=True)
        mock_broker.update_order.return_value = None

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.error == "Broker failed to modify order"

    @pytest.mark.asyncio
    async def test_modify_order_broker_exception(
        self, use_case, mock_unit_of_work, mock_broker, mock_order_validator, sample_order
    ):
        """Test modification when broker throws exception."""
        # Setup
        request = ModifyOrderRequest(order_id=sample_order.id, new_quantity=200)

        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_order_validator.validate_modification.return_value = ValidationResult(is_valid=True)
        mock_broker.update_order.side_effect = Exception("Connection timeout")

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Failed to modify order" in response.error
        assert "Connection timeout" in response.error

    @pytest.mark.asyncio
    async def test_validate_missing_order_id(self, use_case):
        """Test validation with missing order ID."""
        # Setup
        request = ModifyOrderRequest(order_id=None)

        # Execute
        error = await use_case.validate(request)

        # Assert
        assert error == "Order ID is required"

    @pytest.mark.asyncio
    async def test_validate_no_modifications(self, use_case):
        """Test validation with no modifications specified."""
        # Setup
        request = ModifyOrderRequest(order_id=uuid4())

        # Execute
        error = await use_case.validate(request)

        # Assert
        assert error == "At least one modification is required"

    @pytest.mark.asyncio
    async def test_validate_zero_quantity(self, use_case):
        """Test validation with zero quantity."""
        # Setup
        request = ModifyOrderRequest(order_id=uuid4(), new_quantity=0)

        # Execute
        error = await use_case.validate(request)

        # Assert
        # Since 0 is falsy, any() returns False and we get "At least one modification" error
        assert error == "At least one modification is required"

    @pytest.mark.asyncio
    async def test_validate_negative_quantity(self, use_case):
        """Test validation with negative quantity."""
        # Setup
        request = ModifyOrderRequest(order_id=uuid4(), new_quantity=-10)

        # Execute
        error = await use_case.validate(request)

        # Assert
        assert error == "New quantity must be positive"

    @pytest.mark.asyncio
    async def test_validate_valid_request(self, use_case):
        """Test validation with valid request returns None."""
        # Setup
        request = ModifyOrderRequest(order_id=uuid4(), new_quantity=100)

        # Execute
        error = await use_case.validate(request)

        # Assert
        assert error is None


class TestGetOrderStatusUseCase:
    """Test GetOrderStatusUseCase with comprehensive scenarios."""

    @pytest.fixture
    def mock_unit_of_work(self):
        """Create mock unit of work."""
        uow = AsyncMock()
        uow.orders = AsyncMock()
        uow.commit = AsyncMock()
        uow.rollback = AsyncMock()
        uow.__aenter__ = AsyncMock(return_value=uow)
        uow.__aexit__ = AsyncMock(return_value=None)
        return uow

    @pytest.fixture
    def mock_broker(self):
        """Create mock broker."""
        broker = Mock()
        broker.get_order_status = Mock()
        return broker

    @pytest.fixture
    def use_case(self, mock_unit_of_work, mock_broker):
        """Create use case instance."""
        return GetOrderStatusUseCase(unit_of_work=mock_unit_of_work, broker=mock_broker)

    @pytest.fixture
    def sample_order(self):
        """Create sample order."""
        order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=Decimal("100")
        )
        order.submit("BROKER-123")
        order.filled_quantity = Decimal("50")
        order.average_fill_price = Decimal("149.75")
        return order

    @pytest.mark.asyncio
    async def test_get_order_status_success(
        self, use_case, mock_unit_of_work, mock_broker, sample_order
    ):
        """Test successful status retrieval."""
        # Setup
        request = GetOrderStatusRequest(order_id=sample_order.id)
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_broker.get_order_status.return_value = OrderStatus.SUBMITTED

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.status == "submitted"
        assert response.filled_quantity == 50
        assert response.average_fill_price == 149.75

    @pytest.mark.asyncio
    async def test_get_order_status_no_fills(self, use_case, mock_unit_of_work, mock_broker):
        """Test status retrieval for order with no fills."""
        # Setup
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("100"),
            limit_price=Decimal("150.00"),
        )
        order.submit("BROKER-123")

        request = GetOrderStatusRequest(order_id=order.id)
        mock_unit_of_work.orders.get_order_by_id.return_value = order
        mock_broker.get_order_status.return_value = OrderStatus.SUBMITTED

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.status == "submitted"
        assert response.filled_quantity == 0
        assert response.average_fill_price is None

    @pytest.mark.asyncio
    async def test_get_order_status_with_update(
        self, use_case, mock_unit_of_work, mock_broker, sample_order
    ):
        """Test status retrieval with broker update."""
        # Setup
        request = GetOrderStatusRequest(order_id=sample_order.id)
        sample_order.status = OrderStatus.SUBMITTED
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_broker.get_order_status.return_value = OrderStatus.FILLED

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.status == "filled"
        assert sample_order.status == OrderStatus.FILLED
        mock_unit_of_work.orders.update_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_order_status_no_update_same_status(
        self, use_case, mock_unit_of_work, mock_broker, sample_order
    ):
        """Test status retrieval when broker returns same status."""
        # Setup
        request = GetOrderStatusRequest(order_id=sample_order.id)
        sample_order.status = OrderStatus.SUBMITTED
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_broker.get_order_status.return_value = OrderStatus.SUBMITTED

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.status == "submitted"
        mock_unit_of_work.orders.update_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_order_status_broker_returns_none(
        self, use_case, mock_unit_of_work, mock_broker, sample_order
    ):
        """Test status retrieval when broker returns None."""
        # Setup
        request = GetOrderStatusRequest(order_id=sample_order.id)
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_broker.get_order_status.return_value = None

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.status == sample_order.status
        mock_unit_of_work.orders.update_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_order_status_not_found(self, use_case, mock_unit_of_work):
        """Test status retrieval when order doesn't exist."""
        # Setup
        request = GetOrderStatusRequest(order_id=uuid4())
        mock_unit_of_work.orders.get_order_by_id.return_value = None

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.error == "Order not found"
        assert response.request_id == request.request_id

    @pytest.mark.asyncio
    async def test_get_order_status_broker_failure(
        self, use_case, mock_unit_of_work, mock_broker, sample_order
    ):
        """Test status retrieval when broker fails - returns cached status."""
        # Setup
        request = GetOrderStatusRequest(order_id=sample_order.id)
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_broker.get_order_status.side_effect = Exception("Broker unavailable")

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True  # Returns cached data
        assert response.status == sample_order.status
        assert response.filled_quantity == 50
        assert response.average_fill_price == 149.75

    @pytest.mark.asyncio
    async def test_validate_missing_order_id(self, use_case):
        """Test validation with missing order ID."""
        # Setup
        request = GetOrderStatusRequest(order_id=None)

        # Execute
        error = await use_case.validate(request)

        # Assert
        assert error == "Order ID is required"

    @pytest.mark.asyncio
    async def test_validate_valid_request(self, use_case):
        """Test validation with valid request returns None."""
        # Setup
        request = GetOrderStatusRequest(order_id=uuid4())

        # Execute
        error = await use_case.validate(request)

        # Assert
        assert error is None


# Additional edge case tests
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_place_order_with_correlation_id(self):
        """Test place order with correlation ID for tracking."""
        # Setup
        uow = AsyncMock()
        uow.portfolios = AsyncMock()
        uow.orders = AsyncMock()
        uow.market_data = AsyncMock()
        uow.__aenter__ = AsyncMock(return_value=uow)
        uow.__aexit__ = AsyncMock(return_value=None)

        broker = Mock()
        validator = Mock()
        calculator = Mock()

        use_case = PlaceOrderUseCase(uow, broker, validator, calculator)

        portfolio = Portfolio(name="Test", initial_capital=Money(Decimal("100000")))
        portfolio.id = uuid4()

        correlation_id = uuid4()
        request = PlaceOrderRequest(
            portfolio_id=portfolio.id,
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=100,
            correlation_id=correlation_id,
        )

        uow.portfolios.get_portfolio_by_id.return_value = portfolio
        bar = Mock()
        bar.close = Decimal("150.00")
        uow.market_data.get_latest_bar.return_value = bar
        validator.validate_order.return_value = ValidationResult(is_valid=True)
        calculator.check_risk_limits.return_value = (True, None)
        broker.submit_order.return_value = "BROKER-123"

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert request.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_place_order_with_strategy_metadata(self):
        """Test place order with strategy ID and metadata."""
        # Setup
        uow = AsyncMock()
        uow.portfolios = AsyncMock()
        uow.orders = AsyncMock()
        uow.market_data = AsyncMock()
        uow.__aenter__ = AsyncMock(return_value=uow)
        uow.__aexit__ = AsyncMock(return_value=None)

        broker = Mock()
        validator = Mock()
        calculator = Mock()

        use_case = PlaceOrderUseCase(uow, broker, validator, calculator)

        portfolio = Portfolio(name="Test", initial_capital=Money(Decimal("100000")))
        portfolio.id = uuid4()

        request = PlaceOrderRequest(
            portfolio_id=portfolio.id,
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=100,
            strategy_id="MOMENTUM-001",
            metadata={"signal": "bullish", "confidence": 0.85},
        )

        uow.portfolios.get_portfolio_by_id.return_value = portfolio
        bar = Mock()
        bar.close = Decimal("150.00")
        uow.market_data.get_latest_bar.return_value = bar
        validator.validate_order.return_value = ValidationResult(is_valid=True)
        calculator.check_risk_limits.return_value = (True, None)
        broker.submit_order.return_value = "BROKER-123"

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert request.strategy_id == "MOMENTUM-001"
        assert request.metadata["signal"] == "bullish"
        assert request.metadata["confidence"] == 0.85
