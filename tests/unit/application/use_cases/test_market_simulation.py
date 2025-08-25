"""
Comprehensive unit tests for Market Simulation Use Cases.

Tests all market simulation functionality including:
- Market price updates and order triggering
- Pending order processing
- Order trigger condition checking

Full coverage including:
- __init__ methods for all use cases
- __post_init__ methods for all request objects
- validate methods
- process methods
- _should_trigger_order helper methods
- All order types (MARKET, LIMIT, STOP, STOP_LIMIT)
- All order sides (BUY, SELL)
- All edge cases and error paths
"""

import logging
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from src.application.use_cases.market_simulation import (
    CheckOrderTriggerRequest,
    CheckOrderTriggerResponse,
    CheckOrderTriggerUseCase,
    ProcessPendingOrdersRequest,
    ProcessPendingOrdersResponse,
    ProcessPendingOrdersUseCase,
    UpdateMarketPriceRequest,
    UpdateMarketPriceResponse,
    UpdateMarketPriceUseCase,
)
from src.domain.entities.order import OrderSide, OrderStatus, OrderType
from src.domain.value_objects.price import Price
from src.domain.value_objects.symbol import Symbol

# =============================================================================
# UpdateMarketPriceRequest Tests
# =============================================================================


class TestUpdateMarketPriceRequest:
    """Test UpdateMarketPriceRequest DTO."""

    def test_create_request_minimal(self):
        """Test creating request with minimal required fields."""
        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("150.00"))

        assert request.symbol == "AAPL"
        assert request.price == Decimal("150.00")
        assert request.volume is None
        assert request.timestamp is None
        assert isinstance(request.request_id, UUID)
        assert request.correlation_id is None
        assert request.metadata == {}

    def test_create_request_with_all_fields(self):
        """Test creating request with all fields specified."""
        request_id = uuid4()
        correlation_id = uuid4()
        timestamp = datetime.now(UTC)
        metadata = {"source": "market_feed", "priority": "high"}

        request = UpdateMarketPriceRequest(
            symbol="AAPL",
            price=Decimal("150.00"),
            volume=100000,
            timestamp=timestamp,
            request_id=request_id,
            correlation_id=correlation_id,
            metadata=metadata,
        )

        assert request.symbol == "AAPL"
        assert request.price == Decimal("150.00")
        assert request.volume == 100000
        assert request.timestamp == timestamp
        assert request.request_id == request_id
        assert request.correlation_id == correlation_id
        assert request.metadata == metadata

    def test_post_init_with_none_values(self):
        """Test __post_init__ properly handles None values."""
        request = UpdateMarketPriceRequest(
            symbol="AAPL", price=Decimal("150.00"), request_id=None, metadata=None
        )

        assert request.request_id is not None
        assert isinstance(request.request_id, UUID)
        assert request.metadata == {}

    def test_post_init_preserves_existing_values(self):
        """Test __post_init__ preserves explicitly set values."""
        existing_id = uuid4()
        existing_metadata = {"key": "value"}

        request = UpdateMarketPriceRequest(
            symbol="AAPL",
            price=Decimal("150.00"),
            request_id=existing_id,
            metadata=existing_metadata,
        )

        assert request.request_id == existing_id
        assert request.metadata == existing_metadata


# =============================================================================
# ProcessPendingOrdersRequest Tests
# =============================================================================


class TestProcessPendingOrdersRequest:
    """Test ProcessPendingOrdersRequest DTO."""

    def test_create_request_minimal(self):
        """Test creating request with no fields."""
        request = ProcessPendingOrdersRequest()

        assert request.symbol is None
        assert request.current_prices is None
        assert isinstance(request.request_id, UUID)
        assert request.correlation_id is None
        assert request.metadata == {}

    def test_create_request_with_all_fields(self):
        """Test creating request with all fields."""
        request_id = uuid4()
        correlation_id = uuid4()
        current_prices = {"AAPL": Decimal("150.00"), "GOOGL": Decimal("2800.00")}
        metadata = {"batch": "1", "source": "scheduler"}

        request = ProcessPendingOrdersRequest(
            symbol="AAPL",
            current_prices=current_prices,
            request_id=request_id,
            correlation_id=correlation_id,
            metadata=metadata,
        )

        assert request.symbol == "AAPL"
        assert request.current_prices == current_prices
        assert request.request_id == request_id
        assert request.correlation_id == correlation_id
        assert request.metadata == metadata

    def test_post_init_with_none_values(self):
        """Test __post_init__ properly handles None values."""
        request = ProcessPendingOrdersRequest(request_id=None, metadata=None)

        assert request.request_id is not None
        assert isinstance(request.request_id, UUID)
        assert request.metadata == {}


# =============================================================================
# CheckOrderTriggerRequest Tests
# =============================================================================


class TestCheckOrderTriggerRequest:
    """Test CheckOrderTriggerRequest DTO."""

    def test_create_request_minimal(self):
        """Test creating request with minimal required fields."""
        order_id = uuid4()
        request = CheckOrderTriggerRequest(order_id=order_id, current_price=Decimal("150.00"))

        assert request.order_id == order_id
        assert request.current_price == Decimal("150.00")
        assert isinstance(request.request_id, UUID)
        assert request.correlation_id is None
        assert request.metadata == {}

    def test_create_request_with_all_fields(self):
        """Test creating request with all fields."""
        order_id = uuid4()
        request_id = uuid4()
        correlation_id = uuid4()
        metadata = {"check_reason": "price_update"}

        request = CheckOrderTriggerRequest(
            order_id=order_id,
            current_price=Decimal("150.00"),
            request_id=request_id,
            correlation_id=correlation_id,
            metadata=metadata,
        )

        assert request.order_id == order_id
        assert request.current_price == Decimal("150.00")
        assert request.request_id == request_id
        assert request.correlation_id == correlation_id
        assert request.metadata == metadata

    def test_post_init_with_none_values(self):
        """Test __post_init__ properly handles None values."""
        order_id = uuid4()
        request = CheckOrderTriggerRequest(
            order_id=order_id, current_price=Decimal("150.00"), request_id=None, metadata=None
        )

        assert request.request_id is not None
        assert isinstance(request.request_id, UUID)
        assert request.metadata == {}


# =============================================================================
# UpdateMarketPriceUseCase Tests
# =============================================================================


class TestUpdateMarketPriceUseCase:
    """Test UpdateMarketPriceUseCase."""

    @pytest.fixture
    def mock_unit_of_work(self):
        """Create mock unit of work."""
        mock_uow = AsyncMock()
        mock_uow.orders = AsyncMock()
        mock_uow.market_data = AsyncMock()
        mock_uow.commit = AsyncMock()
        mock_uow.rollback = AsyncMock()
        return mock_uow

    @pytest.fixture
    def mock_order_processor(self):
        """Create mock order processor."""
        return MagicMock()

    @pytest.fixture
    def mock_market_microstructure(self):
        """Create mock market microstructure."""
        mock_mm = MagicMock()
        mock_mm.calculate_execution_price = MagicMock(return_value=Price(Decimal("150.00")))
        return mock_mm

    @pytest.fixture
    def use_case(self, mock_unit_of_work, mock_order_processor, mock_market_microstructure):
        """Create use case instance."""
        return UpdateMarketPriceUseCase(
            unit_of_work=mock_unit_of_work,
            order_processor=mock_order_processor,
            market_microstructure=mock_market_microstructure,
        )

    def test_init(self, mock_unit_of_work, mock_order_processor, mock_market_microstructure):
        """Test use case initialization."""
        use_case = UpdateMarketPriceUseCase(
            unit_of_work=mock_unit_of_work,
            order_processor=mock_order_processor,
            market_microstructure=mock_market_microstructure,
        )

        assert use_case.unit_of_work == mock_unit_of_work
        assert use_case.order_processor == mock_order_processor
        assert use_case.market_microstructure == mock_market_microstructure
        assert use_case.name == "UpdateMarketPriceUseCase"
        assert isinstance(use_case.logger, logging.Logger)

    @pytest.mark.asyncio
    async def test_validate_valid_request(self, use_case):
        """Test validation with valid request."""
        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("150.00"), volume=1000)

        error = await use_case.validate(request)
        assert error is None

    @pytest.mark.asyncio
    async def test_validate_zero_price(self, use_case):
        """Test validation with zero price."""
        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("0"))

        error = await use_case.validate(request)
        assert error == "Price must be positive"

    @pytest.mark.asyncio
    async def test_validate_negative_price(self, use_case):
        """Test validation with negative price."""
        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("-10.00"))

        error = await use_case.validate(request)
        assert error == "Price must be positive"

    @pytest.mark.asyncio
    async def test_validate_negative_volume(self, use_case):
        """Test validation with negative volume."""
        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("150.00"), volume=-100)

        error = await use_case.validate(request)
        assert error == "Volume cannot be negative"

    @pytest.mark.asyncio
    async def test_validate_zero_volume(self, use_case):
        """Test validation with zero volume (valid)."""
        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("150.00"), volume=0)

        error = await use_case.validate(request)
        assert error is None

    @pytest.mark.asyncio
    async def test_validate_none_volume(self, use_case):
        """Test validation with None volume (valid)."""
        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("150.00"), volume=None)

        error = await use_case.validate(request)
        assert error is None

    @pytest.mark.asyncio
    async def test_process_no_active_orders(self, use_case, mock_unit_of_work):
        """Test processing when no active orders exist."""
        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("150.00"))

        mock_unit_of_work.orders.get_active_orders.return_value = []

        response = await use_case.process(request)

        assert response.success is True
        assert response.orders_triggered == []
        assert response.orders_filled == []
        assert response.request_id == request.request_id
        mock_unit_of_work.orders.get_active_orders.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_with_custom_timestamp(self, use_case, mock_unit_of_work):
        """Test processing with custom timestamp."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        request = UpdateMarketPriceRequest(
            symbol="AAPL", price=Decimal("150.00"), timestamp=timestamp
        )

        mock_unit_of_work.orders.get_active_orders.return_value = []

        response = await use_case.process(request)

        assert response.success is True

    @pytest.mark.asyncio
    async def test_process_trigger_market_order(
        self, use_case, mock_unit_of_work, mock_market_microstructure
    ):
        """Test processing triggers market order."""
        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("150.00"))

        # Create mock market order
        order = MagicMock()
        order.id = uuid4()
        order.symbol = Symbol("AAPL")
        order.order_type = OrderType.MARKET
        order.side = OrderSide.BUY
        order.quantity = Decimal("100")
        order.status = OrderStatus.PENDING
        order.is_active.return_value = True

        mock_unit_of_work.orders.get_active_orders.return_value = [order]
        mock_market_microstructure.calculate_execution_price.return_value = Price(Decimal("150.05"))

        response = await use_case.process(request)

        assert response.success is True
        assert order.id in response.orders_triggered
        assert order.id in response.orders_filled
        assert order.status == OrderStatus.FILLED
        assert order.average_fill_price == Price(Decimal("150.05"))
        assert order.filled_quantity == Decimal("100")
        assert order.filled_at is not None
        mock_unit_of_work.orders.update_order.assert_called_once_with(order)

    @pytest.mark.asyncio
    async def test_process_trigger_buy_limit_order(
        self, use_case, mock_unit_of_work, mock_market_microstructure
    ):
        """Test processing triggers buy limit order when price <= limit."""
        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("149.00"))

        # Create mock buy limit order
        order = MagicMock()
        order.id = uuid4()
        order.symbol = Symbol("AAPL")
        order.order_type = OrderType.LIMIT
        order.side = OrderSide.BUY
        order.quantity = Decimal("100")
        order.limit_price = Decimal("150.00")
        order.status = OrderStatus.PENDING
        order.is_active.return_value = True

        mock_unit_of_work.orders.get_active_orders.return_value = [order]

        response = await use_case.process(request)

        assert response.success is True
        assert order.id in response.orders_triggered
        assert order.id in response.orders_filled

    @pytest.mark.asyncio
    async def test_process_no_trigger_buy_limit_order(self, use_case, mock_unit_of_work):
        """Test buy limit order not triggered when price > limit."""
        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("151.00"))

        order = MagicMock()
        order.id = uuid4()
        order.symbol = Symbol("AAPL")
        order.order_type = OrderType.LIMIT
        order.side = OrderSide.BUY
        order.quantity = Decimal("100")
        order.limit_price = Decimal("150.00")
        order.is_active.return_value = True

        mock_unit_of_work.orders.get_active_orders.return_value = [order]

        response = await use_case.process(request)

        assert response.success is True
        assert order.id not in response.orders_triggered
        assert order.id not in response.orders_filled

    @pytest.mark.asyncio
    async def test_process_trigger_sell_limit_order(
        self, use_case, mock_unit_of_work, mock_market_microstructure
    ):
        """Test processing triggers sell limit order when price >= limit."""
        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("151.00"))

        order = MagicMock()
        order.id = uuid4()
        order.symbol = Symbol("AAPL")
        order.order_type = OrderType.LIMIT
        order.side = OrderSide.SELL
        order.quantity = Decimal("100")
        order.limit_price = Decimal("150.00")
        order.status = OrderStatus.PENDING
        order.is_active.return_value = True

        mock_unit_of_work.orders.get_active_orders.return_value = [order]

        response = await use_case.process(request)

        assert response.success is True
        assert order.id in response.orders_triggered
        assert order.id in response.orders_filled

    @pytest.mark.asyncio
    async def test_process_no_trigger_sell_limit_order(self, use_case, mock_unit_of_work):
        """Test sell limit order not triggered when price < limit."""
        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("149.00"))

        order = MagicMock()
        order.id = uuid4()
        order.symbol = Symbol("AAPL")
        order.order_type = OrderType.LIMIT
        order.side = OrderSide.SELL
        order.quantity = Decimal("100")
        order.limit_price = Decimal("150.00")
        order.is_active.return_value = True

        mock_unit_of_work.orders.get_active_orders.return_value = [order]

        response = await use_case.process(request)

        assert response.success is True
        assert order.id not in response.orders_triggered
        assert order.id not in response.orders_filled

    @pytest.mark.asyncio
    async def test_process_trigger_buy_stop_order(
        self, use_case, mock_unit_of_work, mock_market_microstructure
    ):
        """Test processing triggers buy stop order when price >= stop."""
        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("151.00"))

        order = MagicMock()
        order.id = uuid4()
        order.symbol = Symbol("AAPL")
        order.order_type = OrderType.STOP
        order.side = OrderSide.BUY
        order.quantity = Decimal("100")
        order.stop_price = Decimal("150.00")
        order.status = OrderStatus.PENDING
        order.is_active.return_value = True

        mock_unit_of_work.orders.get_active_orders.return_value = [order]

        # Stop orders are NOT limit orders, so they don't fill immediately in the current logic
        # They only get triggered. This matches the implementation.
        response = await use_case.process(request)

        assert response.success is True
        assert order.id in response.orders_triggered
        # Stop orders don't get filled immediately in current implementation
        assert order.id not in response.orders_filled

    @pytest.mark.asyncio
    async def test_process_no_trigger_buy_stop_order(self, use_case, mock_unit_of_work):
        """Test buy stop order not triggered when price < stop."""
        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("149.00"))

        order = MagicMock()
        order.id = uuid4()
        order.symbol = Symbol("AAPL")
        order.order_type = OrderType.STOP
        order.side = OrderSide.BUY
        order.quantity = Decimal("100")
        order.stop_price = Decimal("150.00")
        order.is_active.return_value = True

        mock_unit_of_work.orders.get_active_orders.return_value = [order]

        response = await use_case.process(request)

        assert response.success is True
        assert order.id not in response.orders_triggered
        assert order.id not in response.orders_filled

    @pytest.mark.asyncio
    async def test_process_trigger_sell_stop_order(
        self, use_case, mock_unit_of_work, mock_market_microstructure
    ):
        """Test processing triggers sell stop order when price <= stop."""
        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("149.00"))

        order = MagicMock()
        order.id = uuid4()
        order.symbol = Symbol("AAPL")
        order.order_type = OrderType.STOP
        order.side = OrderSide.SELL
        order.quantity = Decimal("100")
        order.stop_price = Decimal("150.00")
        order.status = OrderStatus.PENDING
        order.is_active.return_value = True

        mock_unit_of_work.orders.get_active_orders.return_value = [order]

        response = await use_case.process(request)

        assert response.success is True
        assert order.id in response.orders_triggered
        # Stop orders don't get filled immediately in current implementation
        assert order.id not in response.orders_filled

    @pytest.mark.asyncio
    async def test_process_no_trigger_sell_stop_order(self, use_case, mock_unit_of_work):
        """Test sell stop order not triggered when price > stop."""
        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("151.00"))

        order = MagicMock()
        order.id = uuid4()
        order.symbol = Symbol("AAPL")
        order.order_type = OrderType.STOP
        order.side = OrderSide.SELL
        order.quantity = Decimal("100")
        order.stop_price = Decimal("150.00")
        order.is_active.return_value = True

        mock_unit_of_work.orders.get_active_orders.return_value = [order]

        response = await use_case.process(request)

        assert response.success is True
        assert order.id not in response.orders_triggered
        assert order.id not in response.orders_filled

    @pytest.mark.asyncio
    async def test_process_stop_limit_buy_triggered(
        self, use_case, mock_unit_of_work, mock_market_microstructure
    ):
        """Test buy stop-limit order triggered when stop hit and price <= limit."""
        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("151.00"))

        order = MagicMock()
        order.id = uuid4()
        order.symbol = Symbol("AAPL")
        order.order_type = OrderType.STOP_LIMIT
        order.side = OrderSide.BUY
        order.quantity = Decimal("100")
        order.stop_price = Decimal("150.00")  # Stop triggered at 150
        order.limit_price = Decimal("152.00")  # Buy up to 152
        order.status = OrderStatus.PENDING
        order.is_active.return_value = True

        mock_unit_of_work.orders.get_active_orders.return_value = [order]

        response = await use_case.process(request)

        assert response.success is True
        assert order.id in response.orders_triggered
        # Stop-limit orders don't get filled immediately in current implementation
        assert order.id not in response.orders_filled

    @pytest.mark.asyncio
    async def test_process_stop_limit_buy_not_triggered_stop(self, use_case, mock_unit_of_work):
        """Test buy stop-limit order not triggered when stop not hit."""
        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("149.00"))

        order = MagicMock()
        order.id = uuid4()
        order.symbol = Symbol("AAPL")
        order.order_type = OrderType.STOP_LIMIT
        order.side = OrderSide.BUY
        order.quantity = Decimal("100")
        order.stop_price = Decimal("150.00")
        order.limit_price = Decimal("152.00")
        order.is_active.return_value = True

        mock_unit_of_work.orders.get_active_orders.return_value = [order]

        response = await use_case.process(request)

        assert response.success is True
        assert order.id not in response.orders_triggered
        assert order.id not in response.orders_filled

    @pytest.mark.asyncio
    async def test_process_stop_limit_buy_not_triggered_limit(self, use_case, mock_unit_of_work):
        """Test buy stop-limit order not triggered when stop hit but price > limit."""
        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("153.00"))

        order = MagicMock()
        order.id = uuid4()
        order.symbol = Symbol("AAPL")
        order.order_type = OrderType.STOP_LIMIT
        order.side = OrderSide.BUY
        order.quantity = Decimal("100")
        order.stop_price = Decimal("150.00")  # Stop triggered
        order.limit_price = Decimal("152.00")  # But price too high
        order.is_active.return_value = True

        mock_unit_of_work.orders.get_active_orders.return_value = [order]

        response = await use_case.process(request)

        assert response.success is True
        assert order.id not in response.orders_triggered
        assert order.id not in response.orders_filled

    @pytest.mark.asyncio
    async def test_process_stop_limit_sell_triggered(
        self, use_case, mock_unit_of_work, mock_market_microstructure
    ):
        """Test sell stop-limit order triggered when stop hit and price >= limit."""
        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("149.00"))

        order = MagicMock()
        order.id = uuid4()
        order.symbol = Symbol("AAPL")
        order.order_type = OrderType.STOP_LIMIT
        order.side = OrderSide.SELL
        order.quantity = Decimal("100")
        order.stop_price = Decimal("150.00")  # Stop triggered at 150
        order.limit_price = Decimal("148.00")  # Sell at 148 or higher
        order.status = OrderStatus.PENDING
        order.is_active.return_value = True

        mock_unit_of_work.orders.get_active_orders.return_value = [order]

        response = await use_case.process(request)

        assert response.success is True
        assert order.id in response.orders_triggered
        # Stop-limit orders don't get filled immediately in current implementation
        assert order.id not in response.orders_filled

    @pytest.mark.asyncio
    async def test_process_stop_limit_sell_not_triggered_stop(self, use_case, mock_unit_of_work):
        """Test sell stop-limit order not triggered when stop not hit."""
        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("151.00"))

        order = MagicMock()
        order.id = uuid4()
        order.symbol = Symbol("AAPL")
        order.order_type = OrderType.STOP_LIMIT
        order.side = OrderSide.SELL
        order.quantity = Decimal("100")
        order.stop_price = Decimal("150.00")
        order.limit_price = Decimal("148.00")
        order.is_active.return_value = True

        mock_unit_of_work.orders.get_active_orders.return_value = [order]

        response = await use_case.process(request)

        assert response.success is True
        assert order.id not in response.orders_triggered
        assert order.id not in response.orders_filled

    @pytest.mark.asyncio
    async def test_process_stop_limit_sell_not_triggered_limit(self, use_case, mock_unit_of_work):
        """Test sell stop-limit order not triggered when stop hit but price < limit."""
        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("147.00"))

        order = MagicMock()
        order.id = uuid4()
        order.symbol = Symbol("AAPL")
        order.order_type = OrderType.STOP_LIMIT
        order.side = OrderSide.SELL
        order.quantity = Decimal("100")
        order.stop_price = Decimal("150.00")  # Stop triggered
        order.limit_price = Decimal("148.00")  # But price too low
        order.is_active.return_value = True

        mock_unit_of_work.orders.get_active_orders.return_value = [order]

        response = await use_case.process(request)

        assert response.success is True
        assert order.id not in response.orders_triggered
        assert order.id not in response.orders_filled

    @pytest.mark.asyncio
    async def test_process_ignores_different_symbol(self, use_case, mock_unit_of_work):
        """Test processing ignores orders with different symbols."""
        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("150.00"))

        order = MagicMock()
        order.id = uuid4()
        order.symbol = Symbol("MSFT")
        order.order_type = OrderType.MARKET
        order.is_active.return_value = True

        mock_unit_of_work.orders.get_active_orders.return_value = [order]

        response = await use_case.process(request)

        assert response.success is True
        assert order.id not in response.orders_triggered
        assert order.id not in response.orders_filled

    @pytest.mark.asyncio
    async def test_process_ignores_inactive_orders(self, use_case, mock_unit_of_work):
        """Test processing ignores inactive orders."""
        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("150.00"))

        order = MagicMock()
        order.id = uuid4()
        order.symbol = Symbol("AAPL")
        order.order_type = OrderType.MARKET
        order.is_active.return_value = False

        mock_unit_of_work.orders.get_active_orders.return_value = [order]

        response = await use_case.process(request)

        assert response.success is True
        assert order.id not in response.orders_triggered
        assert order.id not in response.orders_filled

    @pytest.mark.asyncio
    async def test_process_multiple_orders(
        self, use_case, mock_unit_of_work, mock_market_microstructure
    ):
        """Test processing multiple orders simultaneously."""
        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("150.00"))

        # Create multiple orders
        orders = []
        for i in range(3):
            order = MagicMock()
            order.id = uuid4()
            order.symbol = Symbol("AAPL")
            order.order_type = OrderType.MARKET
            order.side = OrderSide.BUY
            order.quantity = Decimal("100")
            order.status = OrderStatus.PENDING
            order.is_active.return_value = True
            orders.append(order)

        mock_unit_of_work.orders.get_active_orders.return_value = orders

        response = await use_case.process(request)

        assert response.success is True
        assert len(response.orders_triggered) == 3
        assert len(response.orders_filled) == 3
        assert mock_unit_of_work.orders.update_order.call_count == 3

    def test_should_trigger_order_inactive(self, use_case):
        """Test _should_trigger_order with inactive order."""
        order = MagicMock()
        order.is_active.return_value = False

        should_trigger, reason = use_case._should_trigger_order(order, Price(Decimal("150.00")))

        assert should_trigger is False
        assert reason is None

    def test_should_trigger_order_market(self, use_case):
        """Test _should_trigger_order with market order."""
        order = MagicMock()
        order.is_active.return_value = True
        order.order_type = OrderType.MARKET

        should_trigger, reason = use_case._should_trigger_order(order, Price(Decimal("150.00")))

        assert should_trigger is True
        assert "Market order triggers immediately" in reason

    def test_should_trigger_order_limit_without_price(self, use_case):
        """Test _should_trigger_order with limit order without limit price."""
        order = MagicMock()
        order.is_active.return_value = True
        order.order_type = OrderType.LIMIT
        order.limit_price = None

        should_trigger, reason = use_case._should_trigger_order(order, Price(Decimal("150.00")))

        assert should_trigger is False
        assert reason is None

    def test_should_trigger_order_stop_without_price(self, use_case):
        """Test _should_trigger_order with stop order without stop price."""
        order = MagicMock()
        order.is_active.return_value = True
        order.order_type = OrderType.STOP
        order.stop_price = None

        should_trigger, reason = use_case._should_trigger_order(order, Price(Decimal("150.00")))

        assert should_trigger is False
        assert reason is None

    def test_should_trigger_order_stop_limit_without_stop(self, use_case):
        """Test _should_trigger_order with stop-limit order without stop price."""
        order = MagicMock()
        order.is_active.return_value = True
        order.order_type = OrderType.STOP_LIMIT
        order.stop_price = None
        order.limit_price = Decimal("150.00")

        should_trigger, reason = use_case._should_trigger_order(order, Price(Decimal("150.00")))

        assert should_trigger is False
        assert reason is None

    def test_should_trigger_order_stop_limit_without_limit(self, use_case):
        """Test _should_trigger_order with stop-limit order without limit price."""
        order = MagicMock()
        order.is_active.return_value = True
        order.order_type = OrderType.STOP_LIMIT
        order.side = OrderSide.BUY
        order.stop_price = Decimal("150.00")
        order.limit_price = None

        should_trigger, reason = use_case._should_trigger_order(order, Price(Decimal("151.00")))

        assert should_trigger is False
        assert reason is None


# =============================================================================
# ProcessPendingOrdersUseCase Tests
# =============================================================================


class TestProcessPendingOrdersUseCase:
    """Test ProcessPendingOrdersUseCase."""

    @pytest.fixture
    def mock_unit_of_work(self):
        """Create mock unit of work."""
        mock_uow = AsyncMock()
        mock_uow.orders = AsyncMock()
        mock_uow.commit = AsyncMock()
        mock_uow.rollback = AsyncMock()
        return mock_uow

    @pytest.fixture
    def mock_order_processor(self):
        """Create mock order processor."""
        return MagicMock()

    @pytest.fixture
    def mock_market_microstructure(self):
        """Create mock market microstructure."""
        mock_mm = MagicMock()
        mock_mm.calculate_execution_price = MagicMock(return_value=Price(Decimal("150.00")))
        return mock_mm

    @pytest.fixture
    def use_case(self, mock_unit_of_work, mock_order_processor, mock_market_microstructure):
        """Create use case instance."""
        return ProcessPendingOrdersUseCase(
            unit_of_work=mock_unit_of_work,
            order_processor=mock_order_processor,
            market_microstructure=mock_market_microstructure,
        )

    def test_init(self, mock_unit_of_work, mock_order_processor, mock_market_microstructure):
        """Test use case initialization."""
        use_case = ProcessPendingOrdersUseCase(
            unit_of_work=mock_unit_of_work,
            order_processor=mock_order_processor,
            market_microstructure=mock_market_microstructure,
        )

        assert use_case.unit_of_work == mock_unit_of_work
        assert use_case.order_processor == mock_order_processor
        assert use_case.market_microstructure == mock_market_microstructure
        assert use_case.name == "ProcessPendingOrdersUseCase"
        assert isinstance(use_case.logger, logging.Logger)

    @pytest.mark.asyncio
    async def test_validate_always_passes(self, use_case):
        """Test validation always passes for this use case."""
        request = ProcessPendingOrdersRequest()
        error = await use_case.validate(request)
        assert error is None

    @pytest.mark.asyncio
    async def test_process_no_orders(self, use_case, mock_unit_of_work):
        """Test processing when no orders exist."""
        request = ProcessPendingOrdersRequest(current_prices={"AAPL": Decimal("150.00")})

        mock_unit_of_work.orders.get_active_orders.return_value = []

        response = await use_case.process(request)

        assert response.success is True
        assert response.processed_count == 0
        assert response.triggered_orders == []
        assert response.filled_orders == []
        assert response.request_id == request.request_id

    @pytest.mark.asyncio
    async def test_process_filter_by_symbol(self, use_case, mock_unit_of_work):
        """Test processing filters orders by symbol."""
        request = ProcessPendingOrdersRequest(
            symbol="AAPL", current_prices={"AAPL": Decimal("150.00"), "MSFT": Decimal("300.00")}
        )

        # Create orders for different symbols
        order1 = MagicMock()
        order1.id = uuid4()
        order1.symbol = Symbol("AAPL")
        order1.order_type = OrderType.MARKET
        order1.side = OrderSide.BUY
        order1.quantity = Decimal("100")
        order1.is_active.return_value = True

        order2 = MagicMock()
        order2.id = uuid4()
        order2.symbol = Symbol("MSFT")
        order2.order_type = OrderType.MARKET
        order2.is_active.return_value = True

        mock_unit_of_work.orders.get_active_orders.return_value = [order1, order2]

        response = await use_case.process(request)

        assert response.success is True
        assert response.processed_count == 1
        assert order1.id in response.triggered_orders
        assert order2.id not in response.triggered_orders

    @pytest.mark.asyncio
    async def test_process_no_symbol_filter(self, use_case, mock_unit_of_work):
        """Test processing without symbol filter processes all orders."""
        request = ProcessPendingOrdersRequest(
            current_prices={"AAPL": Decimal("150.00"), "MSFT": Decimal("300.00")}
        )

        order1 = MagicMock()
        order1.id = uuid4()
        order1.symbol = Symbol("AAPL")
        order1.order_type = OrderType.MARKET
        order1.side = OrderSide.BUY
        order1.quantity = Decimal("100")
        order1.is_active.return_value = True

        order2 = MagicMock()
        order2.id = uuid4()
        order2.symbol = Symbol("MSFT")
        order2.order_type = OrderType.MARKET
        order2.side = OrderSide.BUY
        order2.quantity = Decimal("50")
        order2.is_active.return_value = True

        mock_unit_of_work.orders.get_active_orders.return_value = [order1, order2]

        response = await use_case.process(request)

        assert response.success is True
        assert response.processed_count == 2
        assert order1.id in response.triggered_orders
        assert order2.id in response.triggered_orders

    @pytest.mark.asyncio
    async def test_process_no_price_for_symbol(self, use_case, mock_unit_of_work):
        """Test processing skips orders without price data."""
        request = ProcessPendingOrdersRequest(current_prices={"MSFT": Decimal("300.00")})

        order = MagicMock()
        order.id = uuid4()
        order.symbol = Symbol("AAPL")
        order.order_type = OrderType.MARKET
        order.is_active.return_value = True

        mock_unit_of_work.orders.get_active_orders.return_value = [order]

        response = await use_case.process(request)

        assert response.success is True
        assert response.processed_count == 1
        assert order.id not in response.triggered_orders
        assert order.id not in response.filled_orders

    @pytest.mark.asyncio
    async def test_process_no_current_prices(self, use_case, mock_unit_of_work):
        """Test processing with no current prices provided."""
        request = ProcessPendingOrdersRequest()

        order = MagicMock()
        order.id = uuid4()
        order.symbol = Symbol("AAPL")
        order.order_type = OrderType.MARKET
        order.is_active.return_value = True

        mock_unit_of_work.orders.get_active_orders.return_value = [order]

        response = await use_case.process(request)

        assert response.success is True
        assert response.processed_count == 1
        assert order.id not in response.triggered_orders
        assert order.id not in response.filled_orders

    @pytest.mark.asyncio
    async def test_process_market_order(
        self, use_case, mock_unit_of_work, mock_market_microstructure
    ):
        """Test processing market order gets triggered and filled."""
        request = ProcessPendingOrdersRequest(current_prices={"AAPL": Decimal("150.00")})

        order = MagicMock()
        order.id = uuid4()
        order.symbol = Symbol("AAPL")
        order.order_type = OrderType.MARKET
        order.side = OrderSide.BUY
        order.quantity = Decimal("100")
        order.is_active.return_value = True

        mock_unit_of_work.orders.get_active_orders.return_value = [order]

        response = await use_case.process(request)

        assert response.success is True
        assert response.processed_count == 1
        assert order.id in response.triggered_orders
        assert order.id in response.filled_orders
        assert order.status == OrderStatus.FILLED
        assert order.filled_at is not None
        mock_unit_of_work.orders.update_order.assert_called_once_with(order)

    @pytest.mark.asyncio
    async def test_process_limit_order_conditions(
        self, use_case, mock_unit_of_work, mock_market_microstructure
    ):
        """Test processing correctly evaluates limit order conditions."""
        request = ProcessPendingOrdersRequest(current_prices={"AAPL": Decimal("149.00")})

        # Buy limit - should trigger (price <= limit)
        buy_limit = MagicMock()
        buy_limit.id = uuid4()
        buy_limit.symbol = Symbol("AAPL")
        buy_limit.order_type = OrderType.LIMIT
        buy_limit.side = OrderSide.BUY
        buy_limit.limit_price = Decimal("150.00")
        buy_limit.quantity = Decimal("100")
        buy_limit.is_active.return_value = True

        # Sell limit - should not trigger (price < limit)
        sell_limit = MagicMock()
        sell_limit.id = uuid4()
        sell_limit.symbol = Symbol("AAPL")
        sell_limit.order_type = OrderType.LIMIT
        sell_limit.side = OrderSide.SELL
        sell_limit.limit_price = Decimal("150.00")
        sell_limit.quantity = Decimal("100")
        sell_limit.is_active.return_value = True

        mock_unit_of_work.orders.get_active_orders.return_value = [buy_limit, sell_limit]

        response = await use_case.process(request)

        assert response.success is True
        assert response.processed_count == 2
        assert buy_limit.id in response.triggered_orders
        assert sell_limit.id not in response.triggered_orders

    @pytest.mark.asyncio
    async def test_process_stop_order_conditions(
        self, use_case, mock_unit_of_work, mock_market_microstructure
    ):
        """Test processing correctly evaluates stop order conditions."""
        request = ProcessPendingOrdersRequest(current_prices={"AAPL": Decimal("151.00")})

        # Buy stop - should trigger (price >= stop)
        buy_stop = MagicMock()
        buy_stop.id = uuid4()
        buy_stop.symbol = Symbol("AAPL")
        buy_stop.order_type = OrderType.STOP
        buy_stop.side = OrderSide.BUY
        buy_stop.stop_price = Decimal("150.00")
        buy_stop.quantity = Decimal("100")
        buy_stop.is_active.return_value = True

        # Sell stop - should not trigger (price > stop)
        sell_stop = MagicMock()
        sell_stop.id = uuid4()
        sell_stop.symbol = Symbol("AAPL")
        sell_stop.order_type = OrderType.STOP
        sell_stop.side = OrderSide.SELL
        sell_stop.stop_price = Decimal("150.00")
        sell_stop.quantity = Decimal("100")
        sell_stop.is_active.return_value = True

        mock_unit_of_work.orders.get_active_orders.return_value = [buy_stop, sell_stop]

        response = await use_case.process(request)

        assert response.success is True
        assert response.processed_count == 2
        assert buy_stop.id in response.triggered_orders
        assert sell_stop.id not in response.triggered_orders

    @pytest.mark.asyncio
    async def test_process_multiple_orders(
        self, use_case, mock_unit_of_work, mock_market_microstructure
    ):
        """Test processing multiple orders simultaneously."""
        request = ProcessPendingOrdersRequest(
            current_prices={"AAPL": Decimal("150.00"), "MSFT": Decimal("300.00")}
        )

        orders = []
        for i in range(5):
            order = MagicMock()
            order.id = uuid4()
            order.symbol = Symbol("AAPL" if i < 3 else "MSFT")
            order.order_type = OrderType.MARKET
            order.side = OrderSide.BUY
            order.quantity = Decimal("100")
            order.is_active.return_value = True
            orders.append(order)

        mock_unit_of_work.orders.get_active_orders.return_value = orders

        response = await use_case.process(request)

        assert response.success is True
        assert response.processed_count == 5
        assert len(response.triggered_orders) == 5
        assert len(response.filled_orders) == 5
        assert mock_unit_of_work.orders.update_order.call_count == 5

    def test_should_trigger_order_inactive(self, use_case):
        """Test _should_trigger_order with inactive order."""
        order = MagicMock()
        order.is_active.return_value = False

        should_trigger, reason = use_case._should_trigger_order(order, Price(Decimal("150.00")))

        assert should_trigger is False
        assert reason is None

    def test_should_trigger_order_market(self, use_case):
        """Test _should_trigger_order with market order."""
        order = MagicMock()
        order.is_active.return_value = True
        order.order_type = OrderType.MARKET

        should_trigger, reason = use_case._should_trigger_order(order, Price(Decimal("150.00")))

        assert should_trigger is True
        assert reason == "Market order"

    def test_should_trigger_order_unknown_type(self, use_case):
        """Test _should_trigger_order with unknown order type."""
        order = MagicMock()
        order.is_active.return_value = True
        order.order_type = MagicMock()  # Unknown type

        should_trigger, reason = use_case._should_trigger_order(order, Price(Decimal("150.00")))

        assert should_trigger is False
        assert reason is None

    def test_should_trigger_order_limit_sell(self, use_case):
        """Test _should_trigger_order with sell limit order."""
        order = MagicMock()
        order.is_active.return_value = True
        order.order_type = OrderType.LIMIT
        order.side = OrderSide.SELL
        order.limit_price = Decimal("150.00")

        # Should trigger when price >= limit
        should_trigger, reason = use_case._should_trigger_order(order, Price(Decimal("151.00")))

        assert should_trigger is True
        assert reason == "Sell limit triggered"

        # Should not trigger when price < limit
        should_trigger, reason = use_case._should_trigger_order(order, Price(Decimal("149.00")))

        assert should_trigger is False
        assert reason is None

    def test_should_trigger_order_stop_sell(self, use_case):
        """Test _should_trigger_order with sell stop order."""
        order = MagicMock()
        order.is_active.return_value = True
        order.order_type = OrderType.STOP
        order.side = OrderSide.SELL
        order.stop_price = Decimal("150.00")

        # Should trigger when price <= stop
        should_trigger, reason = use_case._should_trigger_order(order, Price(Decimal("149.00")))

        assert should_trigger is True
        assert reason == "Sell stop triggered"

        # Should not trigger when price > stop
        should_trigger, reason = use_case._should_trigger_order(order, Price(Decimal("151.00")))

        assert should_trigger is False
        assert reason is None


# =============================================================================
# CheckOrderTriggerUseCase Tests
# =============================================================================


class TestCheckOrderTriggerUseCase:
    """Test CheckOrderTriggerUseCase."""

    @pytest.fixture
    def mock_unit_of_work(self):
        """Create mock unit of work."""
        mock_uow = AsyncMock()
        mock_uow.orders = AsyncMock()
        return mock_uow

    @pytest.fixture
    def use_case(self, mock_unit_of_work):
        """Create use case instance."""
        return CheckOrderTriggerUseCase(unit_of_work=mock_unit_of_work)

    def test_init(self, mock_unit_of_work):
        """Test use case initialization."""
        use_case = CheckOrderTriggerUseCase(unit_of_work=mock_unit_of_work)

        assert use_case.unit_of_work == mock_unit_of_work
        assert use_case.name == "CheckOrderTriggerUseCase"
        assert isinstance(use_case.logger, logging.Logger)

    @pytest.mark.asyncio
    async def test_validate_valid_request(self, use_case):
        """Test validation with valid request."""
        request = CheckOrderTriggerRequest(order_id=uuid4(), current_price=Decimal("150.00"))

        error = await use_case.validate(request)
        assert error is None

    @pytest.mark.asyncio
    async def test_validate_zero_price(self, use_case):
        """Test validation with zero price."""
        request = CheckOrderTriggerRequest(order_id=uuid4(), current_price=Decimal("0"))

        error = await use_case.validate(request)
        assert error == "Current price must be positive"

    @pytest.mark.asyncio
    async def test_validate_negative_price(self, use_case):
        """Test validation with negative price."""
        request = CheckOrderTriggerRequest(order_id=uuid4(), current_price=Decimal("-10.00"))

        error = await use_case.validate(request)
        assert error == "Current price must be positive"

    @pytest.mark.asyncio
    async def test_process_order_not_found(self, use_case, mock_unit_of_work):
        """Test processing when order is not found."""
        request_id = uuid4()
        request = CheckOrderTriggerRequest(
            order_id=uuid4(), current_price=Decimal("150.00"), request_id=request_id
        )

        mock_unit_of_work.orders.get_order_by_id.return_value = None

        response = await use_case.process(request)

        assert response.success is False
        assert response.error == "Order not found"
        assert response.should_trigger is False
        assert response.request_id == request_id

    @pytest.mark.asyncio
    async def test_process_order_not_found_no_request_id(self, use_case, mock_unit_of_work):
        """Test processing when order is not found and no request_id provided."""
        request = CheckOrderTriggerRequest(
            order_id=uuid4(), current_price=Decimal("150.00"), request_id=None
        )

        mock_unit_of_work.orders.get_order_by_id.return_value = None

        response = await use_case.process(request)

        assert response.success is False
        assert response.error == "Order not found"
        assert response.should_trigger is False
        assert response.request_id is not None  # Should generate one

    @pytest.mark.asyncio
    async def test_process_inactive_order(self, use_case, mock_unit_of_work):
        """Test processing with inactive order."""
        request = CheckOrderTriggerRequest(order_id=uuid4(), current_price=Decimal("150.00"))

        order = MagicMock()
        order.is_active.return_value = False
        order.status = OrderStatus.FILLED

        mock_unit_of_work.orders.get_order_by_id.return_value = order

        response = await use_case.process(request)

        assert response.success is True
        assert response.should_trigger is False
        assert "not active" in response.reason
        assert "filled" in response.reason.lower()

    @pytest.mark.asyncio
    async def test_process_market_order(self, use_case, mock_unit_of_work):
        """Test processing market order always triggers."""
        request = CheckOrderTriggerRequest(order_id=uuid4(), current_price=Decimal("150.00"))

        order = MagicMock()
        order.order_type = OrderType.MARKET
        order.is_active.return_value = True

        mock_unit_of_work.orders.get_order_by_id.return_value = order

        response = await use_case.process(request)

        assert response.success is True
        assert response.should_trigger is True
        assert response.trigger_price == Decimal("150.00")
        assert "Market order executes immediately" in response.reason

    @pytest.mark.asyncio
    async def test_process_buy_limit_triggered(self, use_case, mock_unit_of_work):
        """Test buy limit order triggers when price <= limit."""
        request = CheckOrderTriggerRequest(order_id=uuid4(), current_price=Decimal("149.00"))

        order = MagicMock()
        order.order_type = OrderType.LIMIT
        order.side = OrderSide.BUY
        order.limit_price = MagicMock()
        order.limit_price = Decimal("150.00")
        order.is_active.return_value = True

        mock_unit_of_work.orders.get_order_by_id.return_value = order

        response = await use_case.process(request)

        assert response.success is True
        assert response.should_trigger is True
        assert response.trigger_price == Decimal("150.00")
        assert "Buy limit" in response.reason
        assert "149" in response.reason and "<= limit 150" in response.reason

    @pytest.mark.asyncio
    async def test_process_buy_limit_not_triggered(self, use_case, mock_unit_of_work):
        """Test buy limit order doesn't trigger when price > limit."""
        request = CheckOrderTriggerRequest(order_id=uuid4(), current_price=Decimal("151.00"))

        order = MagicMock()
        order.order_type = OrderType.LIMIT
        order.side = OrderSide.BUY
        order.limit_price = MagicMock()
        order.limit_price = Decimal("150.00")
        order.is_active.return_value = True

        mock_unit_of_work.orders.get_order_by_id.return_value = order

        response = await use_case.process(request)

        assert response.success is True
        assert response.should_trigger is False
        assert response.trigger_price is None
        assert "Buy limit not triggered" in response.reason
        assert "151" in response.reason and "> limit 150" in response.reason

    @pytest.mark.asyncio
    async def test_process_sell_limit_triggered(self, use_case, mock_unit_of_work):
        """Test sell limit order triggers when price >= limit."""
        request = CheckOrderTriggerRequest(order_id=uuid4(), current_price=Decimal("151.00"))

        order = MagicMock()
        order.order_type = OrderType.LIMIT
        order.side = OrderSide.SELL
        order.limit_price = MagicMock()
        order.limit_price = Decimal("150.00")
        order.is_active.return_value = True

        mock_unit_of_work.orders.get_order_by_id.return_value = order

        response = await use_case.process(request)

        assert response.success is True
        assert response.should_trigger is True
        assert response.trigger_price == Decimal("150.00")
        assert "Sell limit" in response.reason
        assert "151" in response.reason and ">= limit 150" in response.reason

    @pytest.mark.asyncio
    async def test_process_sell_limit_not_triggered(self, use_case, mock_unit_of_work):
        """Test sell limit order doesn't trigger when price < limit."""
        request = CheckOrderTriggerRequest(order_id=uuid4(), current_price=Decimal("149.00"))

        order = MagicMock()
        order.order_type = OrderType.LIMIT
        order.side = OrderSide.SELL
        order.limit_price = MagicMock()
        order.limit_price = Decimal("150.00")
        order.is_active.return_value = True

        mock_unit_of_work.orders.get_order_by_id.return_value = order

        response = await use_case.process(request)

        assert response.success is True
        assert response.should_trigger is False
        assert response.trigger_price is None
        assert "Sell limit not triggered" in response.reason
        assert "149" in response.reason and "< limit 150" in response.reason

    @pytest.mark.asyncio
    async def test_process_buy_stop_triggered(self, use_case, mock_unit_of_work):
        """Test buy stop order triggers when price >= stop."""
        request = CheckOrderTriggerRequest(order_id=uuid4(), current_price=Decimal("151.00"))

        order = MagicMock()
        order.order_type = OrderType.STOP
        order.side = OrderSide.BUY
        order.stop_price = MagicMock()
        order.stop_price = Decimal("150.00")
        order.is_active.return_value = True

        mock_unit_of_work.orders.get_order_by_id.return_value = order

        response = await use_case.process(request)

        assert response.success is True
        assert response.should_trigger is True
        assert response.trigger_price == Decimal("151.00")
        assert "Buy stop" in response.reason
        assert "151" in response.reason and ">= stop 150" in response.reason

    @pytest.mark.asyncio
    async def test_process_buy_stop_not_triggered(self, use_case, mock_unit_of_work):
        """Test buy stop order doesn't trigger when price < stop."""
        request = CheckOrderTriggerRequest(order_id=uuid4(), current_price=Decimal("149.00"))

        order = MagicMock()
        order.order_type = OrderType.STOP
        order.side = OrderSide.BUY
        order.stop_price = MagicMock()
        order.stop_price = Decimal("150.00")
        order.is_active.return_value = True

        mock_unit_of_work.orders.get_order_by_id.return_value = order

        response = await use_case.process(request)

        assert response.success is True
        assert response.should_trigger is False
        assert response.trigger_price is None
        assert "Buy stop not triggered" in response.reason
        assert "149" in response.reason and "< stop 150" in response.reason

    @pytest.mark.asyncio
    async def test_process_sell_stop_triggered(self, use_case, mock_unit_of_work):
        """Test sell stop order triggers when price <= stop."""
        request = CheckOrderTriggerRequest(order_id=uuid4(), current_price=Decimal("149.00"))

        order = MagicMock()
        order.order_type = OrderType.STOP
        order.side = OrderSide.SELL
        order.stop_price = MagicMock()
        order.stop_price = Decimal("150.00")
        order.is_active.return_value = True

        mock_unit_of_work.orders.get_order_by_id.return_value = order

        response = await use_case.process(request)

        assert response.success is True
        assert response.should_trigger is True
        assert response.trigger_price == Decimal("149.00")
        assert "Sell stop" in response.reason
        assert "149" in response.reason and "<= stop 150" in response.reason

    @pytest.mark.asyncio
    async def test_process_sell_stop_not_triggered(self, use_case, mock_unit_of_work):
        """Test sell stop order doesn't trigger when price > stop."""
        request = CheckOrderTriggerRequest(order_id=uuid4(), current_price=Decimal("151.00"))

        order = MagicMock()
        order.order_type = OrderType.STOP
        order.side = OrderSide.SELL
        order.stop_price = MagicMock()
        order.stop_price = Decimal("150.00")
        order.is_active.return_value = True

        mock_unit_of_work.orders.get_order_by_id.return_value = order

        response = await use_case.process(request)

        assert response.success is True
        assert response.should_trigger is False
        assert response.trigger_price is None
        assert "Sell stop not triggered" in response.reason
        assert "151" in response.reason and "> stop 150" in response.reason

    @pytest.mark.asyncio
    async def test_process_limit_order_without_limit_price(self, use_case, mock_unit_of_work):
        """Test processing limit order without limit price."""
        request = CheckOrderTriggerRequest(order_id=uuid4(), current_price=Decimal("150.00"))

        order = MagicMock()
        order.order_type = OrderType.LIMIT
        order.side = OrderSide.BUY
        order.limit_price = None
        order.is_active.return_value = True

        mock_unit_of_work.orders.get_order_by_id.return_value = order

        response = await use_case.process(request)

        assert response.success is True
        assert response.should_trigger is False
        assert response.trigger_price is None
        assert response.reason is None

    @pytest.mark.asyncio
    async def test_process_stop_order_without_stop_price(self, use_case, mock_unit_of_work):
        """Test processing stop order without stop price."""
        request = CheckOrderTriggerRequest(order_id=uuid4(), current_price=Decimal("150.00"))

        order = MagicMock()
        order.order_type = OrderType.STOP
        order.side = OrderSide.BUY
        order.stop_price = None
        order.is_active.return_value = True

        mock_unit_of_work.orders.get_order_by_id.return_value = order

        response = await use_case.process(request)

        assert response.success is True
        assert response.should_trigger is False
        assert response.trigger_price is None
        assert response.reason is None

    @pytest.mark.asyncio
    async def test_process_unknown_order_type(self, use_case, mock_unit_of_work):
        """Test processing order with unknown type."""
        request = CheckOrderTriggerRequest(order_id=uuid4(), current_price=Decimal("150.00"))

        order = MagicMock()
        order.order_type = MagicMock()  # Unknown type
        order.is_active.return_value = True

        mock_unit_of_work.orders.get_order_by_id.return_value = order

        response = await use_case.process(request)

        assert response.success is True
        assert response.should_trigger is False
        assert response.trigger_price is None
        assert response.reason is None


# =============================================================================
# Response DTO Tests
# =============================================================================


class TestResponseDTOs:
    """Test response DTOs."""

    def test_update_market_price_response_defaults(self):
        """Test UpdateMarketPriceResponse with defaults."""
        response = UpdateMarketPriceResponse(success=True)

        assert response.success is True
        assert response.orders_triggered == []
        assert response.orders_filled == []

    def test_update_market_price_response_with_data(self):
        """Test UpdateMarketPriceResponse with data."""
        triggered = [uuid4(), uuid4()]
        filled = [uuid4()]

        response = UpdateMarketPriceResponse(
            success=True, orders_triggered=triggered, orders_filled=filled
        )

        assert response.success is True
        assert response.orders_triggered == triggered
        assert response.orders_filled == filled

    def test_process_pending_orders_response_defaults(self):
        """Test ProcessPendingOrdersResponse with defaults."""
        response = ProcessPendingOrdersResponse(success=True)

        assert response.success is True
        assert response.processed_count == 0
        assert response.triggered_orders == []
        assert response.filled_orders == []

    def test_process_pending_orders_response_with_data(self):
        """Test ProcessPendingOrdersResponse with data."""
        triggered = [uuid4(), uuid4()]
        filled = [uuid4()]

        response = ProcessPendingOrdersResponse(
            success=True, processed_count=5, triggered_orders=triggered, filled_orders=filled
        )

        assert response.success is True
        assert response.processed_count == 5
        assert response.triggered_orders == triggered
        assert response.filled_orders == filled

    def test_check_order_trigger_response_defaults(self):
        """Test CheckOrderTriggerResponse with defaults."""
        response = CheckOrderTriggerResponse(success=True)

        assert response.success is True
        assert response.should_trigger is False
        assert response.trigger_price is None
        assert response.reason is None

    def test_check_order_trigger_response_with_data(self):
        """Test CheckOrderTriggerResponse with data."""
        response = CheckOrderTriggerResponse(
            success=True,
            should_trigger=True,
            trigger_price=Decimal("150.00"),
            reason="Market order executes immediately",
        )

        assert response.success is True
        assert response.should_trigger is True
        assert response.trigger_price == Decimal("150.00")
        assert response.reason == "Market order executes immediately"
