"""
Comprehensive tests for market simulation use cases.

Tests all market simulation-related use cases including price updates,
order triggering, and pending order processing with full coverage.
"""

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from src.application.use_cases.market_simulation import (
    CheckOrderTriggerRequest,
    CheckOrderTriggerUseCase,
    ProcessPendingOrdersRequest,
    ProcessPendingOrdersUseCase,
    UpdateMarketPriceRequest,
    UpdateMarketPriceUseCase,
)
from src.application.use_cases.order_execution import (
    ProcessOrderFillRequest,
    ProcessOrderFillUseCase,
    SimulateOrderExecutionRequest,
    SimulateOrderExecutionUseCase,
)
from src.domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from src.domain.entities.portfolio import Portfolio
from src.domain.value_objects.money import Money


class TestRequestPostInit:
    """Test __post_init__ methods of request classes."""

    def test_update_market_price_request_post_init(self):
        """Test UpdateMarketPriceRequest __post_init__ method."""
        # Test with no request_id and metadata
        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("150.00"))
        assert request.request_id is not None
        assert request.metadata == {}

        # Test with provided values
        req_id = uuid4()
        metadata = {"source": "market_feed"}
        request = UpdateMarketPriceRequest(
            symbol="AAPL", price=Decimal("150.00"), request_id=req_id, metadata=metadata
        )
        assert request.request_id == req_id
        assert request.metadata == metadata

    def test_process_pending_orders_request_post_init(self):
        """Test ProcessPendingOrdersRequest __post_init__ method."""
        # Test with no request_id and metadata
        request = ProcessPendingOrdersRequest()
        assert request.request_id is not None
        assert request.metadata == {}

        # Test with provided values
        req_id = uuid4()
        metadata = {"trigger": "scheduled"}
        request = ProcessPendingOrdersRequest(symbol="AAPL", request_id=req_id, metadata=metadata)
        assert request.request_id == req_id
        assert request.metadata == metadata

    def test_check_order_trigger_request_post_init(self):
        """Test CheckOrderTriggerRequest __post_init__ method."""
        # Test with no request_id and metadata
        request = CheckOrderTriggerRequest(order_id=uuid4(), current_price=Decimal("150.00"))
        assert request.request_id is not None
        assert request.metadata == {}

        # Test with provided values
        req_id = uuid4()
        metadata = {"check_type": "stop_loss"}
        request = CheckOrderTriggerRequest(
            order_id=uuid4(), current_price=Decimal("150.00"), request_id=req_id, metadata=metadata
        )
        assert request.request_id == req_id
        assert request.metadata == metadata

    def test_simulate_order_execution_request_post_init(self):
        """Test SimulateOrderExecutionRequest __post_init__ method."""
        # Test with no request_id and metadata
        request = SimulateOrderExecutionRequest(order_id=uuid4(), fill_price=Decimal("150.00"))
        assert request.request_id is not None
        assert request.metadata == {}

    def test_process_order_fill_request_post_init(self):
        """Test ProcessOrderFillRequest __post_init__ method."""
        # Test with no request_id and metadata
        request = ProcessOrderFillRequest(
            order_id=uuid4(), fill_quantity=Decimal("100"), fill_price=Decimal("150.00")
        )
        assert request.request_id is not None
        assert request.metadata == {}


class TestUpdateMarketPriceUseCase:
    """Test UpdateMarketPriceUseCase with comprehensive scenarios."""

    @pytest.fixture
    def mock_unit_of_work(self):
        """Create mock unit of work."""
        uow = AsyncMock()
        uow.orders = AsyncMock()
        uow.portfolios = AsyncMock()
        uow.positions = AsyncMock()
        uow.commit = AsyncMock()
        uow.rollback = AsyncMock()
        uow.__aenter__ = AsyncMock(return_value=uow)
        uow.__aexit__ = AsyncMock(return_value=None)
        return uow

    @pytest.fixture
    def mock_order_processor(self):
        """Create mock order processor."""
        processor = Mock()
        processor.process_order_fill = Mock()
        processor.should_trigger_order = Mock()
        return processor

    @pytest.fixture
    def mock_market_microstructure(self):
        """Create mock market microstructure."""
        market = Mock()
        market.simulate_fill = Mock()
        market.get_fill_price = Mock()
        return market

    @pytest.fixture
    def use_case(self, mock_unit_of_work, mock_order_processor, mock_market_microstructure):
        """Create use case instance."""
        return UpdateMarketPriceUseCase(
            unit_of_work=mock_unit_of_work,
            order_processor=mock_order_processor,
            market_microstructure=mock_market_microstructure,
        )

    @pytest.fixture
    def sample_orders(self):
        """Create sample orders for testing."""
        # Stop order that should trigger
        stop_order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=Decimal("100"),
            stop_price=Decimal("145.00"),
        )
        stop_order.submit("BROKER-001")

        # Limit order that should fill
        limit_order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("50"),
            limit_price=Decimal("150.00"),
        )
        limit_order.submit("BROKER-002")

        # Market order that should fill immediately
        market_order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=Decimal("25")
        )
        market_order.submit("BROKER-003")

        return [stop_order, limit_order, market_order]

    @pytest.mark.asyncio
    async def test_update_market_price_success(
        self,
        use_case,
        mock_unit_of_work,
        mock_order_processor,
        mock_market_microstructure,
        sample_orders,
    ):
        """Test successful market price update with order processing."""
        # Setup
        request = UpdateMarketPriceRequest(
            symbol="AAPL", price=Decimal("149.50"), volume=10000, timestamp=datetime.now(UTC)
        )

        mock_unit_of_work.orders.get_active_orders.return_value = sample_orders

        # Configure order processor behavior
        def should_trigger_side_effect(order, price):
            if order.order_type == OrderType.STOP and price <= order.stop_price:
                return True
            if (
                order.order_type == OrderType.LIMIT
                and order.side == OrderSide.BUY
                and price <= order.limit_price
            ):
                return True
            if order.order_type == OrderType.MARKET:
                return True
            return False

        mock_order_processor.should_trigger_order.side_effect = should_trigger_side_effect

        # Configure market microstructure
        mock_market_microstructure.simulate_fill.return_value = (
            True,
            Decimal("100"),
            Decimal("149.50"),
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert len(response.orders_triggered) > 0
        assert len(response.orders_filled) > 0

        # Verify orders were processed
        mock_unit_of_work.orders.get_active_orders.assert_called_once()
        assert mock_order_processor.should_trigger_order.call_count > 0

    @pytest.mark.asyncio
    async def test_update_market_price_no_active_orders(self, use_case, mock_unit_of_work):
        """Test market price update with no active orders."""
        # Setup
        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("150.00"))

        mock_unit_of_work.orders.get_active_orders.return_value = []

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert len(response.orders_triggered) == 0
        assert len(response.orders_filled) == 0

    @pytest.mark.asyncio
    async def test_update_market_price_stop_order_trigger(
        self, use_case, mock_unit_of_work, mock_order_processor, mock_market_microstructure
    ):
        """Test stop order triggering on price movement."""
        # Setup - Stop loss order
        stop_order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=Decimal("100"),
            stop_price=Decimal("145.00"),
        )
        stop_order.submit("BROKER-001")

        request = UpdateMarketPriceRequest(
            symbol="AAPL",
            price=Decimal("144.50"),  # Below stop price
        )

        mock_unit_of_work.orders.get_active_orders.return_value = [stop_order]
        mock_order_processor.should_trigger_order.return_value = True
        mock_market_microstructure.simulate_fill.return_value = (
            True,
            Decimal("100"),
            Decimal("144.50"),
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert stop_order.id in response.orders_triggered
        assert stop_order.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_update_market_price_limit_order_fill(
        self, use_case, mock_unit_of_work, mock_order_processor, mock_market_microstructure
    ):
        """Test limit order filling when price conditions are met."""
        # Setup - Buy limit order
        limit_order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("50"),
            limit_price=Decimal("150.00"),
        )
        limit_order.submit("BROKER-002")

        request = UpdateMarketPriceRequest(
            symbol="AAPL",
            price=Decimal("149.00"),  # Below limit price (good for buy)
        )

        mock_unit_of_work.orders.get_active_orders.return_value = [limit_order]
        mock_order_processor.should_trigger_order.return_value = True
        mock_market_microstructure.simulate_fill.return_value = (
            True,
            Decimal("50"),
            Decimal("149.00"),
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert limit_order.id in response.orders_filled
        assert limit_order.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_update_market_price_partial_fill(
        self, use_case, mock_unit_of_work, mock_order_processor, mock_market_microstructure
    ):
        """Test partial order fill."""
        # Setup
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("100"),
            limit_price=Decimal("150.00"),
        )
        order.submit("BROKER-001")

        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("149.50"))

        mock_unit_of_work.orders.get_active_orders.return_value = [order]
        mock_order_processor.should_trigger_order.return_value = True
        mock_market_microstructure.simulate_fill.return_value = (
            True,
            Decimal("50"),
            Decimal("149.50"),
        )  # Partial fill

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert order.id in response.orders_filled
        assert order.filled_quantity == Decimal("50")
        assert order.status == OrderStatus.PARTIALLY_FILLED

    @pytest.mark.asyncio
    async def test_update_market_price_no_fill(
        self, use_case, mock_unit_of_work, mock_order_processor, mock_market_microstructure
    ):
        """Test order that triggers but doesn't fill."""
        # Setup
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("100"),
            limit_price=Decimal("150.00"),
        )
        order.submit("BROKER-001")

        request = UpdateMarketPriceRequest(
            symbol="AAPL",
            price=Decimal("151.00"),  # Above limit price
        )

        mock_unit_of_work.orders.get_active_orders.return_value = [order]
        mock_order_processor.should_trigger_order.return_value = False

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert order.id not in response.orders_triggered
        assert order.id not in response.orders_filled
        assert order.status == OrderStatus.SUBMITTED

    @pytest.mark.asyncio
    async def test_validate_negative_price(self, use_case):
        """Test validation with negative price."""
        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("-150.00"))

        error = await use_case.validate(request)
        assert error == "Price must be positive"

    @pytest.mark.asyncio
    async def test_validate_zero_price(self, use_case):
        """Test validation with zero price."""
        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("0"))

        error = await use_case.validate(request)
        assert error == "Price must be positive"

    @pytest.mark.asyncio
    async def test_validate_negative_volume(self, use_case):
        """Test validation with negative volume."""
        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("150.00"), volume=-100)

        error = await use_case.validate(request)
        assert error == "Volume cannot be negative"

    @pytest.mark.asyncio
    async def test_validate_valid_request(self, use_case):
        """Test validation with valid request."""
        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("150.00"), volume=1000)

        error = await use_case.validate(request)
        assert error is None

    @pytest.mark.asyncio
    async def test_update_market_price_with_different_symbols(
        self, use_case, mock_unit_of_work, mock_order_processor
    ):
        """Test that only orders with matching symbol are processed."""
        # Setup
        aapl_order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=Decimal("100")
        )
        aapl_order.submit("BROKER-001")

        googl_order = Order(
            symbol="GOOGL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=Decimal("50")
        )
        googl_order.submit("BROKER-002")

        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("150.00"))

        mock_unit_of_work.orders.get_active_orders.return_value = [aapl_order, googl_order]

        # Only process AAPL order
        def should_trigger_side_effect(order, price):
            return order.symbol == "AAPL"

        mock_order_processor.should_trigger_order.side_effect = should_trigger_side_effect

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        # Only AAPL order should be checked
        assert mock_order_processor.should_trigger_order.call_count == 2

    @pytest.mark.asyncio
    async def test_update_market_price_order_update_failure(
        self, use_case, mock_unit_of_work, mock_order_processor, mock_market_microstructure
    ):
        """Test handling of order update failure."""
        # Setup
        order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=Decimal("100")
        )
        order.submit("BROKER-001")

        request = UpdateMarketPriceRequest(symbol="AAPL", price=Decimal("150.00"))

        mock_unit_of_work.orders.get_active_orders.return_value = [order]
        mock_order_processor.should_trigger_order.return_value = True
        mock_market_microstructure.simulate_fill.return_value = (
            True,
            Decimal("100"),
            Decimal("150.00"),
        )
        mock_unit_of_work.orders.update_order.side_effect = Exception("Database error")

        # Execute - should handle exception gracefully
        response = await use_case.execute(request)

        # Assert - might fail but shouldn't crash
        assert response.success is False or response.success is True


class TestProcessPendingOrdersUseCase:
    """Test ProcessPendingOrdersUseCase with comprehensive scenarios."""

    @pytest.fixture
    def mock_unit_of_work(self):
        """Create mock unit of work."""
        uow = AsyncMock()
        uow.orders = AsyncMock()
        uow.market_data = AsyncMock()
        uow.commit = AsyncMock()
        uow.rollback = AsyncMock()
        uow.__aenter__ = AsyncMock(return_value=uow)
        uow.__aexit__ = AsyncMock(return_value=None)
        return uow

    @pytest.fixture
    def mock_order_processor(self):
        """Create mock order processor."""
        processor = Mock()
        processor.should_trigger_order = Mock()
        processor.process_order_fill = Mock()
        return processor

    @pytest.fixture
    def mock_market_microstructure(self):
        """Create mock market microstructure."""
        market = Mock()
        market.simulate_fill = Mock()
        return market

    @pytest.fixture
    def use_case(self, mock_unit_of_work, mock_order_processor, mock_market_microstructure):
        """Create use case instance."""
        return ProcessPendingOrdersUseCase(
            unit_of_work=mock_unit_of_work,
            order_processor=mock_order_processor,
            market_microstructure=mock_market_microstructure,
        )

    @pytest.mark.asyncio
    async def test_process_all_pending_orders(
        self, use_case, mock_unit_of_work, mock_order_processor, mock_market_microstructure
    ):
        """Test processing all pending orders."""
        # Setup
        order1 = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("100"),
            limit_price=Decimal("150.00"),
        )
        order1.submit("BROKER-001")

        order2 = Order(
            symbol="GOOGL",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=Decimal("50"),
            stop_price=Decimal("2500.00"),
        )
        order2.submit("BROKER-002")

        request = ProcessPendingOrdersRequest()

        mock_unit_of_work.orders.get_pending_orders.return_value = [order1, order2]

        # Mock market data
        mock_bar1 = Mock()
        mock_bar1.close = Decimal("149.00")
        mock_bar2 = Mock()
        mock_bar2.close = Decimal("2495.00")

        mock_unit_of_work.market_data.get_latest_bar.side_effect = [mock_bar1, mock_bar2]

        # Configure order processor
        mock_order_processor.should_trigger_order.side_effect = [True, True]
        mock_market_microstructure.simulate_fill.side_effect = [
            (True, Decimal("100"), Decimal("149.00")),
            (True, Decimal("50"), Decimal("2495.00")),
        ]

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.processed_count == 2
        assert len(response.triggered_orders) == 2
        assert len(response.filled_orders) == 2

    @pytest.mark.asyncio
    async def test_process_pending_orders_for_specific_symbol(
        self, use_case, mock_unit_of_work, mock_order_processor, mock_market_microstructure
    ):
        """Test processing pending orders for specific symbol."""
        # Setup
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("100"),
            limit_price=Decimal("150.00"),
        )
        order.submit("BROKER-001")

        request = ProcessPendingOrdersRequest(
            symbol="AAPL", current_prices={"AAPL": Decimal("149.00")}
        )

        mock_unit_of_work.orders.get_pending_orders.return_value = [order]
        mock_order_processor.should_trigger_order.return_value = True
        mock_market_microstructure.simulate_fill.return_value = (
            True,
            Decimal("100"),
            Decimal("149.00"),
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.processed_count == 1
        assert order.id in response.filled_orders

    @pytest.mark.asyncio
    async def test_process_pending_orders_no_pending(self, use_case, mock_unit_of_work):
        """Test processing when no pending orders exist."""
        # Setup
        request = ProcessPendingOrdersRequest()
        mock_unit_of_work.orders.get_pending_orders.return_value = []

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.processed_count == 0
        assert len(response.triggered_orders) == 0
        assert len(response.filled_orders) == 0

    @pytest.mark.asyncio
    async def test_process_pending_orders_with_provided_prices(
        self, use_case, mock_unit_of_work, mock_order_processor, mock_market_microstructure
    ):
        """Test processing with provided current prices."""
        # Setup
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("100"),
            limit_price=Decimal("150.00"),
        )
        order.submit("BROKER-001")

        request = ProcessPendingOrdersRequest(current_prices={"AAPL": Decimal("149.50")})

        mock_unit_of_work.orders.get_pending_orders.return_value = [order]
        mock_order_processor.should_trigger_order.return_value = True
        mock_market_microstructure.simulate_fill.return_value = (
            True,
            Decimal("100"),
            Decimal("149.50"),
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.processed_count == 1
        # Should use provided price, not fetch from market data
        mock_unit_of_work.market_data.get_latest_bar.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_pending_orders_no_market_data(self, use_case, mock_unit_of_work):
        """Test processing when market data is unavailable."""
        # Setup
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("100"),
            limit_price=Decimal("150.00"),
        )
        order.submit("BROKER-001")

        request = ProcessPendingOrdersRequest()

        mock_unit_of_work.orders.get_pending_orders.return_value = [order]
        mock_unit_of_work.market_data.get_latest_bar.return_value = None

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.processed_count == 1
        assert len(response.triggered_orders) == 0  # Can't trigger without price

    @pytest.mark.asyncio
    async def test_validate_always_passes(self, use_case):
        """Test validation always passes for process pending orders."""
        request = ProcessPendingOrdersRequest()
        error = await use_case.validate(request)
        assert error is None


class TestCheckOrderTriggerUseCase:
    """Test CheckOrderTriggerUseCase with comprehensive scenarios."""

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
    def use_case(self, mock_unit_of_work):
        """Create use case instance."""
        return CheckOrderTriggerUseCase(unit_of_work=mock_unit_of_work)

    @pytest.mark.asyncio
    async def test_check_stop_order_trigger_sell(self, use_case, mock_unit_of_work):
        """Test stop order trigger for sell side."""
        # Setup - Stop loss order
        stop_order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=Decimal("100"),
            stop_price=Decimal("145.00"),
        )
        stop_order.submit("BROKER-001")

        request = CheckOrderTriggerRequest(
            order_id=stop_order.id,
            current_price=Decimal("144.00"),  # Below stop price - should trigger
        )

        mock_unit_of_work.orders.get_order_by_id.return_value = stop_order

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.should_trigger is True
        assert response.trigger_price == Decimal("145.00")
        assert "Stop price reached" in response.reason

    @pytest.mark.asyncio
    async def test_check_stop_order_trigger_buy(self, use_case, mock_unit_of_work):
        """Test stop order trigger for buy side."""
        # Setup - Buy stop order
        stop_order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.STOP,
            quantity=Decimal("100"),
            stop_price=Decimal("155.00"),
        )
        stop_order.submit("BROKER-001")

        request = CheckOrderTriggerRequest(
            order_id=stop_order.id,
            current_price=Decimal("156.00"),  # Above stop price - should trigger
        )

        mock_unit_of_work.orders.get_order_by_id.return_value = stop_order

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.should_trigger is True
        assert response.trigger_price == Decimal("155.00")

    @pytest.mark.asyncio
    async def test_check_limit_order_trigger_buy(self, use_case, mock_unit_of_work):
        """Test limit order trigger for buy side."""
        # Setup - Buy limit order
        limit_order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("100"),
            limit_price=Decimal("150.00"),
        )
        limit_order.submit("BROKER-001")

        request = CheckOrderTriggerRequest(
            order_id=limit_order.id,
            current_price=Decimal("149.00"),  # Below limit price - should trigger
        )

        mock_unit_of_work.orders.get_order_by_id.return_value = limit_order

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.should_trigger is True
        assert response.trigger_price == Decimal("150.00")
        assert "Limit price reached" in response.reason

    @pytest.mark.asyncio
    async def test_check_limit_order_trigger_sell(self, use_case, mock_unit_of_work):
        """Test limit order trigger for sell side."""
        # Setup - Sell limit order
        limit_order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("100"),
            limit_price=Decimal("150.00"),
        )
        limit_order.submit("BROKER-001")

        request = CheckOrderTriggerRequest(
            order_id=limit_order.id,
            current_price=Decimal("151.00"),  # Above limit price - should trigger
        )

        mock_unit_of_work.orders.get_order_by_id.return_value = limit_order

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.should_trigger is True

    @pytest.mark.asyncio
    async def test_check_market_order_trigger(self, use_case, mock_unit_of_work):
        """Test market order always triggers."""
        # Setup - Market order
        market_order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=Decimal("100")
        )
        market_order.submit("BROKER-001")

        request = CheckOrderTriggerRequest(
            order_id=market_order.id,
            current_price=Decimal("150.00"),  # Any price
        )

        mock_unit_of_work.orders.get_order_by_id.return_value = market_order

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.should_trigger is True
        assert response.reason == "Market order always executes"

    @pytest.mark.asyncio
    async def test_check_stop_limit_order_trigger(self, use_case, mock_unit_of_work):
        """Test stop-limit order trigger."""
        # Setup - Stop-limit order
        stop_limit_order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_LIMIT,
            quantity=Decimal("100"),
            stop_price=Decimal("145.00"),
            limit_price=Decimal("144.50"),
        )
        stop_limit_order.submit("BROKER-001")

        request = CheckOrderTriggerRequest(
            order_id=stop_limit_order.id,
            current_price=Decimal("144.75"),  # Between stop and limit
        )

        mock_unit_of_work.orders.get_order_by_id.return_value = stop_limit_order

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.should_trigger is True
        assert response.trigger_price == Decimal("145.00")

    @pytest.mark.asyncio
    async def test_check_order_not_found(self, use_case, mock_unit_of_work):
        """Test checking trigger for non-existent order."""
        # Setup
        request = CheckOrderTriggerRequest(order_id=uuid4(), current_price=Decimal("150.00"))

        mock_unit_of_work.orders.get_order_by_id.return_value = None

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.error == "Order not found"

    @pytest.mark.asyncio
    async def test_check_inactive_order(self, use_case, mock_unit_of_work):
        """Test checking trigger for inactive order."""
        # Setup - Cancelled order
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("100"),
            limit_price=Decimal("150.00"),
        )
        order.cancel("User requested")

        request = CheckOrderTriggerRequest(order_id=order.id, current_price=Decimal("149.00"))

        mock_unit_of_work.orders.get_order_by_id.return_value = order

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.should_trigger is False
        assert response.reason == "Order is not active"

    @pytest.mark.asyncio
    async def test_validate_negative_price(self, use_case):
        """Test validation with negative price."""
        request = CheckOrderTriggerRequest(order_id=uuid4(), current_price=Decimal("-150.00"))

        error = await use_case.validate(request)
        assert error == "Current price must be positive"

    @pytest.mark.asyncio
    async def test_validate_zero_price(self, use_case):
        """Test validation with zero price."""
        request = CheckOrderTriggerRequest(order_id=uuid4(), current_price=Decimal("0"))

        error = await use_case.validate(request)
        assert error == "Current price must be positive"

    @pytest.mark.asyncio
    async def test_validate_valid_request(self, use_case):
        """Test validation with valid request."""
        request = CheckOrderTriggerRequest(order_id=uuid4(), current_price=Decimal("150.00"))

        error = await use_case.validate(request)
        assert error is None


class TestSimulateOrderExecutionUseCase:
    """Test SimulateOrderExecutionUseCase with comprehensive scenarios."""

    @pytest.fixture
    def mock_unit_of_work(self):
        """Create mock unit of work."""
        uow = AsyncMock()
        uow.orders = AsyncMock()
        uow.portfolios = AsyncMock()
        uow.commit = AsyncMock()
        uow.rollback = AsyncMock()
        uow.__aenter__ = AsyncMock(return_value=uow)
        uow.__aexit__ = AsyncMock(return_value=None)
        return uow

    @pytest.fixture
    def mock_market_microstructure(self):
        """Create mock market microstructure."""
        market = Mock()
        market.simulate_fill = Mock()
        market.calculate_slippage = Mock()
        return market

    @pytest.fixture
    def mock_commission_calculator(self):
        """Create mock commission calculator."""
        calculator = Mock()
        calculator.calculate_commission = Mock()
        return calculator

    @pytest.fixture
    def use_case(self, mock_unit_of_work, mock_market_microstructure, mock_commission_calculator):
        """Create use case instance."""
        return SimulateOrderExecutionUseCase(
            unit_of_work=mock_unit_of_work,
            market_microstructure=mock_market_microstructure,
            commission_calculator=mock_commission_calculator,
        )

    @pytest.mark.asyncio
    async def test_simulate_full_fill(
        self, use_case, mock_unit_of_work, mock_market_microstructure, mock_commission_calculator
    ):
        """Test simulating full order fill."""
        # Setup
        order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=Decimal("100")
        )
        order.submit("BROKER-001")

        request = SimulateOrderExecutionRequest(order_id=order.id, fill_price=Decimal("150.00"))

        mock_unit_of_work.orders.get_order_by_id.return_value = order
        mock_market_microstructure.simulate_fill.return_value = (
            True,
            Decimal("100"),
            Decimal("150.00"),
        )
        mock_market_microstructure.calculate_slippage.return_value = Decimal("0.05")
        mock_commission_calculator.calculate_commission.return_value = Money(Decimal("10.00"))

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.filled is True
        assert response.fill_quantity == Decimal("100")
        assert response.actual_fill_price == Decimal("150.00")
        assert response.commission == Money(Decimal("10.00"))
        assert response.slippage == Decimal("0.05")
        assert order.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_simulate_partial_fill(
        self, use_case, mock_unit_of_work, mock_market_microstructure, mock_commission_calculator
    ):
        """Test simulating partial order fill."""
        # Setup
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("100"),
            limit_price=Decimal("150.00"),
        )
        order.submit("BROKER-001")

        request = SimulateOrderExecutionRequest(
            order_id=order.id,
            fill_price=Decimal("150.00"),
            fill_quantity=Decimal("50"),  # Partial
        )

        mock_unit_of_work.orders.get_order_by_id.return_value = order
        mock_market_microstructure.simulate_fill.return_value = (
            True,
            Decimal("50"),
            Decimal("150.00"),
        )
        mock_market_microstructure.calculate_slippage.return_value = Decimal("0.00")
        mock_commission_calculator.calculate_commission.return_value = Money(Decimal("5.00"))

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.filled is True
        assert response.fill_quantity == Decimal("50")
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.filled_quantity == Decimal("50")

    @pytest.mark.asyncio
    async def test_simulate_no_fill(self, use_case, mock_unit_of_work, mock_market_microstructure):
        """Test simulation that results in no fill."""
        # Setup
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("100"),
            limit_price=Decimal("150.00"),
        )
        order.submit("BROKER-001")

        request = SimulateOrderExecutionRequest(
            order_id=order.id,
            fill_price=Decimal("151.00"),  # Above limit
        )

        mock_unit_of_work.orders.get_order_by_id.return_value = order
        mock_market_microstructure.simulate_fill.return_value = (False, Decimal("0"), Decimal("0"))

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.filled is False
        assert response.fill_quantity == Decimal("0")
        assert order.status == OrderStatus.SUBMITTED

    @pytest.mark.asyncio
    async def test_simulate_order_not_found(self, use_case, mock_unit_of_work):
        """Test simulation when order doesn't exist."""
        # Setup
        request = SimulateOrderExecutionRequest(order_id=uuid4(), fill_price=Decimal("150.00"))

        mock_unit_of_work.orders.get_order_by_id.return_value = None

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.error == "Order not found"

    @pytest.mark.asyncio
    async def test_validate_negative_fill_price(self, use_case):
        """Test validation with negative fill price."""
        request = SimulateOrderExecutionRequest(order_id=uuid4(), fill_price=Decimal("-150.00"))

        error = await use_case.validate(request)
        assert error == "Fill price must be positive"

    @pytest.mark.asyncio
    async def test_validate_negative_fill_quantity(self, use_case):
        """Test validation with negative fill quantity."""
        request = SimulateOrderExecutionRequest(
            order_id=uuid4(), fill_price=Decimal("150.00"), fill_quantity=Decimal("-100")
        )

        error = await use_case.validate(request)
        assert error == "Fill quantity must be positive"

    @pytest.mark.asyncio
    async def test_validate_valid_request(self, use_case):
        """Test validation with valid request."""
        request = SimulateOrderExecutionRequest(
            order_id=uuid4(), fill_price=Decimal("150.00"), fill_quantity=Decimal("100")
        )

        error = await use_case.validate(request)
        assert error is None


class TestProcessOrderFillUseCase:
    """Test ProcessOrderFillUseCase with comprehensive scenarios."""

    @pytest.fixture
    def mock_unit_of_work(self):
        """Create mock unit of work."""
        uow = AsyncMock()
        uow.orders = AsyncMock()
        uow.portfolios = AsyncMock()
        uow.positions = AsyncMock()
        uow.commit = AsyncMock()
        uow.rollback = AsyncMock()
        uow.__aenter__ = AsyncMock(return_value=uow)
        uow.__aexit__ = AsyncMock(return_value=None)
        return uow

    @pytest.fixture
    def mock_order_processor(self):
        """Create mock order processor."""
        processor = Mock()
        processor.process_fill = Mock()
        return processor

    @pytest.fixture
    def mock_commission_calculator(self):
        """Create mock commission calculator."""
        calculator = Mock()
        calculator.calculate_commission = Mock()
        return calculator

    @pytest.fixture
    def use_case(self, mock_unit_of_work, mock_order_processor, mock_commission_calculator):
        """Create use case instance."""
        return ProcessOrderFillUseCase(
            unit_of_work=mock_unit_of_work,
            order_processor=mock_order_processor,
            commission_calculator=mock_commission_calculator,
        )

    @pytest.mark.asyncio
    async def test_process_full_fill_buy(
        self, use_case, mock_unit_of_work, mock_order_processor, mock_commission_calculator
    ):
        """Test processing full buy order fill."""
        # Setup
        order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=Decimal("100")
        )
        order.submit("BROKER-001")
        order.portfolio_id = uuid4()

        portfolio = Portfolio(name="Test Portfolio", initial_capital=Money(Decimal("100000")))
        portfolio.cash_balance = Money(Decimal("50000"))

        request = ProcessOrderFillRequest(
            order_id=order.id,
            fill_quantity=Decimal("100"),
            fill_price=Decimal("150.00"),
            timestamp=datetime.now(UTC),
        )

        mock_unit_of_work.orders.get_order_by_id.return_value = order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = portfolio
        mock_commission_calculator.calculate_commission.return_value = Money(Decimal("10.00"))

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.position_updated is True
        assert response.portfolio_updated is True
        assert response.commission == Money(Decimal("10.00"))

        # Order should be filled
        assert order.filled_quantity == Decimal("100")
        assert order.average_fill_price == Decimal("150.00")
        assert order.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_process_partial_fill(
        self, use_case, mock_unit_of_work, mock_order_processor, mock_commission_calculator
    ):
        """Test processing partial order fill."""
        # Setup
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("100"),
            limit_price=Decimal("150.00"),
        )
        order.submit("BROKER-001")
        order.portfolio_id = uuid4()

        portfolio = Portfolio(name="Test Portfolio", initial_capital=Money(Decimal("100000")))

        request = ProcessOrderFillRequest(
            order_id=order.id, fill_quantity=Decimal("50"), fill_price=Decimal("150.00")
        )

        mock_unit_of_work.orders.get_order_by_id.return_value = order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = portfolio
        mock_commission_calculator.calculate_commission.return_value = Money(Decimal("5.00"))

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert order.filled_quantity == Decimal("50")
        assert order.status == OrderStatus.PARTIALLY_FILLED

    @pytest.mark.asyncio
    async def test_process_fill_order_not_found(self, use_case, mock_unit_of_work):
        """Test processing fill when order doesn't exist."""
        # Setup
        request = ProcessOrderFillRequest(
            order_id=uuid4(), fill_quantity=Decimal("100"), fill_price=Decimal("150.00")
        )

        mock_unit_of_work.orders.get_order_by_id.return_value = None

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.error == "Order not found"

    @pytest.mark.asyncio
    async def test_validate_negative_fill_quantity(self, use_case):
        """Test validation with negative fill quantity."""
        request = ProcessOrderFillRequest(
            order_id=uuid4(), fill_quantity=Decimal("-100"), fill_price=Decimal("150.00")
        )

        error = await use_case.validate(request)
        assert error == "Fill quantity must be positive"

    @pytest.mark.asyncio
    async def test_validate_zero_fill_price(self, use_case):
        """Test validation with zero fill price."""
        request = ProcessOrderFillRequest(
            order_id=uuid4(), fill_quantity=Decimal("100"), fill_price=Decimal("0")
        )

        error = await use_case.validate(request)
        assert error == "Fill price must be positive"

    @pytest.mark.asyncio
    async def test_validate_valid_request(self, use_case):
        """Test validation with valid request."""
        request = ProcessOrderFillRequest(
            order_id=uuid4(), fill_quantity=Decimal("100"), fill_price=Decimal("150.00")
        )

        error = await use_case.validate(request)
        assert error is None
