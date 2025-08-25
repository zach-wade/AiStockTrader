"""
Comprehensive unit tests for Broker Coordinator.

Tests all broker coordinator functionality including:
- Use case factory creation
- Order placement coordination
- Order cancellation coordination
- Market price updates
- Order fill processing
- Order status retrieval
- Pending order processing
- Error handling and edge cases
"""

from decimal import Decimal
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from src.application.coordinators.broker_coordinator import BrokerCoordinator, UseCaseFactory
from src.application.interfaces.broker import IBroker
from src.application.interfaces.unit_of_work import IUnitOfWork
from src.domain.entities.order import OrderStatus


class TestUseCaseFactory:
    """Test UseCaseFactory class."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies."""
        return {
            "unit_of_work": AsyncMock(spec=IUnitOfWork),
            "order_processor": Mock(),
            "commission_calculator": Mock(),
            "market_microstructure": Mock(),
            "risk_calculator": Mock(),
            "order_validator": Mock(),
            "position_manager": Mock(),
        }

    @pytest.fixture
    def factory(self, mock_dependencies):
        """Create factory instance."""
        return UseCaseFactory(**mock_dependencies)

    @pytest.fixture
    def mock_broker(self):
        """Create mock broker."""
        return AsyncMock(spec=IBroker)

    def test_create_place_order_use_case(self, factory, mock_broker):
        """Test creating place order use case."""
        use_case = factory.create_place_order_use_case(mock_broker)

        assert use_case is not None
        assert use_case.unit_of_work == factory.unit_of_work
        assert use_case.broker == mock_broker
        assert use_case.order_validator == factory.order_validator
        assert use_case.risk_calculator == factory.risk_calculator

    def test_create_cancel_order_use_case(self, factory, mock_broker):
        """Test creating cancel order use case."""
        use_case = factory.create_cancel_order_use_case(mock_broker)

        assert use_case is not None
        assert use_case.unit_of_work == factory.unit_of_work
        assert use_case.broker == mock_broker

    def test_create_process_fill_use_case(self, factory):
        """Test creating process fill use case."""
        use_case = factory.create_process_fill_use_case()

        assert use_case is not None
        assert use_case.unit_of_work == factory.unit_of_work
        assert use_case.order_processor == factory.order_processor
        assert use_case.commission_calculator == factory.commission_calculator

    def test_create_simulate_execution_use_case(self, factory):
        """Test creating simulate execution use case."""
        use_case = factory.create_simulate_execution_use_case()

        assert use_case is not None
        assert use_case.unit_of_work == factory.unit_of_work
        assert use_case.market_microstructure == factory.market_microstructure
        assert use_case.commission_calculator == factory.commission_calculator

    def test_create_update_market_price_use_case(self, factory):
        """Test creating update market price use case."""
        use_case = factory.create_update_market_price_use_case()

        assert use_case is not None
        assert use_case.unit_of_work == factory.unit_of_work
        assert use_case.order_processor == factory.order_processor
        assert use_case.market_microstructure == factory.market_microstructure

    def test_create_get_order_status_use_case(self, factory, mock_broker):
        """Test creating get order status use case."""
        use_case = factory.create_get_order_status_use_case(mock_broker)

        assert use_case is not None
        assert use_case.unit_of_work == factory.unit_of_work
        assert use_case.broker == mock_broker

    def test_create_process_pending_orders_use_case(self, factory):
        """Test creating process pending orders use case."""
        use_case = factory.create_process_pending_orders_use_case()

        assert use_case is not None
        assert use_case.unit_of_work == factory.unit_of_work
        assert use_case.order_processor == factory.order_processor
        assert use_case.market_microstructure == factory.market_microstructure


class TestBrokerCoordinator:
    """Test BrokerCoordinator class."""

    @pytest.fixture
    def mock_broker(self):
        """Create mock broker."""
        broker = AsyncMock(spec=IBroker)
        broker.submit_order = AsyncMock(return_value="BROKER-123")
        broker.cancel_order = AsyncMock(return_value=True)
        broker.get_order_status = AsyncMock(return_value=OrderStatus.SUBMITTED)
        return broker

    @pytest.fixture
    def mock_use_case_factory(self):
        """Create mock use case factory."""
        factory = Mock(spec=UseCaseFactory)

        # Add required attributes for the factory
        factory.unit_of_work = AsyncMock(spec=IUnitOfWork)
        factory.order_processor = Mock()
        factory.commission_calculator = Mock()
        factory.market_microstructure = Mock()
        factory.risk_calculator = Mock()
        factory.order_validator = Mock()
        factory.position_manager = Mock()

        # Create mock use cases
        factory.create_place_order_use_case = Mock()
        factory.create_cancel_order_use_case = Mock()
        factory.create_process_fill_use_case = Mock()
        factory.create_update_market_price_use_case = Mock()
        factory.create_get_order_status_use_case = Mock()
        factory.create_process_pending_orders_use_case = Mock()

        return factory

    @pytest.fixture
    def coordinator(self, mock_broker, mock_use_case_factory):
        """Create coordinator instance."""
        return BrokerCoordinator(broker=mock_broker, use_case_factory=mock_use_case_factory)

    @pytest.mark.asyncio
    async def test_place_order_success(self, coordinator, mock_use_case_factory):
        """Test successful order placement."""
        # Setup
        order_request = {
            "portfolio_id": str(uuid4()),
            "symbol": "AAPL",
            "side": "buy",
            "order_type": "market",
            "quantity": 100,
        }

        mock_use_case = AsyncMock()
        mock_response = Mock()
        mock_response.success = True
        mock_response.order_id = uuid4()
        mock_response.broker_order_id = "BROKER-123"
        mock_response.status = "submitted"
        mock_response.error = None
        mock_use_case.execute = AsyncMock(return_value=mock_response)

        mock_use_case_factory.create_place_order_use_case.return_value = mock_use_case

        # Execute
        result = await coordinator.place_order(order_request)

        # Assert
        assert result["success"] is True
        assert result["order_id"] == str(mock_response.order_id)
        assert result["broker_order_id"] == "BROKER-123"
        assert result["status"] == "submitted"
        assert result["error"] is None

        mock_use_case_factory.create_place_order_use_case.assert_called_once()
        mock_use_case.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_place_order_with_limit_price(self, coordinator, mock_use_case_factory):
        """Test placing limit order."""
        # Setup
        order_request = {
            "portfolio_id": str(uuid4()),
            "symbol": "AAPL",
            "side": "sell",
            "order_type": "limit",
            "quantity": 50,
            "limit_price": 150.50,
            "time_in_force": "gtc",
        }

        mock_use_case = AsyncMock()
        mock_response = Mock()
        mock_response.success = True
        mock_response.order_id = uuid4()
        mock_response.broker_order_id = "BROKER-456"
        mock_response.status = "submitted"
        mock_response.error = None
        mock_use_case.execute = AsyncMock(return_value=mock_response)

        mock_use_case_factory.create_place_order_use_case.return_value = mock_use_case

        # Execute
        result = await coordinator.place_order(order_request)

        # Assert
        assert result["success"] is True

        # Verify the request was properly constructed
        call_args = mock_use_case.execute.call_args[0][0]
        assert call_args.order_type == "limit"
        assert call_args.limit_price == 150.50
        assert call_args.time_in_force == "gtc"

    @pytest.mark.asyncio
    async def test_place_order_failure(self, coordinator, mock_use_case_factory):
        """Test order placement failure."""
        # Setup
        order_request = {
            "portfolio_id": str(uuid4()),
            "symbol": "AAPL",
            "side": "buy",
            "order_type": "market",
            "quantity": 100,
        }

        mock_use_case = AsyncMock()
        mock_response = Mock()
        mock_response.success = False
        mock_response.order_id = None
        mock_response.broker_order_id = None
        mock_response.status = None
        mock_response.error = "Insufficient buying power"
        mock_use_case.execute = AsyncMock(return_value=mock_response)

        mock_use_case_factory.create_place_order_use_case.return_value = mock_use_case

        # Execute
        result = await coordinator.place_order(order_request)

        # Assert
        assert result["success"] is False
        assert result["order_id"] is None
        assert result["error"] == "Insufficient buying power"

    @pytest.mark.asyncio
    async def test_cancel_order_success(self, coordinator, mock_use_case_factory):
        """Test successful order cancellation."""
        # Setup
        order_id = uuid4()
        reason = "User requested"

        mock_use_case = AsyncMock()
        mock_response = Mock()
        mock_response.success = True
        mock_response.cancelled = True
        mock_response.final_status = "cancelled"
        mock_response.error = None
        mock_use_case.execute = AsyncMock(return_value=mock_response)

        mock_use_case_factory.create_cancel_order_use_case.return_value = mock_use_case

        # Execute
        result = await coordinator.cancel_order(order_id, reason)

        # Assert
        assert result["success"] is True
        assert result["cancelled"] is True
        assert result["final_status"] == "cancelled"
        assert result["error"] is None

        # Verify the request was properly constructed
        call_args = mock_use_case.execute.call_args[0][0]
        assert call_args.order_id == order_id
        assert call_args.reason == reason

    @pytest.mark.asyncio
    async def test_cancel_order_failure(self, coordinator, mock_use_case_factory):
        """Test order cancellation failure."""
        # Setup
        order_id = uuid4()

        mock_use_case = AsyncMock()
        mock_response = Mock()
        mock_response.success = False
        mock_response.cancelled = False
        mock_response.final_status = None
        mock_response.error = "Order already filled"
        mock_use_case.execute = AsyncMock(return_value=mock_response)

        mock_use_case_factory.create_cancel_order_use_case.return_value = mock_use_case

        # Execute
        result = await coordinator.cancel_order(order_id)

        # Assert
        assert result["success"] is False
        assert result["cancelled"] is False
        assert result["error"] == "Order already filled"

    @pytest.mark.asyncio
    async def test_update_market_price(self, coordinator, mock_broker, mock_use_case_factory):
        """Test updating market price."""
        # Setup
        symbol = "AAPL"
        price = Decimal("150.00")

        mock_use_case = AsyncMock()
        mock_response = Mock()
        mock_response.success = True
        mock_response.orders_triggered = [uuid4(), uuid4()]
        mock_response.orders_filled = [uuid4()]
        mock_response.error = None
        mock_use_case.execute = AsyncMock(return_value=mock_response)

        mock_use_case_factory.create_update_market_price_use_case.return_value = mock_use_case

        # Execute
        result = await coordinator.update_market_price(symbol, price)

        # Assert
        assert result["success"] is True
        assert len(result["orders_triggered"]) == 2
        assert len(result["orders_filled"]) == 1
        assert coordinator.market_prices[symbol] == price

        # Verify broker was updated if it supports it
        if hasattr(mock_broker, "set_market_price"):
            mock_broker.set_market_price.assert_called_once_with(symbol, price)

    @pytest.mark.asyncio
    async def test_update_market_price_with_broker_support(
        self, coordinator, mock_broker, mock_use_case_factory
    ):
        """Test updating market price when broker supports it."""
        # Setup
        symbol = "MSFT"
        price = Decimal("300.00")

        # Add set_market_price method to broker
        mock_broker.set_market_price = Mock()

        mock_use_case = AsyncMock()
        mock_response = Mock()
        mock_response.success = True
        mock_response.orders_triggered = []
        mock_response.orders_filled = []
        mock_response.error = None
        mock_use_case.execute = AsyncMock(return_value=mock_response)

        mock_use_case_factory.create_update_market_price_use_case.return_value = mock_use_case

        # Execute
        result = await coordinator.update_market_price(symbol, price)

        # Assert
        assert result["success"] is True
        assert coordinator.market_prices[symbol] == price
        mock_broker.set_market_price.assert_called_once_with(symbol, price)

    @pytest.mark.asyncio
    async def test_process_order_fill_complete(self, coordinator, mock_use_case_factory):
        """Test processing complete order fill."""
        # Setup
        order_id = uuid4()
        fill_price = Decimal("149.50")
        fill_quantity = None  # Complete fill

        mock_use_case = AsyncMock()
        mock_response = Mock()
        mock_response.success = True
        mock_response.filled = True
        mock_response.fill_price = fill_price
        mock_response.fill_quantity = 100
        mock_response.commission = Decimal("1.00")
        mock_response.position_id = uuid4()
        mock_response.error = None
        mock_use_case.execute = AsyncMock(return_value=mock_response)

        mock_use_case_factory.create_process_fill_use_case.return_value = mock_use_case

        # Execute
        result = await coordinator.process_order_fill(order_id, fill_price, fill_quantity)

        # Assert
        assert result["success"] is True
        assert result["filled"] is True
        assert result["fill_price"] == fill_price
        assert result["fill_quantity"] == 100
        assert result["commission"] == Decimal("1.00")
        assert result["position_id"] == str(mock_response.position_id)

    @pytest.mark.asyncio
    async def test_process_order_fill_partial(self, coordinator, mock_use_case_factory):
        """Test processing partial order fill."""
        # Setup
        order_id = uuid4()
        fill_price = Decimal("149.75")
        fill_quantity = 50

        mock_use_case = AsyncMock()
        mock_response = Mock()
        mock_response.success = True
        mock_response.filled = True
        mock_response.fill_price = fill_price
        mock_response.fill_quantity = fill_quantity
        mock_response.commission = Decimal("0.50")
        mock_response.position_id = None
        mock_response.error = None
        mock_use_case.execute = AsyncMock(return_value=mock_response)

        mock_use_case_factory.create_process_fill_use_case.return_value = mock_use_case

        # Execute
        result = await coordinator.process_order_fill(order_id, fill_price, fill_quantity)

        # Assert
        assert result["success"] is True
        assert result["fill_quantity"] == 50
        assert result["position_id"] is None

    @pytest.mark.asyncio
    async def test_process_order_fill_failure(self, coordinator, mock_use_case_factory):
        """Test order fill processing failure."""
        # Setup
        order_id = uuid4()
        fill_price = Decimal("149.50")

        mock_use_case = AsyncMock()
        mock_response = Mock()
        mock_response.success = False
        mock_response.filled = False
        mock_response.error = "Order not found"
        mock_use_case.execute = AsyncMock(return_value=mock_response)

        mock_use_case_factory.create_process_fill_use_case.return_value = mock_use_case

        # Execute
        result = await coordinator.process_order_fill(order_id, fill_price)

        # Assert
        assert result["success"] is False
        assert result["filled"] is False
        assert result["error"] == "Order not found"

    @pytest.mark.asyncio
    async def test_get_order_status(self, coordinator, mock_use_case_factory):
        """Test getting order status."""
        # Setup
        order_id = uuid4()

        mock_use_case = AsyncMock()
        mock_response = Mock()
        mock_response.success = True
        mock_response.status = "partially_filled"
        mock_response.filled_quantity = 50
        mock_response.average_fill_price = 149.75
        mock_response.error = None
        mock_use_case.execute = AsyncMock(return_value=mock_response)

        mock_use_case_factory.create_get_order_status_use_case.return_value = mock_use_case

        # Execute
        result = await coordinator.get_order_status(order_id)

        # Assert
        assert result["success"] is True
        assert result["status"] == "partially_filled"
        assert result["filled_quantity"] == 50
        assert result["average_fill_price"] == 149.75

    @pytest.mark.asyncio
    async def test_process_pending_orders_with_prices(self, coordinator, mock_use_case_factory):
        """Test processing pending orders with current prices."""
        # Setup
        coordinator.market_prices = {"AAPL": Decimal("150.00"), "MSFT": Decimal("300.00")}

        mock_use_case = AsyncMock()
        mock_response = Mock()
        mock_response.success = True
        mock_response.processed_count = 5
        mock_response.triggered_orders = [uuid4(), uuid4()]
        mock_response.filled_orders = [uuid4()]
        mock_response.error = None
        mock_use_case.execute = AsyncMock(return_value=mock_response)

        mock_use_case_factory.create_process_pending_orders_use_case.return_value = mock_use_case

        # Execute
        result = await coordinator.process_pending_orders()

        # Assert
        assert result["success"] is True
        assert result["processed_count"] == 5
        assert len(result["triggered_orders"]) == 2
        assert len(result["filled_orders"]) == 1

        # Verify prices were passed
        call_args = mock_use_case.execute.call_args[0][0]
        assert call_args.current_prices == coordinator.market_prices

    @pytest.mark.asyncio
    async def test_process_pending_orders_empty_prices(self, coordinator, mock_use_case_factory):
        """Test processing pending orders with no market prices."""
        # Setup
        coordinator.market_prices = {}

        mock_use_case = AsyncMock()
        mock_response = Mock()
        mock_response.success = True
        mock_response.processed_count = 0
        mock_response.triggered_orders = []
        mock_response.filled_orders = []
        mock_response.error = None
        mock_use_case.execute = AsyncMock(return_value=mock_response)

        mock_use_case_factory.create_process_pending_orders_use_case.return_value = mock_use_case

        # Execute
        result = await coordinator.process_pending_orders()

        # Assert
        assert result["success"] is True
        assert result["processed_count"] == 0
        assert len(result["triggered_orders"]) == 0

    def test_create_get_order_status_use_case_helper(
        self, coordinator, mock_broker, mock_use_case_factory
    ):
        """Test helper method for creating get order status use case."""
        use_case = coordinator.create_get_order_status_use_case(mock_broker)

        assert use_case is not None
        assert use_case.unit_of_work == mock_use_case_factory.unit_of_work
        assert use_case.broker == mock_broker

    def test_create_process_pending_orders_use_case_helper(
        self, coordinator, mock_use_case_factory
    ):
        """Test helper method for creating process pending orders use case."""
        use_case = coordinator.create_process_pending_orders_use_case()

        assert use_case is not None
        assert use_case.unit_of_work == mock_use_case_factory.unit_of_work
        assert use_case.order_processor == mock_use_case_factory.order_processor
        assert use_case.market_microstructure == mock_use_case_factory.market_microstructure

    @pytest.mark.asyncio
    async def test_place_order_with_strategy_id(self, coordinator, mock_use_case_factory):
        """Test placing order with strategy ID."""
        # Setup
        order_request = {
            "portfolio_id": str(uuid4()),
            "symbol": "AAPL",
            "side": "buy",
            "order_type": "market",
            "quantity": 100,
            "strategy_id": "MOMENTUM-001",
        }

        mock_use_case = AsyncMock()
        mock_response = Mock()
        mock_response.success = True
        mock_response.order_id = uuid4()
        mock_response.broker_order_id = "BROKER-789"
        mock_response.status = "submitted"
        mock_response.error = None
        mock_use_case.execute = AsyncMock(return_value=mock_response)

        mock_use_case_factory.create_place_order_use_case.return_value = mock_use_case

        # Execute
        result = await coordinator.place_order(order_request)

        # Assert
        assert result["success"] is True

        # Verify strategy ID was passed
        call_args = mock_use_case.execute.call_args[0][0]
        assert call_args.strategy_id == "MOMENTUM-001"

    @pytest.mark.asyncio
    async def test_place_order_with_request_id(self, coordinator, mock_use_case_factory):
        """Test placing order with custom request ID."""
        # Setup
        request_id = str(uuid4())
        order_request = {
            "request_id": request_id,
            "portfolio_id": str(uuid4()),
            "symbol": "AAPL",
            "side": "buy",
            "order_type": "market",
            "quantity": 100,
        }

        mock_use_case = AsyncMock()
        mock_response = Mock()
        mock_response.success = True
        mock_response.order_id = uuid4()
        mock_response.broker_order_id = "BROKER-999"
        mock_response.status = "submitted"
        mock_response.error = None
        mock_use_case.execute = AsyncMock(return_value=mock_response)

        mock_use_case_factory.create_place_order_use_case.return_value = mock_use_case

        # Execute
        result = await coordinator.place_order(order_request)

        # Assert
        assert result["success"] is True

        # Verify request ID was preserved
        call_args = mock_use_case.execute.call_args[0][0]
        assert str(call_args.request_id) == request_id
