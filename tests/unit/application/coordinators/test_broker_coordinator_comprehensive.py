"""
Comprehensive tests for broker coordinator with full coverage.

Tests the BrokerCoordinator class which orchestrates between
thin broker adapters and business use cases.

Covers all scenarios including success, failure, and edge cases.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from src.application.coordinators.broker_coordinator import BrokerCoordinator, UseCaseFactory
from src.application.interfaces.broker import IBroker
from src.application.interfaces.unit_of_work import IUnitOfWork
from src.application.use_cases.market_simulation import (
    ProcessPendingOrdersResponse,
    ProcessPendingOrdersUseCase,
    UpdateMarketPriceResponse,
)
from src.application.use_cases.order_execution import (
    ProcessOrderFillResponse,
    ProcessOrderFillUseCase,
    SimulateOrderExecutionUseCase,
)
from src.application.use_cases.trading import (
    CancelOrderResponse,
    CancelOrderUseCase,
    GetOrderStatusResponse,
    GetOrderStatusUseCase,
    PlaceOrderResponse,
    PlaceOrderUseCase,
)
from src.domain.entities.order import OrderStatus


@pytest.fixture
def mock_broker():
    """Create a mock broker."""
    broker = Mock(spec=IBroker)
    broker.submit_order = Mock(return_value="BROKER123")
    broker.cancel_order = Mock(return_value=True)
    broker.update_order = Mock()
    broker.get_order_status = Mock(return_value=OrderStatus.PENDING)
    broker.set_market_price = Mock()
    return broker


@pytest.fixture
def mock_unit_of_work():
    """Create a mock unit of work."""
    uow = Mock(spec=IUnitOfWork)
    uow.orders = AsyncMock()
    uow.portfolios = AsyncMock()
    uow.positions = AsyncMock()
    uow.market_data = AsyncMock()
    return uow


@pytest.fixture
def mock_use_case_factory():
    """Create a mock use case factory."""
    factory = Mock(spec=UseCaseFactory)
    factory.unit_of_work = Mock(spec=IUnitOfWork)
    factory.order_processor = Mock()
    factory.commission_calculator = Mock()
    factory.market_microstructure = Mock()
    factory.risk_calculator = Mock()
    factory.order_validator = Mock()
    factory.position_manager = Mock()

    # Mock use case creation methods
    factory.create_place_order_use_case = Mock()
    factory.create_cancel_order_use_case = Mock()
    factory.create_process_fill_use_case = Mock()
    factory.create_simulate_execution_use_case = Mock()
    factory.create_update_market_price_use_case = Mock()
    factory.create_get_order_status_use_case = Mock()
    factory.create_process_pending_orders_use_case = Mock()

    return factory


@pytest.fixture
def broker_coordinator(mock_broker, mock_use_case_factory):
    """Create a broker coordinator instance."""
    return BrokerCoordinator(broker=mock_broker, use_case_factory=mock_use_case_factory)


class TestUseCaseFactory:
    """Test UseCaseFactory."""

    def test_create_place_order_use_case(self, mock_unit_of_work, mock_broker):
        """Test creating place order use case."""
        factory = UseCaseFactory(
            unit_of_work=mock_unit_of_work,
            order_processor=Mock(),
            commission_calculator=Mock(),
            market_microstructure=Mock(),
            risk_calculator=Mock(),
            order_validator=Mock(),
            position_manager=Mock(),
        )

        use_case = factory.create_place_order_use_case(mock_broker)

        assert isinstance(use_case, PlaceOrderUseCase)
        assert use_case.unit_of_work == mock_unit_of_work
        assert use_case.broker == mock_broker
        assert use_case.order_validator == factory.order_validator
        assert use_case.risk_calculator == factory.risk_calculator

    def test_create_cancel_order_use_case(self, mock_unit_of_work, mock_broker):
        """Test creating cancel order use case."""
        factory = UseCaseFactory(
            unit_of_work=mock_unit_of_work,
            order_processor=Mock(),
            commission_calculator=Mock(),
            market_microstructure=Mock(),
            risk_calculator=Mock(),
            order_validator=Mock(),
            position_manager=Mock(),
        )

        use_case = factory.create_cancel_order_use_case(mock_broker)

        assert isinstance(use_case, CancelOrderUseCase)
        assert use_case.unit_of_work == mock_unit_of_work
        assert use_case.broker == mock_broker

    def test_create_process_fill_use_case(self, mock_unit_of_work):
        """Test creating process fill use case."""
        factory = UseCaseFactory(
            unit_of_work=mock_unit_of_work,
            order_processor=Mock(),
            commission_calculator=Mock(),
            market_microstructure=Mock(),
            risk_calculator=Mock(),
            order_validator=Mock(),
            position_manager=Mock(),
        )

        use_case = factory.create_process_fill_use_case()

        assert isinstance(use_case, ProcessOrderFillUseCase)
        assert use_case.unit_of_work == mock_unit_of_work
        assert use_case.order_processor == factory.order_processor
        assert use_case.commission_calculator == factory.commission_calculator

    def test_create_simulate_execution_use_case(self, mock_unit_of_work):
        """Test creating simulate execution use case."""
        factory = UseCaseFactory(
            unit_of_work=mock_unit_of_work,
            order_processor=Mock(),
            commission_calculator=Mock(),
            market_microstructure=Mock(),
            risk_calculator=Mock(),
            order_validator=Mock(),
            position_manager=Mock(),
        )

        use_case = factory.create_simulate_execution_use_case()

        assert isinstance(use_case, SimulateOrderExecutionUseCase)
        assert use_case.unit_of_work == mock_unit_of_work
        assert use_case.market_microstructure == factory.market_microstructure
        assert use_case.commission_calculator == factory.commission_calculator


class TestBrokerCoordinator:
    """Test BrokerCoordinator."""

    @pytest.mark.asyncio
    async def test_place_order_success(self, broker_coordinator, mock_use_case_factory):
        """Test successful order placement."""
        # Setup mock use case
        mock_use_case = AsyncMock(spec=PlaceOrderUseCase)
        mock_response = PlaceOrderResponse(
            success=True,
            order_id=uuid4(),
            broker_order_id="BROKER123",
            status="pending",
            request_id=uuid4(),
        )
        mock_use_case.execute.return_value = mock_response
        mock_use_case_factory.create_place_order_use_case.return_value = mock_use_case

        # Create order request
        order_request = {
            "portfolio_id": str(uuid4()),
            "symbol": "AAPL",
            "side": "buy",
            "order_type": "market",
            "quantity": 100,
        }

        # Execute
        result = await broker_coordinator.place_order(order_request)

        # Assert
        assert result["success"] is True
        assert result["broker_order_id"] == "BROKER123"
        assert result["status"] == "pending"
        assert result["order_id"] is not None

        # Verify use case was created and executed
        mock_use_case_factory.create_place_order_use_case.assert_called_once_with(
            broker_coordinator.broker
        )
        mock_use_case.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_place_order_with_prices(self, broker_coordinator, mock_use_case_factory):
        """Test order placement with limit and stop prices."""
        # Setup mock use case
        mock_use_case = AsyncMock(spec=PlaceOrderUseCase)
        mock_response = PlaceOrderResponse(
            success=True,
            order_id=uuid4(),
            broker_order_id="BROKER456",
            status="pending",
            request_id=uuid4(),
        )
        mock_use_case.execute.return_value = mock_response
        mock_use_case_factory.create_place_order_use_case.return_value = mock_use_case

        # Create order request with prices
        order_request = {
            "portfolio_id": str(uuid4()),
            "symbol": "AAPL",
            "side": "sell",
            "order_type": "stop_limit",
            "quantity": 100,
            "limit_price": 148.00,
            "stop_price": 149.00,
            "time_in_force": "gtc",
            "strategy_id": "momentum_001",
        }

        # Execute
        result = await broker_coordinator.place_order(order_request)

        # Assert
        assert result["success"] is True

        # Verify request was properly converted
        call_args = mock_use_case.execute.call_args[0][0]
        assert call_args.limit_price == 148.00
        assert call_args.stop_price == 149.00
        assert call_args.time_in_force == "gtc"
        assert call_args.strategy_id == "momentum_001"

    @pytest.mark.asyncio
    async def test_place_order_failure(self, broker_coordinator, mock_use_case_factory):
        """Test order placement failure."""
        # Setup mock use case
        mock_use_case = AsyncMock(spec=PlaceOrderUseCase)
        mock_response = PlaceOrderResponse(
            success=False, error="Insufficient funds", request_id=uuid4()
        )
        mock_use_case.execute.return_value = mock_response
        mock_use_case_factory.create_place_order_use_case.return_value = mock_use_case

        # Create order request
        order_request = {
            "portfolio_id": str(uuid4()),
            "symbol": "AAPL",
            "side": "buy",
            "order_type": "market",
            "quantity": 1000,
        }

        # Execute
        result = await broker_coordinator.place_order(order_request)

        # Assert
        assert result["success"] is False
        assert result["error"] == "Insufficient funds"
        assert result["order_id"] is None

    @pytest.mark.asyncio
    async def test_cancel_order_success(self, broker_coordinator, mock_use_case_factory):
        """Test successful order cancellation."""
        # Setup mock use case
        mock_use_case = AsyncMock(spec=CancelOrderUseCase)
        mock_response = CancelOrderResponse(
            success=True, cancelled=True, final_status="cancelled", request_id=uuid4()
        )
        mock_use_case.execute.return_value = mock_response
        mock_use_case_factory.create_cancel_order_use_case.return_value = mock_use_case

        order_id = uuid4()

        # Execute
        result = await broker_coordinator.cancel_order(order_id, "User request")

        # Assert
        assert result["success"] is True
        assert result["cancelled"] is True
        assert result["final_status"] == "cancelled"

        # Verify use case was created and executed
        mock_use_case_factory.create_cancel_order_use_case.assert_called_once_with(
            broker_coordinator.broker
        )
        call_args = mock_use_case.execute.call_args[0][0]
        assert call_args.order_id == order_id
        assert call_args.reason == "User request"

    @pytest.mark.asyncio
    async def test_cancel_order_failure(self, broker_coordinator, mock_use_case_factory):
        """Test order cancellation failure."""
        # Setup mock use case
        mock_use_case = AsyncMock(spec=CancelOrderUseCase)
        mock_response = CancelOrderResponse(
            success=False, cancelled=False, error="Order already filled", request_id=uuid4()
        )
        mock_use_case.execute.return_value = mock_response
        mock_use_case_factory.create_cancel_order_use_case.return_value = mock_use_case

        order_id = uuid4()

        # Execute
        result = await broker_coordinator.cancel_order(order_id)

        # Assert
        assert result["success"] is False
        assert result["cancelled"] is False
        assert result["error"] == "Order already filled"

    @pytest.mark.asyncio
    async def test_update_market_price(
        self, broker_coordinator, mock_broker, mock_use_case_factory
    ):
        """Test market price update."""
        # Setup mock use case
        mock_use_case = AsyncMock()
        mock_response = UpdateMarketPriceResponse(
            success=True, orders_triggered=2, orders_filled=1, request_id=uuid4()
        )
        mock_use_case.execute.return_value = mock_response
        mock_use_case_factory.create_update_market_price_use_case.return_value = mock_use_case

        # Execute
        result = await broker_coordinator.update_market_price("AAPL", Decimal("155.00"))

        # Assert
        assert result["success"] is True
        assert result["orders_triggered"] == 2
        assert result["orders_filled"] == 1

        # Verify market price was updated
        assert broker_coordinator.market_prices["AAPL"] == Decimal("155.00")
        mock_broker.set_market_price.assert_called_once_with("AAPL", Decimal("155.00"))

        # Verify use case was executed
        call_args = mock_use_case.execute.call_args[0][0]
        assert call_args.symbol == "AAPL"
        assert call_args.price == Decimal("155.00")

    @pytest.mark.asyncio
    async def test_update_market_price_no_broker_method(
        self, broker_coordinator, mock_broker, mock_use_case_factory
    ):
        """Test market price update when broker doesn't have set_market_price."""
        # Remove set_market_price method
        delattr(mock_broker, "set_market_price")

        # Setup mock use case
        mock_use_case = AsyncMock()
        mock_response = UpdateMarketPriceResponse(
            success=True, orders_triggered=0, orders_filled=0, request_id=uuid4()
        )
        mock_use_case.execute.return_value = mock_response
        mock_use_case_factory.create_update_market_price_use_case.return_value = mock_use_case

        # Execute
        result = await broker_coordinator.update_market_price("AAPL", Decimal("155.00"))

        # Assert
        assert result["success"] is True
        assert broker_coordinator.market_prices["AAPL"] == Decimal("155.00")
        # No error should occur when set_market_price doesn't exist

    @pytest.mark.asyncio
    async def test_process_order_fill_complete(self, broker_coordinator, mock_use_case_factory):
        """Test processing complete order fill."""
        # Setup mock use case
        mock_use_case = AsyncMock()
        mock_response = ProcessOrderFillResponse(
            success=True,
            filled=True,
            fill_price=Decimal("150.00"),
            fill_quantity=100,
            commission=Decimal("10.00"),
            position_id=uuid4(),
            request_id=uuid4(),
        )
        mock_use_case.execute.return_value = mock_response
        mock_use_case_factory.create_process_fill_use_case.return_value = mock_use_case

        order_id = uuid4()

        # Execute
        result = await broker_coordinator.process_order_fill(
            order_id,
            Decimal("150.00"),
            None,  # Complete fill
        )

        # Assert
        assert result["success"] is True
        assert result["filled"] is True
        assert result["fill_price"] == Decimal("150.00")
        assert result["fill_quantity"] == 100
        assert result["commission"] == Decimal("10.00")
        assert result["position_id"] is not None

        # Verify use case was executed
        call_args = mock_use_case.execute.call_args[0][0]
        assert call_args.order_id == order_id
        assert call_args.fill_price == Decimal("150.00")
        assert call_args.fill_quantity is None

    @pytest.mark.asyncio
    async def test_process_order_fill_partial(self, broker_coordinator, mock_use_case_factory):
        """Test processing partial order fill."""
        # Setup mock use case
        mock_use_case = AsyncMock()
        mock_response = ProcessOrderFillResponse(
            success=True,
            filled=True,
            fill_price=Decimal("150.00"),
            fill_quantity=50,
            commission=Decimal("5.00"),
            request_id=uuid4(),
        )
        mock_use_case.execute.return_value = mock_response
        mock_use_case_factory.create_process_fill_use_case.return_value = mock_use_case

        order_id = uuid4()

        # Execute
        result = await broker_coordinator.process_order_fill(
            order_id,
            Decimal("150.00"),
            50,  # Partial fill
        )

        # Assert
        assert result["success"] is True
        assert result["fill_quantity"] == 50

        # Verify use case was executed
        call_args = mock_use_case.execute.call_args[0][0]
        assert call_args.fill_quantity == 50

    @pytest.mark.asyncio
    async def test_process_order_fill_failure(self, broker_coordinator, mock_use_case_factory):
        """Test order fill processing failure."""
        # Setup mock use case
        mock_use_case = AsyncMock()
        mock_response = ProcessOrderFillResponse(
            success=False, filled=False, error="Order not found", request_id=uuid4()
        )
        mock_use_case.execute.return_value = mock_response
        mock_use_case_factory.create_process_fill_use_case.return_value = mock_use_case

        order_id = uuid4()

        # Execute
        result = await broker_coordinator.process_order_fill(order_id, Decimal("150.00"))

        # Assert
        assert result["success"] is False
        assert result["filled"] is False
        assert result["error"] == "Order not found"

    @pytest.mark.asyncio
    async def test_get_order_status_success(self, broker_coordinator, mock_use_case_factory):
        """Test getting order status."""
        # Setup mock use case
        mock_use_case = AsyncMock(spec=GetOrderStatusUseCase)
        mock_response = GetOrderStatusResponse(
            success=True,
            status="partially_filled",
            filled_quantity=50,
            average_fill_price=150.25,
            request_id=uuid4(),
        )
        mock_use_case.execute.return_value = mock_response
        mock_use_case_factory.create_get_order_status_use_case.return_value = mock_use_case

        order_id = uuid4()

        # Execute
        result = await broker_coordinator.get_order_status(order_id)

        # Assert
        assert result["success"] is True
        assert result["status"] == "partially_filled"
        assert result["filled_quantity"] == 50
        assert result["average_fill_price"] == 150.25

        # Verify use case was executed
        call_args = mock_use_case.execute.call_args[0][0]
        assert call_args.order_id == order_id

    @pytest.mark.asyncio
    async def test_process_pending_orders(self, broker_coordinator, mock_use_case_factory):
        """Test processing pending orders."""
        # Setup market prices
        broker_coordinator.market_prices = {"AAPL": Decimal("155.00"), "GOOGL": Decimal("2850.00")}

        # Setup mock use case
        mock_use_case = AsyncMock()
        mock_response = ProcessPendingOrdersResponse(
            success=True,
            processed_count=5,
            triggered_orders=["order1", "order2"],
            filled_orders=["order1"],
            request_id=uuid4(),
        )
        mock_use_case.execute.return_value = mock_response
        mock_use_case_factory.create_process_pending_orders_use_case.return_value = mock_use_case

        # Execute
        result = await broker_coordinator.process_pending_orders()

        # Assert
        assert result["success"] is True
        assert result["processed_count"] == 5
        assert result["triggered_orders"] == ["order1", "order2"]
        assert result["filled_orders"] == ["order1"]

        # Verify use case was executed with current prices
        call_args = mock_use_case.execute.call_args[0][0]
        assert call_args.current_prices == broker_coordinator.market_prices

    @pytest.mark.asyncio
    async def test_process_pending_orders_empty_prices(
        self, broker_coordinator, mock_use_case_factory
    ):
        """Test processing pending orders with no market prices."""
        # No market prices set
        broker_coordinator.market_prices = {}

        # Setup mock use case
        mock_use_case = AsyncMock()
        mock_response = ProcessPendingOrdersResponse(
            success=True,
            processed_count=0,
            triggered_orders=[],
            filled_orders=[],
            request_id=uuid4(),
        )
        mock_use_case.execute.return_value = mock_response
        mock_use_case_factory.create_process_pending_orders_use_case.return_value = mock_use_case

        # Execute
        result = await broker_coordinator.process_pending_orders()

        # Assert
        assert result["success"] is True
        assert result["processed_count"] == 0
        assert result["triggered_orders"] == []
        assert result["filled_orders"] == []

    def test_create_get_order_status_use_case_helper(
        self, broker_coordinator, mock_broker, mock_use_case_factory
    ):
        """Test helper method for creating get order status use case."""
        use_case = broker_coordinator.create_get_order_status_use_case(mock_broker)

        assert isinstance(use_case, GetOrderStatusUseCase)
        assert use_case.unit_of_work == mock_use_case_factory.unit_of_work
        assert use_case.broker == mock_broker

    def test_create_process_pending_orders_use_case_helper(
        self, broker_coordinator, mock_use_case_factory
    ):
        """Test helper method for creating process pending orders use case."""
        use_case = broker_coordinator.create_process_pending_orders_use_case()

        assert isinstance(use_case, ProcessPendingOrdersUseCase)
        assert use_case.unit_of_work == mock_use_case_factory.unit_of_work
        assert use_case.order_processor == mock_use_case_factory.order_processor
        assert use_case.market_microstructure == mock_use_case_factory.market_microstructure
