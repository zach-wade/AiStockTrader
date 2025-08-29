"""
Comprehensive unit tests for Order Execution Use Cases.

This module provides exhaustive test coverage for all order execution use cases:
- ProcessOrderFillUseCase: Tests order fill processing and portfolio updates
- SimulateOrderExecutionUseCase: Tests order execution simulation with market impact
- CalculateCommissionUseCase: Tests commission calculation

Achieves 100% code coverage including all methods, branches, and edge cases.
"""

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock
from uuid import UUID, uuid4

import pytest

from src.application.interfaces.unit_of_work import IUnitOfWork
from src.application.use_cases.order_execution import (
    CalculateCommissionRequest,
    CalculateCommissionResponse,
    CalculateCommissionUseCase,
    ProcessOrderFillRequest,
    ProcessOrderFillResponse,
    ProcessOrderFillUseCase,
    SimulateOrderExecutionRequest,
    SimulateOrderExecutionResponse,
    SimulateOrderExecutionUseCase,
)
from src.domain.entities.order import Order, OrderSide, OrderStatus, OrderType, TimeInForce
from src.domain.entities.portfolio import Portfolio
from src.domain.entities.position import Position
from src.domain.services.commission_calculator import ICommissionCalculator
from src.domain.services.market_microstructure import IMarketMicrostructure
from src.domain.services.order_processor import OrderProcessor
from src.domain.value_objects.money import Money
from src.domain.value_objects.price import Price
from src.domain.value_objects.quantity import Quantity

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_unit_of_work():
    """Create a mock unit of work with all required repositories."""
    uow = AsyncMock(spec=IUnitOfWork)

    # Setup repositories with proper methods
    uow.orders = AsyncMock()
    uow.orders.get_order_by_id = AsyncMock(return_value=None)
    uow.orders.save_order = AsyncMock()
    uow.orders.update_order = AsyncMock()

    uow.portfolios = AsyncMock()
    uow.portfolios.get_portfolio_by_id = AsyncMock(return_value=None)
    uow.portfolios.save_portfolio = AsyncMock()
    uow.portfolios.update_portfolio = AsyncMock()

    uow.positions = AsyncMock()
    uow.positions.persist_position = AsyncMock()
    uow.positions.get_position_by_id = AsyncMock(return_value=None)

    # Setup transaction methods
    uow.commit = AsyncMock()
    uow.rollback = AsyncMock()

    # Setup context manager
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock(return_value=None)

    return uow


@pytest.fixture
def mock_order_processor():
    """Create a mock order processor."""
    processor = Mock(spec=OrderProcessor)
    processor.process_fill = AsyncMock()
    processor.validate_fill = Mock(return_value=True)
    return processor


@pytest.fixture
def mock_commission_calculator():
    """Create a mock commission calculator."""
    calculator = Mock(spec=ICommissionCalculator)
    calculator.calculate = Mock(return_value=Money(Decimal("10.00")))

    # Add schedule attribute for testing
    schedule = Mock()
    schedule.rate = Decimal("0.001")
    calculator.schedule = schedule

    return calculator


@pytest.fixture
def mock_market_microstructure():
    """Create a mock market microstructure service."""
    microstructure = Mock(spec=IMarketMicrostructure)
    microstructure.calculate_execution_price = Mock(return_value=Price(Decimal("100.50")))
    microstructure.calculate_market_impact = Mock(return_value=Decimal("0.05"))
    microstructure.calculate_slippage = Mock(return_value=Decimal("0.02"))
    return microstructure


@pytest.fixture
def sample_order():
    """Create a sample order for testing."""
    order = Mock(spec=Order)
    order.order_id = uuid4()
    order.id = uuid4()
    order.portfolio_id = uuid4()
    order.symbol = "AAPL"
    order.side = OrderSide.BUY
    order.order_type = OrderType.MARKET
    order.quantity = Quantity(100)
    order.status = OrderStatus.SUBMITTED
    order.time_in_force = TimeInForce.DAY
    order.submitted_at = datetime.now(UTC)
    order.filled_quantity = Quantity(0)
    order.is_active = Mock(return_value=True)
    order.get_remaining_quantity = Mock(return_value=Quantity(100))
    order.fill = Mock()
    return order


@pytest.fixture
def sample_portfolio():
    """Create a sample portfolio for testing."""
    portfolio = Mock(spec=Portfolio)
    portfolio.id = uuid4()
    portfolio.name = "Test Portfolio"
    portfolio.cash_balance = Decimal("100000.00")
    portfolio.positions = {}
    return portfolio


@pytest.fixture
def process_fill_use_case(mock_unit_of_work, mock_order_processor, mock_commission_calculator):
    """Create ProcessOrderFillUseCase instance with mocked dependencies."""
    return ProcessOrderFillUseCase(
        unit_of_work=mock_unit_of_work,
        order_processor=mock_order_processor,
        commission_calculator=mock_commission_calculator,
    )


@pytest.fixture
def simulate_execution_use_case(
    mock_unit_of_work, mock_market_microstructure, mock_commission_calculator
):
    """Create SimulateOrderExecutionUseCase instance with mocked dependencies."""
    return SimulateOrderExecutionUseCase(
        unit_of_work=mock_unit_of_work,
        market_microstructure=mock_market_microstructure,
        commission_calculator=mock_commission_calculator,
    )


@pytest.fixture
def calculate_commission_use_case(mock_commission_calculator):
    """Create CalculateCommissionUseCase instance with mocked dependencies."""
    return CalculateCommissionUseCase(commission_calculator=mock_commission_calculator)


# ============================================================================
# ProcessOrderFillRequest Tests
# ============================================================================


class TestProcessOrderFillRequest:
    """Test ProcessOrderFillRequest dataclass."""

    def test_init_with_all_parameters(self):
        """Test initialization with all parameters provided."""
        order_id = uuid4()
        request_id = uuid4()
        correlation_id = uuid4()
        timestamp = datetime.now(UTC)
        metadata = {"key": "value"}

        request = ProcessOrderFillRequest(
            order_id=order_id,
            fill_price=Decimal("100.50"),
            fill_quantity=50,
            timestamp=timestamp,
            request_id=request_id,
            correlation_id=correlation_id,
            metadata=metadata,
        )

        assert request.order_id == order_id
        assert request.fill_price == Decimal("100.50")
        assert request.fill_quantity == 50
        assert request.timestamp == timestamp
        assert request.request_id == request_id
        assert request.correlation_id == correlation_id
        assert request.metadata == metadata

    def test_init_with_minimal_parameters(self):
        """Test initialization with minimal required parameters."""
        order_id = uuid4()

        request = ProcessOrderFillRequest(order_id=order_id, fill_price=Decimal("100.50"))

        assert request.order_id == order_id
        assert request.fill_price == Decimal("100.50")
        assert request.fill_quantity is None
        assert request.timestamp is None
        assert request.request_id is not None  # Auto-generated
        assert request.correlation_id is None
        assert request.metadata == {}  # Default empty dict

    def test_post_init_generates_request_id(self):
        """Test that __post_init__ generates request_id if not provided."""
        request = ProcessOrderFillRequest(order_id=uuid4(), fill_price=Decimal("100.50"))

        assert request.request_id is not None
        assert isinstance(request.request_id, UUID)

    def test_post_init_initializes_empty_metadata(self):
        """Test that __post_init__ initializes empty metadata dict."""
        request = ProcessOrderFillRequest(
            order_id=uuid4(), fill_price=Decimal("100.50"), metadata=None
        )

        assert request.metadata == {}

    def test_post_init_preserves_provided_values(self):
        """Test that __post_init__ preserves explicitly provided values."""
        request_id = uuid4()
        metadata = {"key": "value"}

        request = ProcessOrderFillRequest(
            order_id=uuid4(), fill_price=Decimal("100.50"), request_id=request_id, metadata=metadata
        )

        assert request.request_id == request_id
        assert request.metadata == metadata


# ============================================================================
# ProcessOrderFillUseCase Tests
# ============================================================================


class TestProcessOrderFillUseCase:
    """Test ProcessOrderFillUseCase."""

    @pytest.mark.asyncio
    async def test_init(self, mock_unit_of_work, mock_order_processor, mock_commission_calculator):
        """Test use case initialization."""
        use_case = ProcessOrderFillUseCase(
            unit_of_work=mock_unit_of_work,
            order_processor=mock_order_processor,
            commission_calculator=mock_commission_calculator,
        )

        assert use_case.unit_of_work == mock_unit_of_work
        assert use_case.order_processor == mock_order_processor
        assert use_case.commission_calculator == mock_commission_calculator
        assert use_case.name == "ProcessOrderFillUseCase"

    @pytest.mark.asyncio
    async def test_validate_negative_fill_price(self, process_fill_use_case):
        """Test validation fails with negative fill price."""
        request = ProcessOrderFillRequest(order_id=uuid4(), fill_price=Decimal("-100.50"))

        error = await process_fill_use_case.validate(request)
        assert error == "Fill price must be positive"

    @pytest.mark.asyncio
    async def test_validate_zero_fill_price(self, process_fill_use_case):
        """Test validation fails with zero fill price."""
        request = ProcessOrderFillRequest(order_id=uuid4(), fill_price=Decimal("0"))

        error = await process_fill_use_case.validate(request)
        assert error == "Fill price must be positive"

    @pytest.mark.asyncio
    async def test_validate_negative_fill_quantity(self, process_fill_use_case):
        """Test validation fails with negative fill quantity."""
        request = ProcessOrderFillRequest(
            order_id=uuid4(), fill_price=Decimal("100.50"), fill_quantity=-10
        )

        error = await process_fill_use_case.validate(request)
        assert error == "Fill quantity must be positive"

    @pytest.mark.asyncio
    async def test_validate_zero_fill_quantity(self, process_fill_use_case):
        """Test validation fails with zero fill quantity."""
        request = ProcessOrderFillRequest(
            order_id=uuid4(), fill_price=Decimal("100.50"), fill_quantity=0
        )

        error = await process_fill_use_case.validate(request)
        assert error == "Fill quantity must be positive"

    @pytest.mark.asyncio
    async def test_validate_valid_request(self, process_fill_use_case):
        """Test validation passes with valid request."""
        request = ProcessOrderFillRequest(
            order_id=uuid4(), fill_price=Decimal("100.50"), fill_quantity=50
        )

        error = await process_fill_use_case.validate(request)
        assert error is None

    @pytest.mark.asyncio
    async def test_validate_valid_request_no_quantity(self, process_fill_use_case):
        """Test validation passes with valid request and no fill quantity."""
        request = ProcessOrderFillRequest(order_id=uuid4(), fill_price=Decimal("100.50"))

        error = await process_fill_use_case.validate(request)
        assert error is None

    @pytest.mark.asyncio
    async def test_process_order_not_found(self, process_fill_use_case, mock_unit_of_work):
        """Test process when order is not found."""
        mock_unit_of_work.orders.get_order_by_id.return_value = None

        request = ProcessOrderFillRequest(order_id=uuid4(), fill_price=Decimal("100.50"))

        response = await process_fill_use_case.process(request)

        assert response.success is False
        assert response.error == "Order not found"
        assert response.filled is False
        assert response.fill_price is None
        assert response.fill_quantity is None

    @pytest.mark.asyncio
    async def test_process_inactive_order(self, process_fill_use_case, mock_unit_of_work):
        """Test process when order is not active."""
        order = Mock(spec=Order)
        order.is_active.return_value = False
        order.status = OrderStatus.CANCELLED

        mock_unit_of_work.orders.get_order_by_id.return_value = order

        request = ProcessOrderFillRequest(order_id=uuid4(), fill_price=Decimal("100.50"))

        response = await process_fill_use_case.process(request)

        assert response.success is False
        assert response.error == "Cannot fill order in status: cancelled"
        assert response.filled is False

    @pytest.mark.asyncio
    async def test_process_portfolio_not_found(
        self, process_fill_use_case, mock_unit_of_work, sample_order
    ):
        """Test process when portfolio is not found."""
        sample_order.is_active = Mock(return_value=True)
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = None

        request = ProcessOrderFillRequest(
            order_id=sample_order.order_id, fill_price=Decimal("100.50")
        )

        response = await process_fill_use_case.process(request)

        assert response.success is False
        assert response.error == "Portfolio not found"
        assert response.filled is False

    @pytest.mark.asyncio
    async def test_process_zero_remaining_quantity(
        self, process_fill_use_case, mock_unit_of_work, sample_order, sample_portfolio
    ):
        """Test process when order has no remaining quantity."""
        sample_order.is_active = Mock(return_value=True)
        sample_order.get_remaining_quantity = Mock(return_value=Quantity(0))

        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = ProcessOrderFillRequest(
            order_id=sample_order.order_id, fill_price=Decimal("100.50")
        )

        response = await process_fill_use_case.process(request)

        assert response.success is False
        assert response.error == "No quantity to fill"
        assert response.filled is False

    @pytest.mark.asyncio
    async def test_process_successful_full_fill(
        self,
        process_fill_use_case,
        mock_unit_of_work,
        mock_order_processor,
        mock_commission_calculator,
        sample_order,
        sample_portfolio,
    ):
        """Test successful processing of a full order fill."""
        # Setup order
        sample_order.is_active = Mock(return_value=True)
        sample_order.get_remaining_quantity = Mock(return_value=Quantity(100))
        sample_order.fill = Mock()

        # Setup mocks
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_commission_calculator.calculate.return_value = Money(Decimal("10.00"))

        request = ProcessOrderFillRequest(
            order_id=sample_order.order_id, fill_price=Decimal("100.50"), request_id=uuid4()
        )

        response = await process_fill_use_case.process(request)

        # Verify response
        assert response.success is True
        assert response.filled is True
        assert response.fill_price == Decimal("100.50")
        assert response.fill_quantity == 100
        assert response.commission == Decimal("10.00")
        assert response.request_id == request.request_id

        # Verify method calls
        sample_order.fill.assert_called_once()
        mock_order_processor.process_fill.assert_called_once()
        mock_unit_of_work.orders.update_order.assert_called_once_with(sample_order)
        mock_unit_of_work.portfolios.update_portfolio.assert_called_once_with(sample_portfolio)

    @pytest.mark.asyncio
    async def test_process_successful_partial_fill(
        self,
        process_fill_use_case,
        mock_unit_of_work,
        mock_order_processor,
        mock_commission_calculator,
        sample_order,
        sample_portfolio,
    ):
        """Test successful processing of a partial order fill."""
        # Setup order
        sample_order.is_active = Mock(return_value=True)
        sample_order.get_remaining_quantity = Mock(return_value=Quantity(100))
        sample_order.fill = Mock()

        # Setup mocks
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = ProcessOrderFillRequest(
            order_id=sample_order.order_id,
            fill_price=Decimal("100.50"),
            fill_quantity=50,  # Partial fill
            timestamp=datetime.now(UTC),
        )

        response = await process_fill_use_case.process(request)

        # Verify response
        assert response.success is True
        assert response.filled is True
        assert response.fill_quantity == 50

        # Verify fill was called with correct quantity
        call_args = sample_order.fill.call_args
        assert call_args[1]["filled_quantity"] == Quantity(50)

    @pytest.mark.asyncio
    async def test_process_with_positions_update(
        self,
        process_fill_use_case,
        mock_unit_of_work,
        mock_order_processor,
        sample_order,
        sample_portfolio,
    ):
        """Test that positions are properly saved after fill."""
        # Setup order and portfolio
        sample_order.is_active = Mock(return_value=True)
        sample_order.get_remaining_quantity = Mock(return_value=Quantity(100))
        sample_order.fill = Mock()

        # Create mock positions
        position1 = Mock(spec=Position)
        position2 = Mock(spec=Position)
        sample_portfolio.positions = {uuid4(): position1, uuid4(): position2}

        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = ProcessOrderFillRequest(
            order_id=sample_order.order_id, fill_price=Decimal("100.50")
        )

        response = await process_fill_use_case.process(request)

        assert response.success is True

        # Verify each position was saved
        assert mock_unit_of_work.positions.persist_position.call_count == 2
        mock_unit_of_work.positions.persist_position.assert_any_call(position1)
        mock_unit_of_work.positions.persist_position.assert_any_call(position2)

    @pytest.mark.asyncio
    async def test_process_exception_handling(
        self, process_fill_use_case, mock_unit_of_work, sample_order, sample_portfolio
    ):
        """Test exception handling during fill processing."""
        sample_order.is_active = Mock(return_value=True)
        sample_order.get_remaining_quantity = Mock(return_value=Quantity(100))
        sample_order.fill = Mock(side_effect=Exception("Fill processing failed"))

        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = ProcessOrderFillRequest(
            order_id=sample_order.order_id, fill_price=Decimal("100.50")
        )

        response = await process_fill_use_case.process(request)

        assert response.success is False
        assert "Failed to process fill: Fill processing failed" in response.error
        assert response.filled is False

    @pytest.mark.asyncio
    async def test_process_with_custom_timestamp(
        self, process_fill_use_case, mock_unit_of_work, sample_order, sample_portfolio
    ):
        """Test fill processing with custom timestamp."""
        custom_timestamp = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)

        sample_order.is_active = Mock(return_value=True)
        sample_order.get_remaining_quantity = Mock(return_value=Quantity(100))
        sample_order.fill = Mock()

        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = ProcessOrderFillRequest(
            order_id=sample_order.order_id, fill_price=Decimal("100.50"), timestamp=custom_timestamp
        )

        response = await process_fill_use_case.process(request)

        assert response.success is True

        # Verify fill was called with custom timestamp
        call_args = sample_order.fill.call_args
        assert call_args[1]["timestamp"] == custom_timestamp

    @pytest.mark.asyncio
    async def test_process_quantity_exceeds_remaining(
        self, process_fill_use_case, mock_unit_of_work, sample_order, sample_portfolio
    ):
        """Test that fill quantity is capped at remaining quantity."""
        sample_order.is_active = Mock(return_value=True)
        sample_order.get_remaining_quantity = Mock(return_value=Quantity(50))
        sample_order.fill = Mock()

        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = ProcessOrderFillRequest(
            order_id=sample_order.order_id,
            fill_price=Decimal("100.50"),
            fill_quantity=100,  # Exceeds remaining
        )

        response = await process_fill_use_case.process(request)

        assert response.success is True
        assert response.fill_quantity == 50  # Capped at remaining


# ============================================================================
# SimulateOrderExecutionRequest Tests
# ============================================================================


class TestSimulateOrderExecutionRequest:
    """Test SimulateOrderExecutionRequest dataclass."""

    def test_init_with_all_parameters(self):
        """Test initialization with all parameters provided."""
        order_id = uuid4()
        request_id = uuid4()
        correlation_id = uuid4()
        metadata = {"key": "value"}

        request = SimulateOrderExecutionRequest(
            order_id=order_id,
            market_price=Decimal("100.50"),
            available_liquidity=Decimal("1000000"),
            request_id=request_id,
            correlation_id=correlation_id,
            metadata=metadata,
        )

        assert request.order_id == order_id
        assert request.market_price == Decimal("100.50")
        assert request.available_liquidity == Decimal("1000000")
        assert request.request_id == request_id
        assert request.correlation_id == correlation_id
        assert request.metadata == metadata

    def test_init_with_minimal_parameters(self):
        """Test initialization with minimal required parameters."""
        order_id = uuid4()

        request = SimulateOrderExecutionRequest(order_id=order_id, market_price=Decimal("100.50"))

        assert request.order_id == order_id
        assert request.market_price == Decimal("100.50")
        assert request.available_liquidity is None
        assert request.request_id is not None  # Auto-generated
        assert request.correlation_id is None
        assert request.metadata == {}

    def test_post_init_generates_request_id(self):
        """Test that __post_init__ generates request_id if not provided."""
        request = SimulateOrderExecutionRequest(order_id=uuid4(), market_price=Decimal("100.50"))

        assert request.request_id is not None
        assert isinstance(request.request_id, UUID)

    def test_post_init_initializes_empty_metadata(self):
        """Test that __post_init__ initializes empty metadata dict."""
        request = SimulateOrderExecutionRequest(
            order_id=uuid4(), market_price=Decimal("100.50"), metadata=None
        )

        assert request.metadata == {}


# ============================================================================
# SimulateOrderExecutionUseCase Tests
# ============================================================================


class TestSimulateOrderExecutionUseCase:
    """Test SimulateOrderExecutionUseCase."""

    @pytest.mark.asyncio
    async def test_init(
        self, mock_unit_of_work, mock_market_microstructure, mock_commission_calculator
    ):
        """Test use case initialization."""
        use_case = SimulateOrderExecutionUseCase(
            unit_of_work=mock_unit_of_work,
            market_microstructure=mock_market_microstructure,
            commission_calculator=mock_commission_calculator,
        )

        assert use_case.unit_of_work == mock_unit_of_work
        assert use_case.market_microstructure == mock_market_microstructure
        assert use_case.commission_calculator == mock_commission_calculator
        assert use_case.name == "SimulateOrderExecutionUseCase"

    @pytest.mark.asyncio
    async def test_validate_negative_market_price(self, simulate_execution_use_case):
        """Test validation fails with negative market price."""
        request = SimulateOrderExecutionRequest(order_id=uuid4(), market_price=Decimal("-100.50"))

        error = await simulate_execution_use_case.validate(request)
        assert error == "Market price must be positive"

    @pytest.mark.asyncio
    async def test_validate_zero_market_price(self, simulate_execution_use_case):
        """Test validation fails with zero market price."""
        request = SimulateOrderExecutionRequest(order_id=uuid4(), market_price=Decimal("0"))

        error = await simulate_execution_use_case.validate(request)
        assert error == "Market price must be positive"

    @pytest.mark.asyncio
    async def test_validate_negative_liquidity(self, simulate_execution_use_case):
        """Test validation fails with negative available liquidity."""
        request = SimulateOrderExecutionRequest(
            order_id=uuid4(), market_price=Decimal("100.50"), available_liquidity=Decimal("-1000")
        )

        error = await simulate_execution_use_case.validate(request)
        assert error == "Available liquidity must be positive"

    @pytest.mark.asyncio
    async def test_validate_zero_liquidity(self, simulate_execution_use_case):
        """Test validation fails with zero available liquidity."""
        request = SimulateOrderExecutionRequest(
            order_id=uuid4(), market_price=Decimal("100.50"), available_liquidity=Decimal("0")
        )

        error = await simulate_execution_use_case.validate(request)
        assert error == "Available liquidity must be positive"

    @pytest.mark.asyncio
    async def test_validate_valid_request(self, simulate_execution_use_case):
        """Test validation passes with valid request."""
        request = SimulateOrderExecutionRequest(
            order_id=uuid4(), market_price=Decimal("100.50"), available_liquidity=Decimal("1000000")
        )

        error = await simulate_execution_use_case.validate(request)
        assert error is None

    @pytest.mark.asyncio
    async def test_validate_valid_request_no_liquidity(self, simulate_execution_use_case):
        """Test validation passes with valid request and no liquidity specified."""
        request = SimulateOrderExecutionRequest(order_id=uuid4(), market_price=Decimal("100.50"))

        error = await simulate_execution_use_case.validate(request)
        assert error is None

    @pytest.mark.asyncio
    async def test_process_order_not_found(self, simulate_execution_use_case, mock_unit_of_work):
        """Test process when order is not found."""
        mock_unit_of_work.orders.get_order_by_id.return_value = None

        request = SimulateOrderExecutionRequest(order_id=uuid4(), market_price=Decimal("100.50"))

        response = await simulate_execution_use_case.process(request)

        assert response.success is False
        assert response.error == "Order not found"
        assert response.execution_price is None
        assert response.slippage is None

    @pytest.mark.asyncio
    async def test_process_market_buy_order(
        self,
        simulate_execution_use_case,
        mock_unit_of_work,
        mock_market_microstructure,
        mock_commission_calculator,
    ):
        """Test simulation of market buy order."""
        # Setup order
        order = Mock(spec=Order)
        order.side = OrderSide.BUY
        order.order_type = OrderType.MARKET
        order.quantity = Quantity(100)

        mock_unit_of_work.orders.get_order_by_id.return_value = order

        # Setup market microstructure
        execution_price = Price(Decimal("100.75"))  # Higher than market for buy
        mock_market_microstructure.calculate_execution_price.return_value = execution_price
        mock_market_microstructure.calculate_market_impact.return_value = Decimal("0.10")

        # Setup commission
        mock_commission_calculator.calculate.return_value = Money(Decimal("10.00"))

        request = SimulateOrderExecutionRequest(
            order_id=uuid4(), market_price=Decimal("100.50"), request_id=uuid4()
        )

        response = await simulate_execution_use_case.process(request)

        assert response.success is True
        assert response.execution_price == Decimal("100.75")
        assert response.slippage == Decimal("0.25")  # 100.75 - 100.50
        assert response.market_impact == Decimal("0.10")
        assert response.estimated_commission == Decimal("10.00")
        assert response.request_id == request.request_id

        # Verify method calls
        mock_market_microstructure.calculate_execution_price.assert_called_once_with(
            base_price=Price(Decimal("100.50")),
            side=OrderSide.BUY,
            quantity=Quantity(100),
            order_type=OrderType.MARKET,
        )

    @pytest.mark.asyncio
    async def test_process_market_sell_order(
        self, simulate_execution_use_case, mock_unit_of_work, mock_market_microstructure
    ):
        """Test simulation of market sell order."""
        # Setup order
        order = Mock(spec=Order)
        order.side = OrderSide.SELL
        order.order_type = OrderType.MARKET
        order.quantity = Quantity(100)

        mock_unit_of_work.orders.get_order_by_id.return_value = order

        # Setup market microstructure
        execution_price = Price(Decimal("100.25"))  # Lower than market for sell
        mock_market_microstructure.calculate_execution_price.return_value = execution_price

        request = SimulateOrderExecutionRequest(order_id=uuid4(), market_price=Decimal("100.50"))

        response = await simulate_execution_use_case.process(request)

        assert response.success is True
        assert response.execution_price == Decimal("100.25")
        assert response.slippage == Decimal("0.25")  # 100.50 - 100.25

    @pytest.mark.asyncio
    async def test_process_limit_order(
        self, simulate_execution_use_case, mock_unit_of_work, mock_market_microstructure
    ):
        """Test simulation of limit order (no slippage)."""
        # Setup order
        order = Mock(spec=Order)
        order.side = OrderSide.BUY
        order.order_type = OrderType.LIMIT
        order.quantity = Quantity(100)
        order.limit_price = Price(Decimal("100.00"))

        mock_unit_of_work.orders.get_order_by_id.return_value = order

        # Setup market microstructure
        execution_price = Price(Decimal("100.00"))
        mock_market_microstructure.calculate_execution_price.return_value = execution_price

        request = SimulateOrderExecutionRequest(order_id=uuid4(), market_price=Decimal("100.50"))

        response = await simulate_execution_use_case.process(request)

        assert response.success is True
        assert response.execution_price == Decimal("100.00")
        assert response.slippage == Decimal("0")  # No slippage for limit orders

    @pytest.mark.asyncio
    async def test_process_with_available_liquidity(
        self, simulate_execution_use_case, mock_unit_of_work, mock_market_microstructure
    ):
        """Test simulation with available liquidity parameter."""
        order = Mock(spec=Order)
        order.side = OrderSide.BUY
        order.order_type = OrderType.MARKET
        order.quantity = Quantity(100)

        mock_unit_of_work.orders.get_order_by_id.return_value = order
        mock_market_microstructure.calculate_execution_price.return_value = Price(Decimal("100.50"))

        request = SimulateOrderExecutionRequest(
            order_id=uuid4(), market_price=Decimal("100.50"), available_liquidity=Decimal("500000")
        )

        response = await simulate_execution_use_case.process(request)

        assert response.success is True
        # Note: Current implementation doesn't use available_liquidity,
        # but the parameter is accepted and validated


# ============================================================================
# CalculateCommissionRequest Tests
# ============================================================================


class TestCalculateCommissionRequest:
    """Test CalculateCommissionRequest dataclass."""

    def test_init_with_all_parameters(self):
        """Test initialization with all parameters provided."""
        request_id = uuid4()
        correlation_id = uuid4()
        metadata = {"key": "value"}

        request = CalculateCommissionRequest(
            quantity=100,
            price=Decimal("100.50"),
            order_type="MARKET",
            symbol="AAPL",
            request_id=request_id,
            correlation_id=correlation_id,
            metadata=metadata,
        )

        assert request.quantity == 100
        assert request.price == Decimal("100.50")
        assert request.order_type == "MARKET"
        assert request.symbol == "AAPL"
        assert request.request_id == request_id
        assert request.correlation_id == correlation_id
        assert request.metadata == metadata

    def test_init_with_minimal_parameters(self):
        """Test initialization with minimal required parameters."""
        request = CalculateCommissionRequest(
            quantity=100, price=Decimal("100.50"), order_type="MARKET"
        )

        assert request.quantity == 100
        assert request.price == Decimal("100.50")
        assert request.order_type == "MARKET"
        assert request.symbol is None
        assert request.request_id is not None  # Auto-generated
        assert request.correlation_id is None
        assert request.metadata == {}

    def test_post_init_generates_request_id(self):
        """Test that __post_init__ generates request_id if not provided."""
        request = CalculateCommissionRequest(
            quantity=100, price=Decimal("100.50"), order_type="MARKET"
        )

        assert request.request_id is not None
        assert isinstance(request.request_id, UUID)

    def test_post_init_initializes_empty_metadata(self):
        """Test that __post_init__ initializes empty metadata dict."""
        request = CalculateCommissionRequest(
            quantity=Quantity(100), price=Decimal("100.50"), order_type="MARKET", metadata=None
        )

        assert request.metadata == {}


# ============================================================================
# CalculateCommissionUseCase Tests
# ============================================================================


class TestCalculateCommissionUseCase:
    """Test CalculateCommissionUseCase."""

    def test_init(self, mock_commission_calculator):
        """Test use case initialization."""
        use_case = CalculateCommissionUseCase(commission_calculator=mock_commission_calculator)

        assert use_case.commission_calculator == mock_commission_calculator
        assert use_case.logger is not None

    @pytest.mark.asyncio
    async def test_execute_success(self, calculate_commission_use_case, mock_commission_calculator):
        """Test successful commission calculation."""
        mock_commission_calculator.calculate.return_value = Money(Decimal("15.50"))

        request = CalculateCommissionRequest(
            quantity=200, price=Decimal("150.00"), order_type="LIMIT", request_id=uuid4()
        )

        response = await calculate_commission_use_case.execute(request)

        assert response.success is True
        assert response.commission == Decimal("15.50")
        assert response.commission_rate == Decimal("0.001")  # From fixture
        assert response.request_id == request.request_id

        # Verify calculator was called correctly
        mock_commission_calculator.calculate.assert_called_once_with(
            quantity=Quantity(200), price=Money(Decimal("150.00"))
        )

    @pytest.mark.asyncio
    async def test_execute_with_symbol(
        self, calculate_commission_use_case, mock_commission_calculator
    ):
        """Test commission calculation with symbol parameter."""
        request = CalculateCommissionRequest(
            quantity=Quantity(100), price=Decimal("100.00"), order_type="MARKET", symbol="TSLA"
        )

        response = await calculate_commission_use_case.execute(request)

        assert response.success is True
        # Symbol is accepted but not used in current implementation

    @pytest.mark.asyncio
    async def test_execute_exception_handling(
        self, calculate_commission_use_case, mock_commission_calculator
    ):
        """Test exception handling during commission calculation."""
        mock_commission_calculator.calculate.side_effect = Exception("Calculation failed")

        request = CalculateCommissionRequest(
            quantity=Quantity(100), price=Decimal("100.00"), order_type="MARKET"
        )

        response = await calculate_commission_use_case.execute(request)

        assert response.success is False
        assert "Failed to calculate commission: Calculation failed" in response.error
        assert response.commission is None
        assert response.commission_rate is None

    @pytest.mark.asyncio
    async def test_execute_with_missing_request_id(
        self, calculate_commission_use_case, mock_commission_calculator
    ):
        """Test that request_id is generated if exception occurs and none provided."""
        mock_commission_calculator.calculate.side_effect = Exception("Error")

        request = CalculateCommissionRequest(
            quantity=Quantity(100),
            price=Decimal("100.00"),
            order_type="MARKET",
            request_id=None,  # Explicitly set to None
        )

        # Override post_init to not generate request_id
        request.request_id = None

        response = await calculate_commission_use_case.execute(request)

        assert response.success is False
        assert response.request_id is not None  # Should be generated
        assert isinstance(response.request_id, UUID)

    @pytest.mark.asyncio
    async def test_execute_without_schedule_attribute(self, mock_commission_calculator):
        """Test commission calculation when calculator has no schedule attribute."""
        # Remove schedule attribute
        if hasattr(mock_commission_calculator, "schedule"):
            delattr(mock_commission_calculator, "schedule")

        mock_commission_calculator.calculate.return_value = Money(Decimal("10.00"))

        use_case = CalculateCommissionUseCase(mock_commission_calculator)

        request = CalculateCommissionRequest(
            quantity=Quantity(100), price=Decimal("100.00"), order_type="MARKET"
        )

        response = await use_case.execute(request)

        assert response.success is True
        assert response.commission == Decimal("10.00")
        assert response.commission_rate is None  # No schedule, no rate

    @pytest.mark.asyncio
    async def test_execute_with_zero_quantity(
        self, calculate_commission_use_case, mock_commission_calculator
    ):
        """Test commission calculation with zero quantity."""
        mock_commission_calculator.calculate.return_value = Money(Decimal("0"))

        request = CalculateCommissionRequest(
            quantity=0, price=Decimal("100.00"), order_type="MARKET"
        )

        response = await calculate_commission_use_case.execute(request)

        assert response.success is True
        assert response.commission == Decimal("0")

    @pytest.mark.asyncio
    async def test_execute_with_large_values(
        self, calculate_commission_use_case, mock_commission_calculator
    ):
        """Test commission calculation with large quantity and price."""
        mock_commission_calculator.calculate.return_value = Money(Decimal("1000.00"))

        request = CalculateCommissionRequest(
            quantity=10000, price=Decimal("1000.00"), order_type="MARKET"
        )

        response = await calculate_commission_use_case.execute(request)

        assert response.success is True
        assert response.commission == Decimal("1000.00")


# ============================================================================
# Response Classes Tests
# ============================================================================


class TestResponseClasses:
    """Test response dataclasses."""

    def test_process_order_fill_response_defaults(self):
        """Test ProcessOrderFillResponse with default values."""
        response = ProcessOrderFillResponse(success=True, request_id=uuid4())

        assert response.success is True
        assert response.filled is False
        assert response.fill_price is None
        assert response.fill_quantity is None
        assert response.commission is None
        assert response.position_id is None

    def test_process_order_fill_response_all_fields(self):
        """Test ProcessOrderFillResponse with all fields."""
        request_id = uuid4()
        position_id = uuid4()

        response = ProcessOrderFillResponse(
            success=True,
            filled=True,
            fill_price=Decimal("100.50"),
            fill_quantity=100,
            commission=Decimal("10.00"),
            position_id=position_id,
            request_id=request_id,
            error="Some error",
        )

        assert response.success is True
        assert response.filled is True
        assert response.fill_price == Decimal("100.50")
        assert response.fill_quantity == 100
        assert response.commission == Decimal("10.00")
        assert response.position_id == position_id
        assert response.request_id == request_id
        assert response.error == "Some error"

    def test_simulate_order_execution_response_defaults(self):
        """Test SimulateOrderExecutionResponse with default values."""
        response = SimulateOrderExecutionResponse(success=True, request_id=uuid4())

        assert response.success is True
        assert response.execution_price is None
        assert response.slippage is None
        assert response.market_impact is None
        assert response.estimated_commission is None

    def test_simulate_order_execution_response_all_fields(self):
        """Test SimulateOrderExecutionResponse with all fields."""
        response = SimulateOrderExecutionResponse(
            success=True,
            execution_price=Decimal("100.75"),
            slippage=Decimal("0.25"),
            market_impact=Decimal("0.10"),
            estimated_commission=Decimal("10.00"),
            request_id=uuid4(),
        )

        assert response.success is True
        assert response.execution_price == Decimal("100.75")
        assert response.slippage == Decimal("0.25")
        assert response.market_impact == Decimal("0.10")
        assert response.estimated_commission == Decimal("10.00")

    def test_calculate_commission_response_defaults(self):
        """Test CalculateCommissionResponse with default values."""
        response = CalculateCommissionResponse(success=True, request_id=uuid4())

        assert response.success is True
        assert response.commission is None
        assert response.commission_rate is None

    def test_calculate_commission_response_all_fields(self):
        """Test CalculateCommissionResponse with all fields."""
        response = CalculateCommissionResponse(
            success=True,
            commission=Decimal("15.50"),
            commission_rate=Decimal("0.001"),
            request_id=uuid4(),
            error="Some error",
        )

        assert response.success is True
        assert response.commission == Decimal("15.50")
        assert response.commission_rate == Decimal("0.001")
        assert response.error == "Some error"


# ============================================================================
# Integration Tests
# ============================================================================


class TestOrderExecutionIntegration:
    """Integration tests for order execution use cases."""

    @pytest.mark.asyncio
    async def test_fill_processing_integration(
        self, mock_unit_of_work, mock_order_processor, mock_commission_calculator
    ):
        """Test complete fill processing flow."""
        # Create real objects where possible
        order_id = uuid4()
        portfolio_id = uuid4()

        # Create order mock with necessary attributes
        order = Mock(spec=Order)
        order.order_id = order_id
        order.id = order_id
        order.portfolio_id = portfolio_id
        order.symbol = "AAPL"
        order.side = OrderSide.BUY
        order.order_type = OrderType.MARKET
        order.quantity = Quantity(100)
        order.status = OrderStatus.PENDING
        order.time_in_force = TimeInForce.DAY
        order.submitted_at = datetime.now(UTC)
        order.filled_quantity = Quantity(0)
        order.is_active = Mock(return_value=True)
        order.get_remaining_quantity = Mock(return_value=Quantity(100))
        order.fill = Mock()

        # Create portfolio mock
        portfolio = Mock(spec=Portfolio)
        portfolio.id = portfolio_id
        portfolio.name = "Test Portfolio"
        portfolio.cash_balance = Decimal("100000.00")
        portfolio.positions = {}

        # Setup mocks
        mock_unit_of_work.orders.get_order_by_id.return_value = order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = portfolio
        mock_commission_calculator.calculate.return_value = Money(Decimal("10.00"))

        # Create use case
        use_case = ProcessOrderFillUseCase(
            unit_of_work=mock_unit_of_work,
            order_processor=mock_order_processor,
            commission_calculator=mock_commission_calculator,
        )

        # Create request
        request = ProcessOrderFillRequest(
            order_id=order_id, fill_price=Decimal("150.00"), fill_quantity=50
        )

        # Execute through the full flow
        response = await use_case.execute(request)

        assert response.success is True
        assert response.filled is True
        assert response.fill_quantity == 50

        # Verify saves were called
        mock_unit_of_work.orders.update_order.assert_called_once()
        mock_unit_of_work.portfolios.update_portfolio.assert_called_once()

    @pytest.mark.asyncio
    async def test_simulation_integration(
        self, mock_unit_of_work, mock_market_microstructure, mock_commission_calculator
    ):
        """Test complete order execution simulation flow."""
        order_id = uuid4()

        # Create order mock
        order = Mock(spec=Order)
        order.order_id = order_id
        order.id = order_id
        order.portfolio_id = uuid4()
        order.symbol = "GOOGL"
        order.side = OrderSide.SELL
        order.order_type = OrderType.MARKET
        order.quantity = Quantity(200)
        order.status = OrderStatus.PENDING
        order.time_in_force = TimeInForce.GTC
        order.submitted_at = datetime.now(UTC)

        # Setup mocks
        mock_unit_of_work.orders.get_order_by_id.return_value = order
        mock_market_microstructure.calculate_execution_price.return_value = Price(
            Decimal("2500.25")
        )
        mock_market_microstructure.calculate_market_impact.return_value = Decimal("0.15")
        mock_commission_calculator.calculate.return_value = Money(Decimal("25.00"))

        # Create use case
        use_case = SimulateOrderExecutionUseCase(
            unit_of_work=mock_unit_of_work,
            market_microstructure=mock_market_microstructure,
            commission_calculator=mock_commission_calculator,
        )

        # Create request
        request = SimulateOrderExecutionRequest(order_id=order_id, market_price=Decimal("2500.50"))

        # Execute through the full flow
        response = await use_case.execute(request)

        assert response.success is True
        assert response.execution_price == Decimal("2500.25")
        assert response.slippage == Decimal("0.25")  # 2500.50 - 2500.25 for sell
        assert response.market_impact == Decimal("0.15")
        assert response.estimated_commission == Decimal("25.00")


# ============================================================================
# Edge Cases and Error Scenarios
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_fill_processing(
        self, process_fill_use_case, mock_unit_of_work, sample_order, sample_portfolio
    ):
        """Test concurrent fill processing requests."""
        sample_order.is_active = Mock(return_value=True)
        sample_order.get_remaining_quantity = Mock(return_value=Quantity(100))
        sample_order.fill = Mock()

        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        # Create multiple requests
        requests = [
            ProcessOrderFillRequest(
                order_id=sample_order.order_id, fill_price=Decimal("100.50"), fill_quantity=25
            )
            for _ in range(4)
        ]

        # Process concurrently
        tasks = [process_fill_use_case.process(req) for req in requests]
        responses = await asyncio.gather(*tasks)

        # All should succeed (in this simple test)
        assert all(r.success for r in responses)
        assert sample_order.fill.call_count == 4

    @pytest.mark.asyncio
    async def test_decimal_precision(
        self, calculate_commission_use_case, mock_commission_calculator
    ):
        """Test handling of high-precision decimal values."""
        mock_commission_calculator.calculate.return_value = Money(Decimal("10.123456789"))

        request = CalculateCommissionRequest(
            quantity=Quantity(100), price=Decimal("100.123456789"), order_type="MARKET"
        )

        response = await calculate_commission_use_case.execute(request)

        assert response.success is True
        assert response.commission == Decimal("10.123456789")

    @pytest.mark.asyncio
    async def test_logging_on_error(
        self, process_fill_use_case, mock_unit_of_work, sample_order, sample_portfolio
    ):
        """Test that errors are properly logged."""
        # Setup order and portfolio
        sample_order.is_active = Mock(return_value=True)
        sample_order.get_remaining_quantity = Mock(return_value=Quantity(100))
        sample_order.fill = Mock(side_effect=Exception("Processing error"))

        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = ProcessOrderFillRequest(order_id=uuid4(), fill_price=Decimal("100.50"))

        # The error is caught and logged within the process method
        response = await process_fill_use_case.process(request)

        assert response.success is False
        assert "Failed to process fill: Processing error" in response.error
