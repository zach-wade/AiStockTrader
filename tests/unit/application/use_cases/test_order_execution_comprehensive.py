"""
Comprehensive tests for order execution use cases with full coverage.

Tests all order execution-related use cases including:
- ProcessOrderFillUseCase
- SimulateOrderExecutionUseCase
- CalculateCommissionUseCase

Covers all scenarios including success, failure, validation, and edge cases.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from src.application.interfaces.repositories import (
    IOrderRepository,
    IPortfolioRepository,
    IPositionRepository,
)
from src.application.interfaces.unit_of_work import IUnitOfWork
from src.application.use_cases.order_execution import (
    CalculateCommissionRequest,
    CalculateCommissionUseCase,
    ProcessOrderFillRequest,
    ProcessOrderFillResponse,
    ProcessOrderFillUseCase,
    SimulateOrderExecutionRequest,
    SimulateOrderExecutionResponse,
    SimulateOrderExecutionUseCase,
)
from src.domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from src.domain.entities.portfolio import Portfolio
from src.domain.entities.position import Position
from src.domain.services.commission_calculator import ICommissionCalculator
from src.domain.services.market_microstructure import IMarketMicrostructure
from src.domain.services.order_processor import FillDetails, OrderProcessor
from src.domain.value_objects.money import Money
from src.domain.value_objects.price import Price
from src.domain.value_objects.quantity import Quantity


@pytest.fixture
def mock_unit_of_work():
    """Create a mock unit of work."""
    uow = Mock(spec=IUnitOfWork)

    # Setup repository mocks
    uow.orders = AsyncMock(spec=IOrderRepository)
    uow.portfolios = AsyncMock(spec=IPortfolioRepository)
    uow.positions = AsyncMock(spec=IPositionRepository)

    # Setup transaction methods
    uow.begin = AsyncMock()
    uow.commit = AsyncMock()
    uow.rollback = AsyncMock()

    # Make it work as async context manager
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock(return_value=None)

    return uow


@pytest.fixture
def mock_order_processor():
    """Create a mock order processor."""
    processor = Mock(spec=OrderProcessor)
    processor.process_fill = AsyncMock()
    return processor


@pytest.fixture
def mock_commission_calculator():
    """Create a mock commission calculator."""
    calculator = Mock(spec=ICommissionCalculator)
    calculator.calculate = Mock(return_value=Money(Decimal("10.00")))

    # Add schedule attribute for rate access
    calculator.schedule = Mock()
    calculator.schedule.rate = Decimal("0.001")

    return calculator


@pytest.fixture
def mock_market_microstructure():
    """Create a mock market microstructure."""
    microstructure = Mock(spec=IMarketMicrostructure)
    microstructure.calculate_execution_price = Mock(return_value=Price(Decimal("150.50")))
    microstructure.calculate_market_impact = Mock(return_value=Decimal("0.02"))
    return microstructure


@pytest.fixture
def sample_order():
    """Create a sample order for testing."""
    order = Order(
        symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=Decimal("100")
    )
    order.portfolio_id = uuid4()
    order.status = OrderStatus.PENDING
    order.filled_quantity = Decimal("0")
    return order


@pytest.fixture
def sample_portfolio():
    """Create a sample portfolio for testing."""
    portfolio = Portfolio(
        id=uuid4(),
        name="Test Portfolio",
        initial_capital=Decimal("100000"),
        cash_balance=Decimal("50000"),
    )

    # Add a position
    position = Position(
        symbol="AAPL", quantity=Decimal("50"), average_entry_price=Decimal("145.00")
    )
    portfolio.positions = {position.id: position}

    return portfolio


class TestProcessOrderFillUseCase:
    """Test ProcessOrderFillUseCase."""

    @pytest.mark.asyncio
    async def test_process_complete_fill_success(
        self,
        mock_unit_of_work,
        mock_order_processor,
        mock_commission_calculator,
        sample_order,
        sample_portfolio,
    ):
        """Test processing a complete order fill."""
        # Setup
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        use_case = ProcessOrderFillUseCase(
            unit_of_work=mock_unit_of_work,
            order_processor=mock_order_processor,
            commission_calculator=mock_commission_calculator,
        )

        request = ProcessOrderFillRequest(
            order_id=sample_order.id,
            fill_price=Decimal("150.00"),
            fill_quantity=None,  # Complete fill
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.filled is True
        assert response.fill_price == Decimal("150.00")
        assert response.fill_quantity == 100  # Full order quantity
        assert response.commission == Decimal("10.00")

        # Verify interactions
        mock_commission_calculator.calculate.assert_called_once()
        mock_order_processor.process_fill.assert_called_once()
        mock_unit_of_work.orders.update_order.assert_called_once()
        mock_unit_of_work.portfolios.update_portfolio.assert_called_once()
        mock_unit_of_work.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_partial_fill_success(
        self,
        mock_unit_of_work,
        mock_order_processor,
        mock_commission_calculator,
        sample_order,
        sample_portfolio,
    ):
        """Test processing a partial order fill."""
        # Setup
        sample_order.quantity = Decimal("200")
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        use_case = ProcessOrderFillUseCase(
            unit_of_work=mock_unit_of_work,
            order_processor=mock_order_processor,
            commission_calculator=mock_commission_calculator,
        )

        request = ProcessOrderFillRequest(
            order_id=sample_order.id,
            fill_price=Decimal("150.00"),
            fill_quantity=50,  # Partial fill
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.filled is True
        assert response.fill_quantity == 50

        # Verify order was updated with partial fill
        call_args = mock_order_processor.process_fill.call_args[0][0]
        assert isinstance(call_args, FillDetails)
        assert call_args.fill_quantity == Quantity(Decimal("50"))

    @pytest.mark.asyncio
    async def test_process_fill_with_timestamp(
        self,
        mock_unit_of_work,
        mock_order_processor,
        mock_commission_calculator,
        sample_order,
        sample_portfolio,
    ):
        """Test processing fill with specific timestamp."""
        # Setup
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        use_case = ProcessOrderFillUseCase(
            unit_of_work=mock_unit_of_work,
            order_processor=mock_order_processor,
            commission_calculator=mock_commission_calculator,
        )

        fill_time = datetime.now(UTC) - timedelta(minutes=5)
        request = ProcessOrderFillRequest(
            order_id=sample_order.id, fill_price=Decimal("150.00"), timestamp=fill_time
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True

        # Verify timestamp was used
        call_args = mock_order_processor.process_fill.call_args[0][0]
        assert call_args.timestamp == fill_time

    @pytest.mark.asyncio
    async def test_process_fill_order_not_found(
        self, mock_unit_of_work, mock_order_processor, mock_commission_calculator
    ):
        """Test processing fill when order doesn't exist."""
        # Setup
        mock_unit_of_work.orders.get_order_by_id.return_value = None

        use_case = ProcessOrderFillUseCase(
            unit_of_work=mock_unit_of_work,
            order_processor=mock_order_processor,
            commission_calculator=mock_commission_calculator,
        )

        request = ProcessOrderFillRequest(order_id=uuid4(), fill_price=Decimal("150.00"))

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Order not found" in response.error
        mock_order_processor.process_fill.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_fill_inactive_order(
        self,
        mock_unit_of_work,
        mock_order_processor,
        mock_commission_calculator,
        sample_order,
        sample_portfolio,
    ):
        """Test processing fill for inactive order."""
        # Setup
        sample_order.status = OrderStatus.CANCELLED
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        use_case = ProcessOrderFillUseCase(
            unit_of_work=mock_unit_of_work,
            order_processor=mock_order_processor,
            commission_calculator=mock_commission_calculator,
        )

        request = ProcessOrderFillRequest(order_id=sample_order.id, fill_price=Decimal("150.00"))

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Cannot fill order in status: cancelled" in response.error
        mock_order_processor.process_fill.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_fill_portfolio_not_found(
        self, mock_unit_of_work, mock_order_processor, mock_commission_calculator, sample_order
    ):
        """Test processing fill when portfolio doesn't exist."""
        # Setup
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = None

        use_case = ProcessOrderFillUseCase(
            unit_of_work=mock_unit_of_work,
            order_processor=mock_order_processor,
            commission_calculator=mock_commission_calculator,
        )

        request = ProcessOrderFillRequest(order_id=sample_order.id, fill_price=Decimal("150.00"))

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Portfolio not found" in response.error
        mock_order_processor.process_fill.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_fill_no_remaining_quantity(
        self,
        mock_unit_of_work,
        mock_order_processor,
        mock_commission_calculator,
        sample_order,
        sample_portfolio,
    ):
        """Test processing fill when order is already filled."""
        # Setup
        sample_order.filled_quantity = sample_order.quantity
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        use_case = ProcessOrderFillUseCase(
            unit_of_work=mock_unit_of_work,
            order_processor=mock_order_processor,
            commission_calculator=mock_commission_calculator,
        )

        request = ProcessOrderFillRequest(order_id=sample_order.id, fill_price=Decimal("150.00"))

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "No quantity to fill" in response.error

    @pytest.mark.asyncio
    async def test_validate_negative_fill_price(
        self, mock_unit_of_work, mock_order_processor, mock_commission_calculator
    ):
        """Test validation with negative fill price."""
        use_case = ProcessOrderFillUseCase(
            unit_of_work=mock_unit_of_work,
            order_processor=mock_order_processor,
            commission_calculator=mock_commission_calculator,
        )

        request = ProcessOrderFillRequest(order_id=uuid4(), fill_price=Decimal("-10.00"))

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Fill price must be positive" in response.error

    @pytest.mark.asyncio
    async def test_validate_negative_fill_quantity(
        self, mock_unit_of_work, mock_order_processor, mock_commission_calculator
    ):
        """Test validation with negative fill quantity."""
        use_case = ProcessOrderFillUseCase(
            unit_of_work=mock_unit_of_work,
            order_processor=mock_order_processor,
            commission_calculator=mock_commission_calculator,
        )

        request = ProcessOrderFillRequest(
            order_id=uuid4(), fill_price=Decimal("150.00"), fill_quantity=-50
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Fill quantity must be positive" in response.error

    @pytest.mark.asyncio
    async def test_process_fill_exception_handling(
        self,
        mock_unit_of_work,
        mock_order_processor,
        mock_commission_calculator,
        sample_order,
        sample_portfolio,
    ):
        """Test exception handling during fill processing."""
        # Setup
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_order_processor.process_fill.side_effect = Exception("Processing error")

        use_case = ProcessOrderFillUseCase(
            unit_of_work=mock_unit_of_work,
            order_processor=mock_order_processor,
            commission_calculator=mock_commission_calculator,
        )

        request = ProcessOrderFillRequest(order_id=sample_order.id, fill_price=Decimal("150.00"))

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Failed to process fill" in response.error
        assert "Processing error" in response.error
        mock_unit_of_work.rollback.assert_called_once()


class TestSimulateOrderExecutionUseCase:
    """Test SimulateOrderExecutionUseCase."""

    @pytest.mark.asyncio
    async def test_simulate_market_order_execution(
        self,
        mock_unit_of_work,
        mock_market_microstructure,
        mock_commission_calculator,
        sample_order,
    ):
        """Test simulating market order execution."""
        # Setup
        sample_order.order_type = OrderType.MARKET
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order

        use_case = SimulateOrderExecutionUseCase(
            unit_of_work=mock_unit_of_work,
            market_microstructure=mock_market_microstructure,
            commission_calculator=mock_commission_calculator,
        )

        request = SimulateOrderExecutionRequest(
            order_id=sample_order.id, market_price=Decimal("150.00")
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.execution_price == Decimal("150.50")
        assert response.slippage == Decimal("0.50")  # Buy order, so positive slippage
        assert response.market_impact == Decimal("0.02")
        assert response.estimated_commission == Decimal("10.00")

        # Verify calculations were called
        mock_market_microstructure.calculate_execution_price.assert_called_once()
        mock_market_microstructure.calculate_market_impact.assert_called_once()
        mock_commission_calculator.calculate.assert_called_once()

    @pytest.mark.asyncio
    async def test_simulate_limit_order_execution(
        self,
        mock_unit_of_work,
        mock_market_microstructure,
        mock_commission_calculator,
        sample_order,
    ):
        """Test simulating limit order execution."""
        # Setup
        sample_order.order_type = OrderType.LIMIT
        sample_order.limit_price = Decimal("149.00")
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order

        use_case = SimulateOrderExecutionUseCase(
            unit_of_work=mock_unit_of_work,
            market_microstructure=mock_market_microstructure,
            commission_calculator=mock_commission_calculator,
        )

        request = SimulateOrderExecutionRequest(
            order_id=sample_order.id, market_price=Decimal("150.00")
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.slippage == Decimal("0")  # No slippage for limit orders

    @pytest.mark.asyncio
    async def test_simulate_sell_order_slippage(
        self,
        mock_unit_of_work,
        mock_market_microstructure,
        mock_commission_calculator,
        sample_order,
    ):
        """Test slippage calculation for sell orders."""
        # Setup
        sample_order.side = OrderSide.SELL
        sample_order.order_type = OrderType.MARKET
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_market_microstructure.calculate_execution_price.return_value = Price(Decimal("149.50"))

        use_case = SimulateOrderExecutionUseCase(
            unit_of_work=mock_unit_of_work,
            market_microstructure=mock_market_microstructure,
            commission_calculator=mock_commission_calculator,
        )

        request = SimulateOrderExecutionRequest(
            order_id=sample_order.id, market_price=Decimal("150.00")
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.execution_price == Decimal("149.50")
        assert response.slippage == Decimal("0.50")  # Sell order, positive slippage (worse price)

    @pytest.mark.asyncio
    async def test_simulate_with_liquidity_constraint(
        self,
        mock_unit_of_work,
        mock_market_microstructure,
        mock_commission_calculator,
        sample_order,
    ):
        """Test simulation with liquidity constraints."""
        # Setup
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order

        use_case = SimulateOrderExecutionUseCase(
            unit_of_work=mock_unit_of_work,
            market_microstructure=mock_market_microstructure,
            commission_calculator=mock_commission_calculator,
        )

        request = SimulateOrderExecutionRequest(
            order_id=sample_order.id,
            market_price=Decimal("150.00"),
            available_liquidity=Decimal("50000"),
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        # Verify liquidity was considered (implementation dependent)

    @pytest.mark.asyncio
    async def test_simulate_order_not_found(
        self, mock_unit_of_work, mock_market_microstructure, mock_commission_calculator
    ):
        """Test simulation when order doesn't exist."""
        # Setup
        mock_unit_of_work.orders.get_order_by_id.return_value = None

        use_case = SimulateOrderExecutionUseCase(
            unit_of_work=mock_unit_of_work,
            market_microstructure=mock_market_microstructure,
            commission_calculator=mock_commission_calculator,
        )

        request = SimulateOrderExecutionRequest(order_id=uuid4(), market_price=Decimal("150.00"))

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Order not found" in response.error

    @pytest.mark.asyncio
    async def test_validate_negative_market_price(
        self, mock_unit_of_work, mock_market_microstructure, mock_commission_calculator
    ):
        """Test validation with negative market price."""
        use_case = SimulateOrderExecutionUseCase(
            unit_of_work=mock_unit_of_work,
            market_microstructure=mock_market_microstructure,
            commission_calculator=mock_commission_calculator,
        )

        request = SimulateOrderExecutionRequest(order_id=uuid4(), market_price=Decimal("-150.00"))

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Market price must be positive" in response.error

    @pytest.mark.asyncio
    async def test_validate_negative_liquidity(
        self, mock_unit_of_work, mock_market_microstructure, mock_commission_calculator
    ):
        """Test validation with negative liquidity."""
        use_case = SimulateOrderExecutionUseCase(
            unit_of_work=mock_unit_of_work,
            market_microstructure=mock_market_microstructure,
            commission_calculator=mock_commission_calculator,
        )

        request = SimulateOrderExecutionRequest(
            order_id=uuid4(), market_price=Decimal("150.00"), available_liquidity=Decimal("-1000")
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Available liquidity must be positive" in response.error


class TestCalculateCommissionUseCase:
    """Test CalculateCommissionUseCase."""

    @pytest.mark.asyncio
    async def test_calculate_commission_success(self, mock_commission_calculator):
        """Test successful commission calculation."""
        use_case = CalculateCommissionUseCase(commission_calculator=mock_commission_calculator)

        request = CalculateCommissionRequest(
            quantity=100, price=Decimal("150.00"), order_type="market"
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.commission == Decimal("10.00")
        assert response.commission_rate == Decimal("0.001")

        # Verify calculation was called
        mock_commission_calculator.calculate.assert_called_once_with(
            quantity=Quantity(100), price=Money(Decimal("150.00"))
        )

    @pytest.mark.asyncio
    async def test_calculate_commission_with_symbol(self, mock_commission_calculator):
        """Test commission calculation with symbol."""
        use_case = CalculateCommissionUseCase(commission_calculator=mock_commission_calculator)

        request = CalculateCommissionRequest(
            quantity=100, price=Decimal("150.00"), order_type="limit", symbol="AAPL"
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.commission == Decimal("10.00")

    @pytest.mark.asyncio
    async def test_calculate_commission_no_schedule(self, mock_commission_calculator):
        """Test commission calculation without schedule access."""
        # Remove schedule attribute
        delattr(mock_commission_calculator, "schedule")

        use_case = CalculateCommissionUseCase(commission_calculator=mock_commission_calculator)

        request = CalculateCommissionRequest(
            quantity=100, price=Decimal("150.00"), order_type="market"
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.commission == Decimal("10.00")
        assert response.commission_rate is None

    @pytest.mark.asyncio
    async def test_calculate_commission_exception(self, mock_commission_calculator):
        """Test commission calculation with exception."""
        mock_commission_calculator.calculate.side_effect = Exception("Calculation error")

        use_case = CalculateCommissionUseCase(commission_calculator=mock_commission_calculator)

        request = CalculateCommissionRequest(
            quantity=100, price=Decimal("150.00"), order_type="market"
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Failed to calculate commission" in response.error
        assert "Calculation error" in response.error


class TestOrderExecutionRequestResponseDTOs:
    """Test order execution request and response DTOs."""

    def test_process_fill_request_defaults(self):
        """Test ProcessOrderFillRequest default values."""
        order_id = uuid4()
        request = ProcessOrderFillRequest(order_id=order_id, fill_price=Decimal("150.00"))

        assert request.order_id == order_id
        assert request.fill_price == Decimal("150.00")
        assert request.fill_quantity is None
        assert request.timestamp is None
        assert request.request_id is not None
        assert request.metadata == {}

    def test_process_fill_request_with_values(self):
        """Test ProcessOrderFillRequest with all values."""
        order_id = uuid4()
        request_id = uuid4()
        fill_time = datetime.now(UTC)

        request = ProcessOrderFillRequest(
            order_id=order_id,
            fill_price=Decimal("150.00"),
            fill_quantity=50,
            timestamp=fill_time,
            request_id=request_id,
            metadata={"source": "broker"},
        )

        assert request.fill_quantity == 50
        assert request.timestamp == fill_time
        assert request.request_id == request_id
        assert request.metadata == {"source": "broker"}

    def test_simulate_execution_request_defaults(self):
        """Test SimulateOrderExecutionRequest defaults."""
        order_id = uuid4()
        request = SimulateOrderExecutionRequest(order_id=order_id, market_price=Decimal("150.00"))

        assert request.order_id == order_id
        assert request.market_price == Decimal("150.00")
        assert request.available_liquidity is None
        assert request.request_id is not None

    def test_calculate_commission_request(self):
        """Test CalculateCommissionRequest."""
        request = CalculateCommissionRequest(
            quantity=100, price=Decimal("150.00"), order_type="market", symbol="AAPL"
        )

        assert request.quantity == 100
        assert request.price == Decimal("150.00")
        assert request.order_type == "market"
        assert request.symbol == "AAPL"

    def test_process_fill_response(self):
        """Test ProcessOrderFillResponse."""
        response = ProcessOrderFillResponse(
            success=True,
            filled=True,
            fill_price=Decimal("150.00"),
            fill_quantity=100,
            commission=Decimal("10.00"),
            position_id=uuid4(),
            request_id=uuid4(),
        )

        assert response.filled is True
        assert response.fill_price == Decimal("150.00")
        assert response.fill_quantity == 100
        assert response.commission == Decimal("10.00")
        assert response.position_id is not None

    def test_simulate_execution_response(self):
        """Test SimulateOrderExecutionResponse."""
        response = SimulateOrderExecutionResponse(
            success=True,
            execution_price=Decimal("150.50"),
            slippage=Decimal("0.50"),
            market_impact=Decimal("0.02"),
            estimated_commission=Decimal("10.00"),
            request_id=uuid4(),
        )

        assert response.execution_price == Decimal("150.50")
        assert response.slippage == Decimal("0.50")
        assert response.market_impact == Decimal("0.02")
        assert response.estimated_commission == Decimal("10.00")
