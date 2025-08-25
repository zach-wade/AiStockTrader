"""
Comprehensive Unit Tests for Order Execution Use Cases

Tests all order execution use cases with complete coverage including:
- ProcessOrderFillUseCase
- SimulateOrderExecutionUseCase
- CalculateCommissionUseCase

Achieves 80%+ coverage with focus on business logic validation.
"""

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

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
from src.domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from src.domain.entities.portfolio import Portfolio
from src.domain.services.commission_calculator import ICommissionCalculator
from src.domain.services.market_microstructure import IMarketMicrostructure
from src.domain.services.order_processor import OrderProcessor
from src.domain.value_objects.money import Money
from src.domain.value_objects.price import Price
from src.domain.value_objects.quantity import Quantity


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
    uow.positions = AsyncMock()

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
    calculator.calculate = Mock(return_value=Money(Decimal("1.00")))
    calculator.schedule = Mock(rate=Decimal("0.001"))
    return calculator


@pytest.fixture
def mock_market_microstructure():
    """Create a mock market microstructure service."""
    market = Mock(spec=IMarketMicrostructure)
    market.calculate_execution_price = Mock(return_value=Price(Decimal("100.00")))
    market.calculate_market_impact = Mock(return_value=Decimal("0.01"))
    return market


@pytest.fixture
def sample_order():
    """Create a sample order for testing."""
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Quantity(Decimal("100")),
        limit_price=Price(Decimal("150.00")),
    )
    order.id = uuid4()
    order.portfolio_id = uuid4()
    order.status = OrderStatus.SUBMITTED
    return order


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


# Test ProcessOrderFillUseCase
class TestProcessOrderFillUseCase:
    """Test ProcessOrderFillUseCase class."""

    @pytest.mark.asyncio
    async def test_init(self, mock_unit_of_work, mock_order_processor, mock_commission_calculator):
        """Test use case initialization."""
        use_case = ProcessOrderFillUseCase(
            mock_unit_of_work, mock_order_processor, mock_commission_calculator
        )

        assert use_case.unit_of_work == mock_unit_of_work
        assert use_case.order_processor == mock_order_processor
        assert use_case.commission_calculator == mock_commission_calculator
        assert use_case.name == "ProcessOrderFillUseCase"

    @pytest.mark.asyncio
    async def test_validate_success(
        self, mock_unit_of_work, mock_order_processor, mock_commission_calculator
    ):
        """Test successful validation."""
        use_case = ProcessOrderFillUseCase(
            mock_unit_of_work, mock_order_processor, mock_commission_calculator
        )

        request = ProcessOrderFillRequest(
            order_id=uuid4(),
            fill_price=Decimal("100.00"),
            fill_quantity=50,
        )

        result = await use_case.validate(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_negative_price(
        self, mock_unit_of_work, mock_order_processor, mock_commission_calculator
    ):
        """Test validation with negative fill price."""
        use_case = ProcessOrderFillUseCase(
            mock_unit_of_work, mock_order_processor, mock_commission_calculator
        )

        request = ProcessOrderFillRequest(
            order_id=uuid4(),
            fill_price=Decimal("-100.00"),
        )

        result = await use_case.validate(request)
        assert result == "Fill price must be positive"

    @pytest.mark.asyncio
    async def test_validate_zero_price(
        self, mock_unit_of_work, mock_order_processor, mock_commission_calculator
    ):
        """Test validation with zero fill price."""
        use_case = ProcessOrderFillUseCase(
            mock_unit_of_work, mock_order_processor, mock_commission_calculator
        )

        request = ProcessOrderFillRequest(
            order_id=uuid4(),
            fill_price=Decimal("0"),
        )

        result = await use_case.validate(request)
        assert result == "Fill price must be positive"

    @pytest.mark.asyncio
    async def test_validate_negative_quantity(
        self, mock_unit_of_work, mock_order_processor, mock_commission_calculator
    ):
        """Test validation with negative fill quantity."""
        use_case = ProcessOrderFillUseCase(
            mock_unit_of_work, mock_order_processor, mock_commission_calculator
        )

        request = ProcessOrderFillRequest(
            order_id=uuid4(),
            fill_price=Decimal("100.00"),
            fill_quantity=-50,
        )

        result = await use_case.validate(request)
        assert result == "Fill quantity must be positive"

    @pytest.mark.asyncio
    async def test_validate_zero_quantity(
        self, mock_unit_of_work, mock_order_processor, mock_commission_calculator
    ):
        """Test validation with zero fill quantity."""
        use_case = ProcessOrderFillUseCase(
            mock_unit_of_work, mock_order_processor, mock_commission_calculator
        )

        request = ProcessOrderFillRequest(
            order_id=uuid4(),
            fill_price=Decimal("100.00"),
            fill_quantity=0,
        )

        result = await use_case.validate(request)
        assert result == "Fill quantity must be positive"

    @pytest.mark.asyncio
    async def test_process_successful_fill(
        self,
        mock_unit_of_work,
        mock_order_processor,
        mock_commission_calculator,
        sample_order,
        sample_portfolio,
    ):
        """Test successful order fill processing."""
        use_case = ProcessOrderFillUseCase(
            mock_unit_of_work, mock_order_processor, mock_commission_calculator
        )

        # Setup mocks
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = ProcessOrderFillRequest(
            order_id=sample_order.id,
            fill_price=Decimal("100.00"),
            fill_quantity=50,
            timestamp=datetime.now(UTC),
        )

        response = await use_case.process(request)

        assert response.success is True
        assert response.filled is True
        assert response.fill_price == Decimal("100.00")
        assert response.fill_quantity == 50
        assert response.commission == Decimal("1.00")

        # Verify interactions
        mock_unit_of_work.orders.get_order_by_id.assert_called_once_with(sample_order.id)
        mock_unit_of_work.portfolios.get_portfolio_by_id.assert_called_once_with(
            sample_order.portfolio_id
        )
        mock_unit_of_work.orders.update_order.assert_called_once()
        mock_unit_of_work.portfolios.update_portfolio.assert_called_once()
        mock_order_processor.process_fill.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_partial_fill(
        self,
        mock_unit_of_work,
        mock_order_processor,
        mock_commission_calculator,
        sample_order,
        sample_portfolio,
    ):
        """Test partial order fill."""
        use_case = ProcessOrderFillUseCase(
            mock_unit_of_work, mock_order_processor, mock_commission_calculator
        )

        # Setup mocks
        sample_order.filled_quantity = Decimal("30")
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = ProcessOrderFillRequest(
            order_id=sample_order.id,
            fill_price=Decimal("100.00"),
            fill_quantity=20,  # Partial fill
        )

        response = await use_case.process(request)

        assert response.success is True
        assert response.filled is True
        assert response.fill_quantity == 20

    @pytest.mark.asyncio
    async def test_process_fill_entire_order(
        self,
        mock_unit_of_work,
        mock_order_processor,
        mock_commission_calculator,
        sample_order,
        sample_portfolio,
    ):
        """Test filling entire order when fill_quantity is None."""
        use_case = ProcessOrderFillUseCase(
            mock_unit_of_work, mock_order_processor, mock_commission_calculator
        )

        # Setup mocks
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = ProcessOrderFillRequest(
            order_id=sample_order.id,
            fill_price=Decimal("100.00"),
            fill_quantity=None,  # Fill entire order
        )

        response = await use_case.process(request)

        assert response.success is True
        assert response.filled is True
        assert response.fill_quantity == 100  # Full order quantity

    @pytest.mark.asyncio
    async def test_process_order_not_found(
        self, mock_unit_of_work, mock_order_processor, mock_commission_calculator
    ):
        """Test processing when order not found."""
        use_case = ProcessOrderFillUseCase(
            mock_unit_of_work, mock_order_processor, mock_commission_calculator
        )

        # Setup mocks
        mock_unit_of_work.orders.get_order_by_id.return_value = None

        request = ProcessOrderFillRequest(
            order_id=uuid4(),
            fill_price=Decimal("100.00"),
        )

        response = await use_case.process(request)

        assert response.success is False
        assert response.error == "Order not found"
        assert response.filled is False

    @pytest.mark.asyncio
    async def test_process_inactive_order(
        self, mock_unit_of_work, mock_order_processor, mock_commission_calculator, sample_order
    ):
        """Test processing inactive order."""
        use_case = ProcessOrderFillUseCase(
            mock_unit_of_work, mock_order_processor, mock_commission_calculator
        )

        # Setup mocks
        sample_order.status = OrderStatus.CANCELLED
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order

        request = ProcessOrderFillRequest(
            order_id=sample_order.id,
            fill_price=Decimal("100.00"),
        )

        response = await use_case.process(request)

        assert response.success is False
        assert "Cannot fill order in status: cancelled" in response.error
        assert response.filled is False

    @pytest.mark.asyncio
    async def test_process_portfolio_not_found(
        self, mock_unit_of_work, mock_order_processor, mock_commission_calculator, sample_order
    ):
        """Test processing when portfolio not found."""
        use_case = ProcessOrderFillUseCase(
            mock_unit_of_work, mock_order_processor, mock_commission_calculator
        )

        # Setup mocks
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = None

        request = ProcessOrderFillRequest(
            order_id=sample_order.id,
            fill_price=Decimal("100.00"),
        )

        response = await use_case.process(request)

        assert response.success is False
        assert response.error == "Portfolio not found"
        assert response.filled is False

    @pytest.mark.asyncio
    async def test_process_no_quantity_to_fill(
        self,
        mock_unit_of_work,
        mock_order_processor,
        mock_commission_calculator,
        sample_order,
        sample_portfolio,
    ):
        """Test processing when order is already fully filled."""
        use_case = ProcessOrderFillUseCase(
            mock_unit_of_work, mock_order_processor, mock_commission_calculator
        )

        # Setup mocks - order is already fully filled
        sample_order.filled_quantity = sample_order.quantity
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = ProcessOrderFillRequest(
            order_id=sample_order.id,
            fill_price=Decimal("100.00"),
        )

        response = await use_case.process(request)

        assert response.success is False
        assert response.error == "No quantity to fill"
        assert response.filled is False

    @pytest.mark.asyncio
    async def test_process_fill_exception(
        self,
        mock_unit_of_work,
        mock_order_processor,
        mock_commission_calculator,
        sample_order,
        sample_portfolio,
    ):
        """Test exception handling during fill processing."""
        use_case = ProcessOrderFillUseCase(
            mock_unit_of_work, mock_order_processor, mock_commission_calculator
        )

        # Setup mocks
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_order_processor.process_fill.side_effect = Exception("Processing error")

        request = ProcessOrderFillRequest(
            order_id=sample_order.id,
            fill_price=Decimal("100.00"),
        )

        response = await use_case.process(request)

        assert response.success is False
        assert "Failed to process fill: Processing error" in response.error
        assert response.filled is False

    @pytest.mark.asyncio
    async def test_process_with_metadata(
        self,
        mock_unit_of_work,
        mock_order_processor,
        mock_commission_calculator,
        sample_order,
        sample_portfolio,
    ):
        """Test processing with metadata and correlation ID."""
        use_case = ProcessOrderFillUseCase(
            mock_unit_of_work, mock_order_processor, mock_commission_calculator
        )

        # Setup mocks
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = ProcessOrderFillRequest(
            order_id=sample_order.id,
            fill_price=Decimal("100.00"),
            correlation_id=uuid4(),
            metadata={"source": "test", "priority": "high"},
        )

        response = await use_case.process(request)

        assert response.success is True
        assert response.request_id == request.request_id


# Test SimulateOrderExecutionUseCase
class TestSimulateOrderExecutionUseCase:
    """Test SimulateOrderExecutionUseCase class."""

    @pytest.mark.asyncio
    async def test_init(
        self, mock_unit_of_work, mock_market_microstructure, mock_commission_calculator
    ):
        """Test use case initialization."""
        use_case = SimulateOrderExecutionUseCase(
            mock_unit_of_work, mock_market_microstructure, mock_commission_calculator
        )

        assert use_case.unit_of_work == mock_unit_of_work
        assert use_case.market_microstructure == mock_market_microstructure
        assert use_case.commission_calculator == mock_commission_calculator
        assert use_case.name == "SimulateOrderExecutionUseCase"

    @pytest.mark.asyncio
    async def test_validate_success(
        self, mock_unit_of_work, mock_market_microstructure, mock_commission_calculator
    ):
        """Test successful validation."""
        use_case = SimulateOrderExecutionUseCase(
            mock_unit_of_work, mock_market_microstructure, mock_commission_calculator
        )

        request = SimulateOrderExecutionRequest(
            order_id=uuid4(),
            market_price=Decimal("100.00"),
            available_liquidity=Decimal("10000.00"),
        )

        result = await use_case.validate(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_negative_price(
        self, mock_unit_of_work, mock_market_microstructure, mock_commission_calculator
    ):
        """Test validation with negative market price."""
        use_case = SimulateOrderExecutionUseCase(
            mock_unit_of_work, mock_market_microstructure, mock_commission_calculator
        )

        request = SimulateOrderExecutionRequest(
            order_id=uuid4(),
            market_price=Decimal("-100.00"),
        )

        result = await use_case.validate(request)
        assert result == "Market price must be positive"

    @pytest.mark.asyncio
    async def test_validate_zero_price(
        self, mock_unit_of_work, mock_market_microstructure, mock_commission_calculator
    ):
        """Test validation with zero market price."""
        use_case = SimulateOrderExecutionUseCase(
            mock_unit_of_work, mock_market_microstructure, mock_commission_calculator
        )

        request = SimulateOrderExecutionRequest(
            order_id=uuid4(),
            market_price=Decimal("0"),
        )

        result = await use_case.validate(request)
        assert result == "Market price must be positive"

    @pytest.mark.asyncio
    async def test_validate_negative_liquidity(
        self, mock_unit_of_work, mock_market_microstructure, mock_commission_calculator
    ):
        """Test validation with negative liquidity."""
        use_case = SimulateOrderExecutionUseCase(
            mock_unit_of_work, mock_market_microstructure, mock_commission_calculator
        )

        request = SimulateOrderExecutionRequest(
            order_id=uuid4(),
            market_price=Decimal("100.00"),
            available_liquidity=Decimal("-1000.00"),
        )

        result = await use_case.validate(request)
        assert result == "Available liquidity must be positive"

    @pytest.mark.asyncio
    async def test_process_successful_simulation(
        self,
        mock_unit_of_work,
        mock_market_microstructure,
        mock_commission_calculator,
        sample_order,
    ):
        """Test successful order execution simulation."""
        use_case = SimulateOrderExecutionUseCase(
            mock_unit_of_work, mock_market_microstructure, mock_commission_calculator
        )

        # Setup mocks
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_market_microstructure.calculate_execution_price.return_value = Price(Decimal("100.50"))
        mock_market_microstructure.calculate_market_impact.return_value = Decimal("0.02")
        mock_commission_calculator.calculate.return_value = Money(Decimal("2.50"))

        request = SimulateOrderExecutionRequest(
            order_id=sample_order.id,
            market_price=Decimal("100.00"),
            available_liquidity=Decimal("50000.00"),
        )

        response = await use_case.process(request)

        assert response.success is True
        assert response.execution_price == Decimal("100.50")
        assert response.slippage == Decimal("0")  # Limit order has no slippage
        assert response.market_impact == Decimal("0.02")
        assert response.estimated_commission == Decimal("2.50")

    @pytest.mark.asyncio
    async def test_process_market_order_buy_slippage(
        self,
        mock_unit_of_work,
        mock_market_microstructure,
        mock_commission_calculator,
    ):
        """Test market order buy with slippage calculation."""
        use_case = SimulateOrderExecutionUseCase(
            mock_unit_of_work, mock_market_microstructure, mock_commission_calculator
        )

        # Create market buy order
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("100"),
        )
        order.id = uuid4()

        # Setup mocks
        mock_unit_of_work.orders.get_order_by_id.return_value = order
        mock_market_microstructure.calculate_execution_price.return_value = Price(
            Decimal("101.00")  # Higher than market price for buy
        )

        request = SimulateOrderExecutionRequest(
            order_id=order.id,
            market_price=Decimal("100.00"),
        )

        response = await use_case.process(request)

        assert response.success is True
        assert response.execution_price == Decimal("101.00")
        assert response.slippage == Decimal("1.00")  # Buy slippage

    @pytest.mark.asyncio
    async def test_process_market_order_sell_slippage(
        self,
        mock_unit_of_work,
        mock_market_microstructure,
        mock_commission_calculator,
    ):
        """Test market order sell with slippage calculation."""
        use_case = SimulateOrderExecutionUseCase(
            mock_unit_of_work, mock_market_microstructure, mock_commission_calculator
        )

        # Create market sell order
        order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("100"),
        )
        order.id = uuid4()

        # Setup mocks
        mock_unit_of_work.orders.get_order_by_id.return_value = order
        mock_market_microstructure.calculate_execution_price.return_value = Price(
            Decimal("99.00")  # Lower than market price for sell
        )

        request = SimulateOrderExecutionRequest(
            order_id=order.id,
            market_price=Decimal("100.00"),
        )

        response = await use_case.process(request)

        assert response.success is True
        assert response.execution_price == Decimal("99.00")
        assert response.slippage == Decimal("1.00")  # Sell slippage

    @pytest.mark.asyncio
    async def test_process_order_not_found(
        self, mock_unit_of_work, mock_market_microstructure, mock_commission_calculator
    ):
        """Test simulation when order not found."""
        use_case = SimulateOrderExecutionUseCase(
            mock_unit_of_work, mock_market_microstructure, mock_commission_calculator
        )

        # Setup mocks
        mock_unit_of_work.orders.get_order_by_id.return_value = None

        request = SimulateOrderExecutionRequest(
            order_id=uuid4(),
            market_price=Decimal("100.00"),
        )

        response = await use_case.process(request)

        assert response.success is False
        assert response.error == "Order not found"

    @pytest.mark.asyncio
    async def test_process_with_metadata(
        self,
        mock_unit_of_work,
        mock_market_microstructure,
        mock_commission_calculator,
        sample_order,
    ):
        """Test simulation with metadata and correlation ID."""
        use_case = SimulateOrderExecutionUseCase(
            mock_unit_of_work, mock_market_microstructure, mock_commission_calculator
        )

        # Setup mocks
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order

        request = SimulateOrderExecutionRequest(
            order_id=sample_order.id,
            market_price=Decimal("100.00"),
            correlation_id=uuid4(),
            metadata={"simulation": "test"},
        )

        response = await use_case.process(request)

        assert response.success is True
        assert response.request_id == request.request_id


# Test CalculateCommissionUseCase
class TestCalculateCommissionUseCase:
    """Test CalculateCommissionUseCase class."""

    def test_init(self, mock_commission_calculator):
        """Test use case initialization."""
        use_case = CalculateCommissionUseCase(mock_commission_calculator)

        assert use_case.commission_calculator == mock_commission_calculator
        assert use_case.logger is not None

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_commission_calculator):
        """Test successful commission calculation."""
        use_case = CalculateCommissionUseCase(mock_commission_calculator)

        # Setup mock
        mock_commission_calculator.calculate.return_value = Money(Decimal("5.00"))
        mock_commission_calculator.schedule = Mock(rate=Decimal("0.002"))

        request = CalculateCommissionRequest(
            quantity=100,
            price=Decimal("150.00"),
            order_type="limit",
            symbol="AAPL",
        )

        response = await use_case.execute(request)

        assert response.success is True
        assert response.commission == Decimal("5.00")
        assert response.commission_rate == Decimal("0.002")

        # Verify calculation was called
        mock_commission_calculator.calculate.assert_called_once()
        call_args = mock_commission_calculator.calculate.call_args
        assert isinstance(call_args[1]["quantity"], Quantity)
        assert isinstance(call_args[1]["price"], Money)

    @pytest.mark.asyncio
    async def test_execute_without_schedule(self, mock_commission_calculator):
        """Test commission calculation without schedule attribute."""
        use_case = CalculateCommissionUseCase(mock_commission_calculator)

        # Setup mock without schedule
        mock_commission_calculator.calculate.return_value = Money(Decimal("3.00"))
        delattr(mock_commission_calculator, "schedule")

        request = CalculateCommissionRequest(
            quantity=50,
            price=Decimal("100.00"),
            order_type="market",
        )

        response = await use_case.execute(request)

        assert response.success is True
        assert response.commission == Decimal("3.00")
        assert response.commission_rate is None

    @pytest.mark.asyncio
    async def test_execute_calculation_error(self, mock_commission_calculator):
        """Test commission calculation with error."""
        use_case = CalculateCommissionUseCase(mock_commission_calculator)

        # Setup mock to raise exception
        mock_commission_calculator.calculate.side_effect = ValueError("Calculation error")

        request = CalculateCommissionRequest(
            quantity=100,
            price=Decimal("150.00"),
            order_type="limit",
        )

        response = await use_case.execute(request)

        assert response.success is False
        assert "Failed to calculate commission: Calculation error" in response.error

    @pytest.mark.asyncio
    async def test_execute_with_metadata(self, mock_commission_calculator):
        """Test commission calculation with metadata."""
        use_case = CalculateCommissionUseCase(mock_commission_calculator)

        # Setup mock
        mock_commission_calculator.calculate.return_value = Money(Decimal("2.00"))

        request = CalculateCommissionRequest(
            quantity=25,
            price=Decimal("80.00"),
            order_type="stop",
            correlation_id=uuid4(),
            metadata={"broker": "test_broker"},
        )

        response = await use_case.execute(request)

        assert response.success is True
        assert response.request_id == request.request_id

    @pytest.mark.asyncio
    async def test_execute_logging_on_error(self, mock_commission_calculator):
        """Test that errors are properly logged."""
        use_case = CalculateCommissionUseCase(mock_commission_calculator)

        # Setup mock to raise exception
        mock_commission_calculator.calculate.side_effect = RuntimeError("Test error")

        request = CalculateCommissionRequest(
            quantity=100,
            price=Decimal("150.00"),
            order_type="limit",
        )

        with patch.object(use_case.logger, "error") as mock_logger:
            response = await use_case.execute(request)

            assert response.success is False
            mock_logger.assert_called_once()
            assert "Failed to calculate commission" in str(mock_logger.call_args)


# Test Request/Response DTOs
class TestRequestResponseDTOs:
    """Test request and response data classes."""

    def test_process_order_fill_request_init(self):
        """Test ProcessOrderFillRequest initialization."""
        request = ProcessOrderFillRequest(
            order_id=uuid4(),
            fill_price=Decimal("100.00"),
        )

        assert request.request_id is not None
        assert request.metadata == {}
        assert request.fill_quantity is None
        assert request.timestamp is None

    def test_process_order_fill_request_with_values(self):
        """Test ProcessOrderFillRequest with all values."""
        order_id = uuid4()
        request_id = uuid4()
        correlation_id = uuid4()
        timestamp = datetime.now(UTC)

        request = ProcessOrderFillRequest(
            order_id=order_id,
            fill_price=Decimal("150.00"),
            fill_quantity=75,
            timestamp=timestamp,
            request_id=request_id,
            correlation_id=correlation_id,
            metadata={"test": "data"},
        )

        assert request.order_id == order_id
        assert request.fill_price == Decimal("150.00")
        assert request.fill_quantity == 75
        assert request.timestamp == timestamp
        assert request.request_id == request_id
        assert request.correlation_id == correlation_id
        assert request.metadata == {"test": "data"}

    def test_simulate_order_execution_request_init(self):
        """Test SimulateOrderExecutionRequest initialization."""
        request = SimulateOrderExecutionRequest(
            order_id=uuid4(),
            market_price=Decimal("100.00"),
        )

        assert request.request_id is not None
        assert request.metadata == {}
        assert request.available_liquidity is None

    def test_calculate_commission_request_init(self):
        """Test CalculateCommissionRequest initialization."""
        request = CalculateCommissionRequest(
            quantity=100,
            price=Decimal("50.00"),
            order_type="limit",
        )

        assert request.request_id is not None
        assert request.metadata == {}
        assert request.symbol is None

    def test_process_order_fill_response(self):
        """Test ProcessOrderFillResponse initialization."""
        response = ProcessOrderFillResponse(
            success=True,
            filled=True,
            fill_price=Decimal("100.00"),
            fill_quantity=50,
            commission=Decimal("1.00"),
            position_id=uuid4(),
        )

        assert response.success is True
        assert response.filled is True
        assert response.fill_price == Decimal("100.00")
        assert response.fill_quantity == 50
        assert response.commission == Decimal("1.00")
        assert response.position_id is not None

    def test_simulate_order_execution_response(self):
        """Test SimulateOrderExecutionResponse initialization."""
        response = SimulateOrderExecutionResponse(
            success=True,
            execution_price=Decimal("100.50"),
            slippage=Decimal("0.50"),
            market_impact=Decimal("0.01"),
            estimated_commission=Decimal("2.00"),
        )

        assert response.success is True
        assert response.execution_price == Decimal("100.50")
        assert response.slippage == Decimal("0.50")
        assert response.market_impact == Decimal("0.01")
        assert response.estimated_commission == Decimal("2.00")

    def test_calculate_commission_response(self):
        """Test CalculateCommissionResponse initialization."""
        response = CalculateCommissionResponse(
            success=True,
            commission=Decimal("5.00"),
            commission_rate=Decimal("0.001"),
        )

        assert response.success is True
        assert response.commission == Decimal("5.00")
        assert response.commission_rate == Decimal("0.001")
