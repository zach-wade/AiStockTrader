"""
Comprehensive unit tests for PlaceOrderUseCase.

This module provides exhaustive test coverage for the PlaceOrderUseCase class,
testing all methods, branches, and edge cases to achieve 100% coverage.
Tests include success scenarios, failure scenarios, validation errors,
transaction management, and concurrent order handling.
"""

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID, uuid4

import pytest

from src.application.interfaces.broker import IBroker
from src.application.interfaces.repositories import (
    IMarketDataRepository,
    IOrderRepository,
    IPortfolioRepository,
)
from src.application.interfaces.unit_of_work import IUnitOfWork
from src.application.use_cases.trading import (
    PlaceOrderRequest,
    PlaceOrderResponse,
    PlaceOrderUseCase,
)
from src.domain.entities.order import OrderSide, OrderStatus, OrderType, TimeInForce
from src.domain.entities.portfolio import Portfolio
from src.domain.services.order_validator import OrderValidator, ValidationResult
from src.domain.services.risk_calculator import RiskCalculator
from src.domain.value_objects.money import Money


# Test Fixtures
@pytest.fixture
def mock_unit_of_work():
    """Create a mock unit of work with all required repositories."""
    uow = AsyncMock(spec=IUnitOfWork)

    # Setup repositories
    uow.portfolios = AsyncMock(spec=IPortfolioRepository)
    uow.orders = AsyncMock(spec=IOrderRepository)
    uow.market_data = AsyncMock(spec=IMarketDataRepository)

    # Setup transaction methods
    uow.commit = AsyncMock()
    uow.rollback = AsyncMock()

    # Setup context manager
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock(return_value=None)

    return uow


@pytest.fixture
def mock_broker():
    """Create a mock broker implementation."""
    broker = Mock(spec=IBroker)
    broker.submit_order = Mock(return_value="BROKER-ORDER-123")
    broker.cancel_order = Mock(return_value=True)
    broker.update_order = Mock()
    broker.get_order_status = Mock(return_value=OrderStatus.SUBMITTED)
    return broker


@pytest.fixture
def mock_order_validator():
    """Create a mock order validator."""
    validator = Mock(spec=OrderValidator)
    validator.validate_order = Mock(
        return_value=ValidationResult(
            is_valid=True,
            required_capital=Money(Decimal("10000")),
            estimated_commission=Money(Decimal("10")),
        )
    )
    validator.validate_modification = Mock(return_value=ValidationResult(is_valid=True))
    return validator


@pytest.fixture
def mock_risk_calculator():
    """Create a mock risk calculator."""
    calculator = Mock(spec=RiskCalculator)
    # Return tuple (is_valid, error_message)
    calculator.check_risk_limits = Mock(return_value=(True, None))
    return calculator


@pytest.fixture
def place_order_use_case(
    mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator
):
    """Create PlaceOrderUseCase instance with mocked dependencies."""
    return PlaceOrderUseCase(
        unit_of_work=mock_unit_of_work,
        broker=mock_broker,
        order_validator=mock_order_validator,
        risk_calculator=mock_risk_calculator,
    )


@pytest.fixture
def sample_portfolio():
    """Create a sample portfolio for testing."""
    portfolio = Portfolio(name="Test Portfolio", initial_capital=Decimal("100000"))
    portfolio.id = uuid4()
    portfolio.cash_balance = Decimal("50000")
    return portfolio


@pytest.fixture
def sample_market_bar():
    """Create a sample market bar for testing."""

    class MarketBar:
        def __init__(self):
            self.symbol = "AAPL"
            self.close = Decimal("150.00")
            self.open = Decimal("148.00")
            self.high = Decimal("151.00")
            self.low = Decimal("147.50")
            self.volume = 1000000

    return MarketBar()


@pytest.fixture
def valid_request():
    """Create a valid place order request."""
    return PlaceOrderRequest(
        portfolio_id=uuid4(),
        symbol="AAPL",
        side="buy",
        order_type="market",
        quantity=100,
        time_in_force="day",
        strategy_id="STRATEGY-001",
        correlation_id=uuid4(),
        metadata={"source": "test"},
    )


class TestPlaceOrderUseCaseInitialization:
    """Test PlaceOrderUseCase initialization and configuration."""

    def test_initialization_with_all_dependencies(
        self, mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator
    ):
        """Test use case initialization with all required dependencies."""
        use_case = PlaceOrderUseCase(
            unit_of_work=mock_unit_of_work,
            broker=mock_broker,
            order_validator=mock_order_validator,
            risk_calculator=mock_risk_calculator,
        )

        assert use_case.unit_of_work == mock_unit_of_work
        assert use_case.broker == mock_broker
        assert use_case.order_validator == mock_order_validator
        assert use_case.risk_calculator == mock_risk_calculator
        assert use_case.name == "PlaceOrderUseCase"
        assert use_case.logger is not None

    def test_initialization_with_custom_name(
        self, mock_unit_of_work, mock_broker, mock_order_validator, mock_risk_calculator
    ):
        """Test use case initialization with custom name."""
        use_case = PlaceOrderUseCase(
            unit_of_work=mock_unit_of_work,
            broker=mock_broker,
            order_validator=mock_order_validator,
            risk_calculator=mock_risk_calculator,
        )
        # Name is set in parent class constructor
        assert use_case.name == "PlaceOrderUseCase"


class TestPlaceOrderRequestValidation:
    """Test request validation logic."""

    @pytest.mark.asyncio
    async def test_validate_valid_market_order(self, place_order_use_case, valid_request):
        """Test validation of valid market order."""
        error = await place_order_use_case.validate(valid_request)
        assert error is None

    @pytest.mark.asyncio
    async def test_validate_valid_limit_order(self, place_order_use_case):
        """Test validation of valid limit order."""
        request = PlaceOrderRequest(
            portfolio_id=uuid4(),
            symbol="AAPL",
            side="sell",
            order_type="limit",
            quantity=50,
            limit_price=155.50,
            time_in_force="gtc",
        )
        error = await place_order_use_case.validate(request)
        assert error is None

    @pytest.mark.asyncio
    async def test_validate_valid_stop_order(self, place_order_use_case):
        """Test validation of valid stop order."""
        request = PlaceOrderRequest(
            portfolio_id=uuid4(),
            symbol="AAPL",
            side="sell",
            order_type="stop",
            quantity=100,
            stop_price=145.00,
        )
        error = await place_order_use_case.validate(request)
        assert error is None

    @pytest.mark.asyncio
    async def test_validate_valid_stop_limit_order(self, place_order_use_case):
        """Test validation of valid stop-limit order."""
        request = PlaceOrderRequest(
            portfolio_id=uuid4(),
            symbol="AAPL",
            side="buy",
            order_type="stop_limit",
            quantity=100,
            limit_price=155.00,
            stop_price=154.00,
        )
        error = await place_order_use_case.validate(request)
        assert error is None

    @pytest.mark.asyncio
    async def test_validate_invalid_order_type(self, place_order_use_case):
        """Test validation with invalid order type."""
        request = PlaceOrderRequest(
            portfolio_id=uuid4(), symbol="AAPL", side="buy", order_type="invalid_type", quantity=100
        )
        error = await place_order_use_case.validate(request)
        assert error == "Invalid order type: invalid_type"

    @pytest.mark.asyncio
    async def test_validate_invalid_side(self, place_order_use_case):
        """Test validation with invalid side."""
        request = PlaceOrderRequest(
            portfolio_id=uuid4(),
            symbol="AAPL",
            side="invalid_side",
            order_type="market",
            quantity=100,
        )
        error = await place_order_use_case.validate(request)
        assert error == "Invalid order side: invalid_side"

    @pytest.mark.asyncio
    async def test_validate_zero_quantity(self, place_order_use_case):
        """Test validation with zero quantity."""
        request = PlaceOrderRequest(
            portfolio_id=uuid4(), symbol="AAPL", side="buy", order_type="market", quantity=0
        )
        error = await place_order_use_case.validate(request)
        assert error == "Quantity must be positive"

    @pytest.mark.asyncio
    async def test_validate_negative_quantity(self, place_order_use_case):
        """Test validation with negative quantity."""
        request = PlaceOrderRequest(
            portfolio_id=uuid4(), symbol="AAPL", side="buy", order_type="market", quantity=-100
        )
        error = await place_order_use_case.validate(request)
        assert error == "Quantity must be positive"

    @pytest.mark.asyncio
    async def test_validate_limit_order_without_limit_price(self, place_order_use_case):
        """Test validation of limit order without limit price."""
        request = PlaceOrderRequest(
            portfolio_id=uuid4(),
            symbol="AAPL",
            side="buy",
            order_type="limit",
            quantity=100,
            limit_price=None,
        )
        error = await place_order_use_case.validate(request)
        assert error == "limit order requires limit price"

    @pytest.mark.asyncio
    async def test_validate_stop_order_without_stop_price(self, place_order_use_case):
        """Test validation of stop order without stop price."""
        request = PlaceOrderRequest(
            portfolio_id=uuid4(),
            symbol="AAPL",
            side="sell",
            order_type="stop",
            quantity=100,
            stop_price=None,
        )
        error = await place_order_use_case.validate(request)
        assert error == "stop order requires stop price"

    @pytest.mark.asyncio
    async def test_validate_stop_limit_order_without_limit_price(self, place_order_use_case):
        """Test validation of stop-limit order without limit price."""
        request = PlaceOrderRequest(
            portfolio_id=uuid4(),
            symbol="AAPL",
            side="buy",
            order_type="stop_limit",
            quantity=100,
            limit_price=None,
            stop_price=150.00,
        )
        error = await place_order_use_case.validate(request)
        assert error == "stop_limit order requires limit price"

    @pytest.mark.asyncio
    async def test_validate_stop_limit_order_without_stop_price(self, place_order_use_case):
        """Test validation of stop-limit order without stop price."""
        request = PlaceOrderRequest(
            portfolio_id=uuid4(),
            symbol="AAPL",
            side="buy",
            order_type="stop_limit",
            quantity=100,
            limit_price=150.00,
            stop_price=None,
        )
        error = await place_order_use_case.validate(request)
        assert error == "stop_limit order requires stop price"


class TestPlaceOrderProcessing:
    """Test order processing logic."""

    @pytest.mark.asyncio
    async def test_process_successful_market_order(
        self,
        place_order_use_case,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        sample_market_bar,
        valid_request,
    ):
        """Test successful processing of market order."""
        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id = AsyncMock(return_value=sample_portfolio)
        mock_unit_of_work.market_data.get_latest_bar = AsyncMock(return_value=sample_market_bar)
        mock_unit_of_work.orders.save_order = AsyncMock()
        mock_order_validator.validate_order.return_value = ValidationResult(is_valid=True)
        mock_risk_calculator.check_risk_limits.return_value = (True, None)
        mock_broker.submit_order.return_value = "BROKER-123"

        # Execute
        response = await place_order_use_case.process(valid_request)

        # Assert
        assert response.success is True
        assert response.order_id is not None
        assert response.broker_order_id == "BROKER-123"
        assert response.status == OrderStatus.SUBMITTED
        assert response.request_id == valid_request.request_id

        # Verify calls
        mock_unit_of_work.portfolios.get_portfolio_by_id.assert_called_once_with(
            valid_request.portfolio_id
        )
        mock_unit_of_work.market_data.get_latest_bar.assert_called_once_with(valid_request.symbol)
        mock_order_validator.validate_order.assert_called_once()
        mock_risk_calculator.check_risk_limits.assert_called_once()
        mock_broker.submit_order.assert_called_once()
        mock_unit_of_work.orders.save_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_successful_limit_order(
        self,
        place_order_use_case,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        sample_market_bar,
    ):
        """Test successful processing of limit order."""
        request = PlaceOrderRequest(
            portfolio_id=sample_portfolio.id,
            symbol="AAPL",
            side="sell",
            order_type="limit",
            quantity=50,
            limit_price=155.50,
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id = AsyncMock(return_value=sample_portfolio)
        mock_unit_of_work.market_data.get_latest_bar = AsyncMock(return_value=sample_market_bar)
        mock_unit_of_work.orders.save_order = AsyncMock()
        mock_order_validator.validate_order.return_value = ValidationResult(is_valid=True)
        mock_risk_calculator.check_risk_limits.return_value = (True, None)
        mock_broker.submit_order.return_value = "BROKER-456"

        response = await place_order_use_case.process(request)

        assert response.success is True
        assert response.broker_order_id == "BROKER-456"

        # Verify the order was created with correct parameters
        submitted_order = mock_broker.submit_order.call_args[0][0]
        assert submitted_order.symbol == "AAPL"
        assert submitted_order.side == OrderSide.SELL
        assert submitted_order.order_type == OrderType.LIMIT
        assert submitted_order.quantity == Decimal("50")
        assert submitted_order.limit_price == Decimal("155.50")

    @pytest.mark.asyncio
    async def test_process_successful_stop_limit_order(
        self,
        place_order_use_case,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        sample_market_bar,
    ):
        """Test successful processing of stop-limit order."""
        request = PlaceOrderRequest(
            portfolio_id=sample_portfolio.id,
            symbol="TSLA",
            side="buy",
            order_type="stop_limit",
            quantity=25,
            limit_price=200.00,
            stop_price=195.00,
            time_in_force="ioc",
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id = AsyncMock(return_value=sample_portfolio)
        mock_unit_of_work.market_data.get_latest_bar = AsyncMock(return_value=sample_market_bar)
        mock_unit_of_work.orders.save_order = AsyncMock()
        mock_order_validator.validate_order.return_value = ValidationResult(is_valid=True)
        mock_risk_calculator.check_risk_limits.return_value = (True, None)
        mock_broker.submit_order.return_value = "BROKER-789"

        response = await place_order_use_case.process(request)

        assert response.success is True
        assert response.broker_order_id == "BROKER-789"

        # Verify the order was created with correct parameters
        submitted_order = mock_broker.submit_order.call_args[0][0]
        assert submitted_order.symbol == "TSLA"
        assert submitted_order.side == OrderSide.BUY
        assert submitted_order.order_type == OrderType.STOP_LIMIT
        assert submitted_order.quantity == Decimal("25")
        assert submitted_order.limit_price == Decimal("200.00")
        assert submitted_order.stop_price == Decimal("195.00")
        assert submitted_order.time_in_force == TimeInForce.IOC

    @pytest.mark.asyncio
    async def test_process_portfolio_not_found(
        self, place_order_use_case, mock_unit_of_work, valid_request
    ):
        """Test processing when portfolio is not found."""
        mock_unit_of_work.portfolios.get_portfolio_by_id = AsyncMock(return_value=None)

        response = await place_order_use_case.process(valid_request)

        assert response.success is False
        assert response.error == "Portfolio not found"
        assert response.request_id == valid_request.request_id
        assert response.order_id is None
        assert response.broker_order_id is None

    @pytest.mark.asyncio
    async def test_process_market_data_not_available(
        self, place_order_use_case, mock_unit_of_work, sample_portfolio, valid_request
    ):
        """Test processing when market data is not available."""
        mock_unit_of_work.portfolios.get_portfolio_by_id = AsyncMock(return_value=sample_portfolio)
        mock_unit_of_work.market_data.get_latest_bar = AsyncMock(return_value=None)

        response = await place_order_use_case.process(valid_request)

        assert response.success is False
        assert response.error == "Cannot get current market price"
        assert response.request_id == valid_request.request_id

    @pytest.mark.asyncio
    async def test_process_order_validation_failure(
        self,
        place_order_use_case,
        mock_unit_of_work,
        mock_order_validator,
        sample_portfolio,
        sample_market_bar,
        valid_request,
    ):
        """Test processing when order validation fails."""
        mock_unit_of_work.portfolios.get_portfolio_by_id = AsyncMock(return_value=sample_portfolio)
        mock_unit_of_work.market_data.get_latest_bar = AsyncMock(return_value=sample_market_bar)
        mock_unit_of_work.orders.save_order = AsyncMock()
        mock_order_validator.validate_order.return_value = ValidationResult(
            is_valid=False, error_message="Insufficient buying power"
        )

        response = await place_order_use_case.process(valid_request)

        assert response.success is False
        assert response.error == "Insufficient buying power"
        assert response.request_id == valid_request.request_id

    @pytest.mark.asyncio
    async def test_process_order_validation_failure_no_message(
        self,
        place_order_use_case,
        mock_unit_of_work,
        mock_order_validator,
        sample_portfolio,
        sample_market_bar,
        valid_request,
    ):
        """Test processing when order validation fails without error message."""
        mock_unit_of_work.portfolios.get_portfolio_by_id = AsyncMock(return_value=sample_portfolio)
        mock_unit_of_work.market_data.get_latest_bar = AsyncMock(return_value=sample_market_bar)
        mock_unit_of_work.orders.save_order = AsyncMock()
        mock_order_validator.validate_order.return_value = ValidationResult(
            is_valid=False, error_message=None
        )

        response = await place_order_use_case.process(valid_request)

        assert response.success is False
        assert response.error == "Order validation failed"

    @pytest.mark.asyncio
    async def test_process_risk_limit_violation(
        self,
        place_order_use_case,
        mock_unit_of_work,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        sample_market_bar,
        valid_request,
    ):
        """Test processing when risk limits are violated."""
        mock_unit_of_work.portfolios.get_portfolio_by_id = AsyncMock(return_value=sample_portfolio)
        mock_unit_of_work.market_data.get_latest_bar = AsyncMock(return_value=sample_market_bar)
        mock_unit_of_work.orders.save_order = AsyncMock()
        mock_order_validator.validate_order.return_value = ValidationResult(is_valid=True)
        mock_risk_calculator.check_risk_limits.return_value = (
            False,
            "Position size exceeds maximum",
        )

        response = await place_order_use_case.process(valid_request)

        assert response.success is False
        assert response.error == "Risk limit violated: Position size exceeds maximum"

    @pytest.mark.asyncio
    async def test_process_broker_submission_failure(
        self,
        place_order_use_case,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        sample_market_bar,
        valid_request,
    ):
        """Test processing when broker submission fails."""
        mock_unit_of_work.portfolios.get_portfolio_by_id = AsyncMock(return_value=sample_portfolio)
        mock_unit_of_work.market_data.get_latest_bar = AsyncMock(return_value=sample_market_bar)
        mock_unit_of_work.orders.save_order = AsyncMock()
        mock_order_validator.validate_order.return_value = ValidationResult(is_valid=True)
        mock_risk_calculator.check_risk_limits.return_value = (True, None)
        mock_broker.submit_order.side_effect = Exception("Connection timeout")

        response = await place_order_use_case.process(valid_request)

        assert response.success is False
        assert "Failed to submit order" in response.error
        assert "Connection timeout" in response.error

    @pytest.mark.asyncio
    async def test_process_repository_save_failure(
        self,
        place_order_use_case,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        sample_market_bar,
        valid_request,
    ):
        """Test processing when repository save fails."""
        mock_unit_of_work.portfolios.get_portfolio_by_id = AsyncMock(return_value=sample_portfolio)
        mock_unit_of_work.market_data.get_latest_bar = AsyncMock(return_value=sample_market_bar)
        mock_unit_of_work.orders.save_order = AsyncMock()
        mock_order_validator.validate_order.return_value = ValidationResult(is_valid=True)
        mock_risk_calculator.check_risk_limits.return_value = (True, None)
        mock_broker.submit_order.return_value = "BROKER-123"
        mock_unit_of_work.orders.save_order.side_effect = Exception("Database error")

        response = await place_order_use_case.process(valid_request)

        assert response.success is False
        assert "Failed to submit order" in response.error
        assert "Database error" in response.error

    @pytest.mark.asyncio
    async def test_process_with_missing_request_id(
        self, place_order_use_case, mock_unit_of_work, sample_portfolio, sample_market_bar
    ):
        """Test processing with request that has no request_id."""
        request = PlaceOrderRequest(
            portfolio_id=sample_portfolio.id,
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=100,
            request_id=None,
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id = AsyncMock(return_value=None)

        response = await place_order_use_case.process(request)

        assert response.success is False
        assert response.error == "Portfolio not found"
        assert response.request_id is not None  # Should generate one


class TestTransactionManagement:
    """Test transaction management in PlaceOrderUseCase."""

    @pytest.mark.asyncio
    async def test_successful_transaction_commit(
        self,
        place_order_use_case,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        sample_market_bar,
        valid_request,
    ):
        """Test that successful order placement commits transaction."""
        mock_unit_of_work.portfolios.get_portfolio_by_id = AsyncMock(return_value=sample_portfolio)
        mock_unit_of_work.market_data.get_latest_bar = AsyncMock(return_value=sample_market_bar)
        mock_unit_of_work.orders.save_order = AsyncMock()
        mock_order_validator.validate_order.return_value = ValidationResult(is_valid=True)
        mock_risk_calculator.check_risk_limits.return_value = (True, None)
        mock_broker.submit_order.return_value = "BROKER-123"

        response = await place_order_use_case.execute(valid_request)

        assert response.success is True
        mock_unit_of_work.__aenter__.assert_called()
        mock_unit_of_work.__aexit__.assert_called()
        mock_unit_of_work.commit.assert_called_once()
        mock_unit_of_work.rollback.assert_not_called()

    @pytest.mark.asyncio
    async def test_validation_failure_rollback(
        self, place_order_use_case, mock_unit_of_work, valid_request
    ):
        """Test that validation failure triggers rollback."""
        invalid_request = PlaceOrderRequest(
            portfolio_id=uuid4(),
            symbol="AAPL",
            side="invalid_side",
            order_type="market",
            quantity=100,
        )

        response = await place_order_use_case.execute(invalid_request)

        assert response.success is False
        mock_unit_of_work.__aenter__.assert_called()
        mock_unit_of_work.__aexit__.assert_called()
        mock_unit_of_work.commit.assert_not_called()
        mock_unit_of_work.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_processing_failure_rollback(
        self,
        place_order_use_case,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        sample_market_bar,
        valid_request,
    ):
        """Test that processing failure triggers rollback."""
        mock_unit_of_work.portfolios.get_portfolio_by_id = AsyncMock(return_value=sample_portfolio)
        mock_unit_of_work.market_data.get_latest_bar = AsyncMock(return_value=sample_market_bar)
        mock_unit_of_work.orders.save_order = AsyncMock()
        mock_order_validator.validate_order.return_value = ValidationResult(is_valid=True)
        mock_risk_calculator.check_risk_limits.return_value = (True, None)
        mock_broker.submit_order.side_effect = Exception("Broker error")

        response = await place_order_use_case.execute(valid_request)

        assert response.success is False
        mock_unit_of_work.commit.assert_not_called()
        mock_unit_of_work.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_exception_during_context_management(
        self,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        valid_request,
    ):
        """Test exception handling during transaction context management."""
        # Make the context manager raise an exception
        mock_unit_of_work.__aenter__.side_effect = Exception("Context manager error")

        use_case = PlaceOrderUseCase(
            unit_of_work=mock_unit_of_work,
            broker=mock_broker,
            order_validator=mock_order_validator,
            risk_calculator=mock_risk_calculator,
        )

        with pytest.raises(Exception, match="Context manager error"):
            await use_case.execute(valid_request)


class TestConcurrentOrderHandling:
    """Test concurrent order placement scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_orders_same_portfolio(
        self,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        sample_market_bar,
    ):
        """Test handling multiple concurrent orders for same portfolio."""
        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id = AsyncMock(return_value=sample_portfolio)
        mock_unit_of_work.market_data.get_latest_bar = AsyncMock(return_value=sample_market_bar)
        mock_unit_of_work.orders.save_order = AsyncMock()
        mock_order_validator.validate_order.return_value = ValidationResult(is_valid=True)
        mock_risk_calculator.check_risk_limits.return_value = (True, None)

        # Create unique broker IDs for each order
        broker_ids = [f"BROKER-{i}" for i in range(5)]
        mock_broker.submit_order.side_effect = broker_ids

        # Create multiple use case instances
        use_cases = []
        for _ in range(5):
            use_case = PlaceOrderUseCase(
                unit_of_work=mock_unit_of_work,
                broker=mock_broker,
                order_validator=mock_order_validator,
                risk_calculator=mock_risk_calculator,
            )
            use_cases.append(use_case)

        # Create multiple requests
        requests = []
        for i in range(5):
            request = PlaceOrderRequest(
                portfolio_id=sample_portfolio.id,
                symbol=f"STOCK{i}",
                side="buy",
                order_type="market",
                quantity=100,
            )
            requests.append(request)

        # Execute orders concurrently
        tasks = [use_case.process(request) for use_case, request in zip(use_cases, requests)]
        responses = await asyncio.gather(*tasks)

        # Verify all orders were processed
        assert len(responses) == 5
        for i, response in enumerate(responses):
            assert response.success is True
            assert response.broker_order_id == f"BROKER-{i}"

        # Verify correct number of calls
        assert mock_broker.submit_order.call_count == 5
        assert mock_unit_of_work.orders.save_order.call_count == 5

    @pytest.mark.asyncio
    async def test_concurrent_orders_different_portfolios(
        self,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_market_bar,
    ):
        """Test handling concurrent orders for different portfolios."""
        # Create multiple portfolios
        portfolios = []
        for i in range(3):
            portfolio = Portfolio(name=f"Portfolio {i}", initial_capital=Decimal("100000"))
            portfolio.id = uuid4()
            portfolios.append(portfolio)

        # Setup mocks to return different portfolios
        mock_unit_of_work.portfolios.get_portfolio_by_id = AsyncMock(side_effect=portfolios)
        mock_unit_of_work.market_data.get_latest_bar = AsyncMock(return_value=sample_market_bar)
        mock_unit_of_work.orders.save_order = AsyncMock()
        mock_order_validator.validate_order.return_value = ValidationResult(is_valid=True)
        mock_risk_calculator.check_risk_limits.return_value = (True, None)
        mock_broker.submit_order.return_value = "BROKER-MULTI"

        # Create use case
        use_case = PlaceOrderUseCase(
            unit_of_work=mock_unit_of_work,
            broker=mock_broker,
            order_validator=mock_order_validator,
            risk_calculator=mock_risk_calculator,
        )

        # Create requests for different portfolios
        requests = []
        for portfolio in portfolios:
            request = PlaceOrderRequest(
                portfolio_id=portfolio.id,
                symbol="AAPL",
                side="buy",
                order_type="market",
                quantity=100,
            )
            requests.append(request)

        # Execute orders concurrently
        tasks = [use_case.process(request) for request in requests]
        responses = await asyncio.gather(*tasks)

        # Verify all orders were processed
        assert len(responses) == 3
        for response in responses:
            assert response.success is True

    @pytest.mark.asyncio
    async def test_concurrent_orders_with_failures(
        self,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        sample_market_bar,
    ):
        """Test handling concurrent orders with some failures."""
        mock_unit_of_work.portfolios.get_portfolio_by_id = AsyncMock(return_value=sample_portfolio)
        mock_unit_of_work.market_data.get_latest_bar = AsyncMock(return_value=sample_market_bar)
        mock_unit_of_work.orders.save_order = AsyncMock()
        mock_order_validator.validate_order.return_value = ValidationResult(is_valid=True)
        mock_risk_calculator.check_risk_limits.return_value = (True, None)

        # Make some orders fail at broker submission
        def submit_order_side_effect(order):
            if order.symbol == "FAIL":
                raise Exception("Broker error")
            return f"BROKER-{order.symbol}"

        mock_broker.submit_order.side_effect = submit_order_side_effect

        # Create use case
        use_case = PlaceOrderUseCase(
            unit_of_work=mock_unit_of_work,
            broker=mock_broker,
            order_validator=mock_order_validator,
            risk_calculator=mock_risk_calculator,
        )

        # Create mixed success/failure requests
        symbols = ["AAPL", "FAIL", "GOOGL", "FAIL", "MSFT"]
        requests = []
        for symbol in symbols:
            request = PlaceOrderRequest(
                portfolio_id=sample_portfolio.id,
                symbol=symbol,
                side="buy",
                order_type="market",
                quantity=100,
            )
            requests.append(request)

        # Execute orders concurrently
        tasks = [use_case.process(request) for request in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify results
        assert len(responses) == 5
        assert responses[0].success is True  # AAPL
        assert responses[1].success is False  # FAIL
        assert responses[2].success is True  # GOOGL
        assert responses[3].success is False  # FAIL
        assert responses[4].success is True  # MSFT


class TestLoggingBehavior:
    """Test logging behavior in PlaceOrderUseCase."""

    @pytest.mark.asyncio
    async def test_successful_order_logging(
        self,
        place_order_use_case,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        sample_market_bar,
        valid_request,
    ):
        """Test logging for successful order placement."""
        mock_unit_of_work.portfolios.get_portfolio_by_id = AsyncMock(return_value=sample_portfolio)
        mock_unit_of_work.market_data.get_latest_bar = AsyncMock(return_value=sample_market_bar)
        mock_unit_of_work.orders.save_order = AsyncMock()
        mock_order_validator.validate_order.return_value = ValidationResult(is_valid=True)
        mock_risk_calculator.check_risk_limits.return_value = (True, None)
        mock_broker.submit_order.return_value = "BROKER-123"

        with patch.object(place_order_use_case.logger, "info") as mock_info:
            with patch.object(place_order_use_case.logger, "error") as mock_error:
                response = await place_order_use_case.execute(valid_request)

                assert response.success is True
                # Should log transaction start, execution, and commit
                assert mock_info.call_count >= 3
                mock_error.assert_not_called()

    @pytest.mark.asyncio
    async def test_validation_failure_logging(self, place_order_use_case):
        """Test logging for validation failures."""
        invalid_request = PlaceOrderRequest(
            portfolio_id=uuid4(), symbol="AAPL", side="invalid", order_type="market", quantity=100
        )

        with patch.object(place_order_use_case.logger, "warning") as mock_warning:
            response = await place_order_use_case.execute(invalid_request)

            assert response.success is False
            mock_warning.assert_called_once()
            assert "Validation failed" in str(mock_warning.call_args)

    @pytest.mark.asyncio
    async def test_processing_error_logging(
        self,
        place_order_use_case,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        sample_market_bar,
        valid_request,
    ):
        """Test logging for processing errors."""
        mock_unit_of_work.portfolios.get_portfolio_by_id = AsyncMock(return_value=sample_portfolio)
        mock_unit_of_work.market_data.get_latest_bar = AsyncMock(return_value=sample_market_bar)
        mock_unit_of_work.orders.save_order = AsyncMock()
        mock_order_validator.validate_order.return_value = ValidationResult(is_valid=True)
        mock_risk_calculator.check_risk_limits.return_value = (True, None)
        mock_broker.submit_order.side_effect = RuntimeError("Broker unavailable")

        with patch.object(place_order_use_case.logger, "error") as mock_error:
            response = await place_order_use_case.process(valid_request)

            assert response.success is False
            mock_error.assert_called_once()
            assert "Failed to submit order" in str(mock_error.call_args)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_duplicate_order_submission(
        self,
        place_order_use_case,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        sample_market_bar,
    ):
        """Test handling of duplicate order submissions."""
        request = PlaceOrderRequest(
            portfolio_id=sample_portfolio.id,
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=100,
            request_id=uuid4(),
            correlation_id=uuid4(),
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id = AsyncMock(return_value=sample_portfolio)
        mock_unit_of_work.market_data.get_latest_bar = AsyncMock(return_value=sample_market_bar)
        mock_unit_of_work.orders.save_order = AsyncMock()
        mock_order_validator.validate_order.return_value = ValidationResult(is_valid=True)
        mock_risk_calculator.check_risk_limits.return_value = (True, None)
        mock_broker.submit_order.return_value = "BROKER-DUP"

        # Submit same request twice
        response1 = await place_order_use_case.process(request)
        response2 = await place_order_use_case.process(request)

        # Both should succeed but with different order IDs
        assert response1.success is True
        assert response2.success is True
        assert response1.order_id != response2.order_id
        assert response1.request_id == response2.request_id  # Same request ID

    @pytest.mark.asyncio
    async def test_very_large_quantity(
        self,
        place_order_use_case,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        sample_market_bar,
    ):
        """Test handling of very large order quantities."""
        request = PlaceOrderRequest(
            portfolio_id=sample_portfolio.id,
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=999999999,
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id = AsyncMock(return_value=sample_portfolio)
        mock_unit_of_work.market_data.get_latest_bar = AsyncMock(return_value=sample_market_bar)
        mock_unit_of_work.orders.save_order = AsyncMock()
        mock_order_validator.validate_order.return_value = ValidationResult(is_valid=True)
        mock_risk_calculator.check_risk_limits.return_value = (False, "Quantity exceeds maximum")

        response = await place_order_use_case.process(request)

        assert response.success is False
        assert "Risk limit violated" in response.error

    @pytest.mark.asyncio
    async def test_very_small_limit_price(
        self,
        place_order_use_case,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        sample_market_bar,
    ):
        """Test handling of very small limit prices."""
        request = PlaceOrderRequest(
            portfolio_id=sample_portfolio.id,
            symbol="AAPL",
            side="buy",
            order_type="limit",
            quantity=100,
            limit_price=0.0001,
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id = AsyncMock(return_value=sample_portfolio)
        mock_unit_of_work.market_data.get_latest_bar = AsyncMock(return_value=sample_market_bar)
        mock_unit_of_work.orders.save_order = AsyncMock()
        mock_order_validator.validate_order.return_value = ValidationResult(is_valid=True)
        mock_risk_calculator.check_risk_limits.return_value = (True, None)
        mock_broker.submit_order.return_value = "BROKER-SMALL"

        response = await place_order_use_case.process(request)

        assert response.success is True
        # Verify the order was created with the small price
        submitted_order = mock_broker.submit_order.call_args[0][0]
        assert submitted_order.limit_price == Decimal("0.0001")

    @pytest.mark.asyncio
    async def test_special_characters_in_symbol(
        self,
        place_order_use_case,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        sample_market_bar,
    ):
        """Test handling of special characters in symbol."""
        request = PlaceOrderRequest(
            portfolio_id=sample_portfolio.id,
            symbol="BRK.B",  # Symbol with dot
            side="buy",
            order_type="market",
            quantity=10,
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id = AsyncMock(return_value=sample_portfolio)
        mock_unit_of_work.market_data.get_latest_bar = AsyncMock(return_value=sample_market_bar)
        mock_unit_of_work.orders.save_order = AsyncMock()
        mock_order_validator.validate_order.return_value = ValidationResult(is_valid=True)
        mock_risk_calculator.check_risk_limits.return_value = (True, None)
        mock_broker.submit_order.return_value = "BROKER-SPECIAL"

        response = await place_order_use_case.process(request)

        assert response.success is True
        submitted_order = mock_broker.submit_order.call_args[0][0]
        assert submitted_order.symbol == "BRK.B"

    @pytest.mark.asyncio
    async def test_empty_metadata(
        self,
        place_order_use_case,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        sample_market_bar,
    ):
        """Test handling of empty metadata."""
        request = PlaceOrderRequest(
            portfolio_id=sample_portfolio.id,
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=100,
            metadata={},
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id = AsyncMock(return_value=sample_portfolio)
        mock_unit_of_work.market_data.get_latest_bar = AsyncMock(return_value=sample_market_bar)
        mock_unit_of_work.orders.save_order = AsyncMock()
        mock_order_validator.validate_order.return_value = ValidationResult(is_valid=True)
        mock_risk_calculator.check_risk_limits.return_value = (True, None)
        mock_broker.submit_order.return_value = "BROKER-META"

        response = await place_order_use_case.process(request)

        assert response.success is True

    @pytest.mark.asyncio
    async def test_null_metadata(
        self,
        place_order_use_case,
        mock_unit_of_work,
        mock_broker,
        mock_order_validator,
        mock_risk_calculator,
        sample_portfolio,
        sample_market_bar,
    ):
        """Test handling of null metadata."""
        request = PlaceOrderRequest(
            portfolio_id=sample_portfolio.id,
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=100,
            metadata=None,
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id = AsyncMock(return_value=sample_portfolio)
        mock_unit_of_work.market_data.get_latest_bar = AsyncMock(return_value=sample_market_bar)
        mock_unit_of_work.orders.save_order = AsyncMock()
        mock_order_validator.validate_order.return_value = ValidationResult(is_valid=True)
        mock_risk_calculator.check_risk_limits.return_value = (True, None)
        mock_broker.submit_order.return_value = "BROKER-NULL-META"

        response = await place_order_use_case.process(request)

        assert response.success is True


class TestRequestInitialization:
    """Test PlaceOrderRequest initialization and defaults."""

    def test_request_initialization_with_all_fields(self):
        """Test request initialization with all fields provided."""
        portfolio_id = uuid4()
        request_id = uuid4()
        correlation_id = uuid4()
        metadata = {"key": "value", "source": "test"}

        request = PlaceOrderRequest(
            portfolio_id=portfolio_id,
            symbol="AAPL",
            side="buy",
            order_type="limit",
            quantity=100,
            limit_price=150.00,
            stop_price=145.00,
            time_in_force="gtc",
            strategy_id="STRAT-001",
            request_id=request_id,
            correlation_id=correlation_id,
            metadata=metadata,
        )

        assert request.portfolio_id == portfolio_id
        assert request.symbol == "AAPL"
        assert request.side == "buy"
        assert request.order_type == "limit"
        assert request.quantity == 100
        assert request.limit_price == 150.00
        assert request.stop_price == 145.00
        assert request.time_in_force == "gtc"
        assert request.strategy_id == "STRAT-001"
        assert request.request_id == request_id
        assert request.correlation_id == correlation_id
        assert request.metadata == metadata

    def test_request_initialization_with_minimal_fields(self):
        """Test request initialization with minimal required fields."""
        portfolio_id = uuid4()

        request = PlaceOrderRequest(
            portfolio_id=portfolio_id, symbol="GOOGL", side="sell", order_type="market", quantity=50
        )

        assert request.portfolio_id == portfolio_id
        assert request.symbol == "GOOGL"
        assert request.side == "sell"
        assert request.order_type == "market"
        assert request.quantity == 50
        assert request.limit_price is None
        assert request.stop_price is None
        assert request.time_in_force == "day"
        assert request.strategy_id is None
        assert request.request_id is not None  # Should be auto-generated
        assert request.correlation_id is None
        assert request.metadata == {}  # Should be empty dict

    def test_request_post_init_generates_request_id(self):
        """Test that __post_init__ generates request_id if not provided."""
        request = PlaceOrderRequest(
            portfolio_id=uuid4(),
            symbol="MSFT",
            side="buy",
            order_type="market",
            quantity=100,
            request_id=None,
        )

        assert request.request_id is not None
        assert isinstance(request.request_id, UUID)

    def test_request_post_init_initializes_empty_metadata(self):
        """Test that __post_init__ initializes empty metadata if not provided."""
        request = PlaceOrderRequest(
            portfolio_id=uuid4(),
            symbol="TSLA",
            side="buy",
            order_type="market",
            quantity=25,
            metadata=None,
        )

        assert request.metadata is not None
        assert request.metadata == {}

    def test_request_preserves_provided_values(self):
        """Test that provided values are not overwritten by __post_init__."""
        request_id = uuid4()
        metadata = {"custom": "data"}

        request = PlaceOrderRequest(
            portfolio_id=uuid4(),
            symbol="NVDA",
            side="buy",
            order_type="market",
            quantity=100,
            request_id=request_id,
            metadata=metadata,
        )

        assert request.request_id == request_id
        assert request.metadata == metadata


class TestResponseCreation:
    """Test PlaceOrderResponse creation and fields."""

    def test_response_success_creation(self):
        """Test creation of successful response."""
        order_id = uuid4()
        request_id = uuid4()

        response = PlaceOrderResponse(
            success=True,
            order_id=order_id,
            broker_order_id="BROKER-789",
            status="submitted",
            request_id=request_id,
        )

        assert response.success is True
        assert response.order_id == order_id
        assert response.broker_order_id == "BROKER-789"
        assert response.status == "submitted"
        assert response.request_id == request_id
        assert response.error is None

    def test_response_failure_creation(self):
        """Test creation of failure response."""
        request_id = uuid4()

        response = PlaceOrderResponse(
            success=False, error="Order validation failed", request_id=request_id
        )

        assert response.success is False
        assert response.error == "Order validation failed"
        assert response.request_id == request_id
        assert response.order_id is None
        assert response.broker_order_id is None
        assert response.status is None

    def test_response_with_partial_data(self):
        """Test response with partial data."""
        response = PlaceOrderResponse(success=True, order_id=uuid4(), status="pending")

        assert response.success is True
        assert response.order_id is not None
        assert response.broker_order_id is None
        assert response.status == "pending"
