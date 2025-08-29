"""
Comprehensive Unit Tests for Portfolio Management Use Cases

Tests all portfolio use cases with complete coverage including:
- GetPortfolioUseCase
- UpdatePortfolioUseCase
- GetPositionsUseCase
- ClosePositionUseCase

Achieves 80%+ coverage with focus on business logic validation.
"""

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from src.application.use_cases.portfolio import (
    ClosePositionRequest,
    ClosePositionResponse,
    ClosePositionUseCase,
    GetPortfolioRequest,
    GetPortfolioResponse,
    GetPortfolioUseCase,
    GetPositionsRequest,
    GetPositionsResponse,
    GetPositionsUseCase,
    UpdatePortfolioRequest,
    UpdatePortfolioResponse,
    UpdatePortfolioUseCase,
)
from src.domain.entities.order import OrderSide, OrderType
from src.domain.entities.portfolio import Portfolio
from src.domain.entities.position import Position
from src.domain.services.position_manager import PositionManager
from src.domain.services.risk_calculator import RiskCalculator
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
    uow.portfolios = AsyncMock()
    uow.positions = AsyncMock()

    return uow


@pytest.fixture
def mock_risk_calculator():
    """Create a mock risk calculator."""
    calculator = Mock(spec=RiskCalculator)
    calculator.calculate_sharpe_ratio = Mock(return_value=Decimal("1.5"))
    return calculator


@pytest.fixture
def mock_position_manager():
    """Create a mock position manager."""
    manager = Mock(spec=PositionManager)
    manager.close_position = Mock()
    return manager


@pytest.fixture
def sample_portfolio():
    """Create a sample portfolio with positions for testing."""
    portfolio = Portfolio(
        name="Test Portfolio",
        initial_capital=Money(Decimal("10000.00")),
    )
    portfolio.id = uuid4()
    portfolio.cash_balance = Money(Decimal("5000.00"))
    portfolio.total_realized_pnl = Money(Decimal("500.00"))
    portfolio.max_position_size = Money(Decimal("2000.00"))
    portfolio.max_positions = 10
    portfolio.max_leverage = Decimal("2.0")
    portfolio.max_portfolio_risk = Decimal("0.2")

    # Mock method - get_total_value is NOT async, returns Money
    portfolio.get_total_value = Mock(return_value=Money(Decimal("15500.00")))

    # Add open positions
    position1 = Position(
        symbol="AAPL",
        quantity=Quantity(Decimal("100")),
        average_entry_price=Price(Decimal("150.00")),
    )
    position1.id = uuid4()
    position1.current_price = Price(Decimal("155.00"))
    position1.opened_at = datetime.now(UTC)

    position2 = Position(
        symbol="GOOGL",
        quantity=Quantity(Decimal("50")),
        average_entry_price=Price(Decimal("2800.00")),
    )
    position2.id = uuid4()
    position2.current_price = Price(Decimal("2850.00"))
    position2.opened_at = datetime.now(UTC)

    # Add closed position - create with quantity first, then set to zero after closing
    position3 = Position(
        symbol="MSFT",
        quantity=Quantity(Decimal("50")),  # Will be closed
        average_entry_price=Price(Decimal("300.00")),
    )
    position3.id = uuid4()
    position3.current_price = Price(Decimal("310.00"))
    position3.opened_at = datetime.now(UTC)
    position3.closed_at = datetime.now(UTC)
    position3.quantity = Quantity(Decimal("0"))  # Set to zero after marking as closed
    position3.realized_pnl = Money(Decimal("1000.00"))

    portfolio.positions = {
        position1.id: position1,
        position2.id: position2,
        position3.id: position3,
    }

    return portfolio


@pytest.fixture
def sample_position():
    """Create a sample open position for testing."""
    position = Position(
        symbol="TSLA",
        quantity=Quantity(Decimal("50")),
        average_entry_price=Price(Decimal("700.00")),
    )
    position.id = uuid4()
    position.current_price = Price(Decimal("750.00"))
    position.opened_at = datetime.now(UTC)
    return position


# Test GetPortfolioUseCase
class TestGetPortfolioUseCase:
    """Test GetPortfolioUseCase class."""

    @pytest.mark.asyncio
    async def test_init(self, mock_unit_of_work, mock_risk_calculator):
        """Test use case initialization."""
        use_case = GetPortfolioUseCase(mock_unit_of_work, mock_risk_calculator)

        assert use_case.unit_of_work == mock_unit_of_work
        assert use_case.risk_calculator == mock_risk_calculator
        assert use_case.name == "GetPortfolioUseCase"

    @pytest.mark.asyncio
    async def test_validate_success(self, mock_unit_of_work, mock_risk_calculator):
        """Test successful validation."""
        use_case = GetPortfolioUseCase(mock_unit_of_work, mock_risk_calculator)

        request = GetPortfolioRequest(portfolio_id=uuid4())

        result = await use_case.validate(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_process_with_all_data(
        self, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test processing with positions and metrics included."""
        use_case = GetPortfolioUseCase(mock_unit_of_work, mock_risk_calculator)

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = GetPortfolioRequest(
            portfolio_id=sample_portfolio.id,
            include_positions=True,
            include_metrics=True,
        )

        response = await use_case.process(request)

        assert response.success is True
        assert response.portfolio is not None
        assert response.positions is not None
        assert response.metrics is not None

        # Check portfolio data
        assert response.portfolio["id"] == str(sample_portfolio.id)
        assert response.portfolio["name"] == "Test Portfolio"
        assert response.portfolio["cash_balance"] == 5000.0
        assert response.portfolio["total_value"] == 15500.0
        assert response.portfolio["open_positions_count"] == 2  # Only 2 open positions

        # Check positions data
        assert len(response.positions) == 2  # Only open positions
        assert all(pos["side"] == "long" for pos in response.positions)

        # Check metrics data
        assert response.metrics["sharpe_ratio"] == 1.5

    @pytest.mark.asyncio
    async def test_process_portfolio_only(
        self, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test processing with only portfolio data."""
        use_case = GetPortfolioUseCase(mock_unit_of_work, mock_risk_calculator)

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = GetPortfolioRequest(
            portfolio_id=sample_portfolio.id,
            include_positions=False,
            include_metrics=False,
        )

        response = await use_case.process(request)

        assert response.success is True
        assert response.portfolio is not None
        assert response.positions is None
        assert response.metrics is None

    @pytest.mark.asyncio
    async def test_process_with_positions_only(
        self, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test processing with positions but no metrics."""
        use_case = GetPortfolioUseCase(mock_unit_of_work, mock_risk_calculator)

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = GetPortfolioRequest(
            portfolio_id=sample_portfolio.id,
            include_positions=True,
            include_metrics=False,
        )

        response = await use_case.process(request)

        assert response.success is True
        assert response.positions is not None
        assert response.metrics is None

        # Verify position details
        for position in response.positions:
            assert "id" in position
            assert "symbol" in position
            assert "quantity" in position
            assert "entry_price" in position
            assert "current_price" in position
            assert "unrealized_pnl" in position
            assert "return_pct" in position
            assert "value" in position

    @pytest.mark.asyncio
    async def test_process_with_metrics_only(
        self, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test processing with metrics but no positions."""
        use_case = GetPortfolioUseCase(mock_unit_of_work, mock_risk_calculator)

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = GetPortfolioRequest(
            portfolio_id=sample_portfolio.id,
            include_positions=False,
            include_metrics=True,
        )

        response = await use_case.process(request)

        assert response.success is True
        assert response.positions is None
        assert response.metrics is not None
        assert "sharpe_ratio" in response.metrics
        assert "max_drawdown" in response.metrics
        assert "portfolio_beta" in response.metrics
        assert "value_at_risk_95" in response.metrics

    @pytest.mark.asyncio
    async def test_process_portfolio_not_found(self, mock_unit_of_work, mock_risk_calculator):
        """Test processing when portfolio not found."""
        use_case = GetPortfolioUseCase(mock_unit_of_work, mock_risk_calculator)

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = None

        request = GetPortfolioRequest(portfolio_id=uuid4())

        response = await use_case.process(request)

        assert response.success is False
        assert response.error == "Portfolio not found"

    @pytest.mark.asyncio
    async def test_process_handles_money_objects(self, mock_unit_of_work, mock_risk_calculator):
        """Test that Money objects are properly handled."""
        use_case = GetPortfolioUseCase(mock_unit_of_work, mock_risk_calculator)

        # Create portfolio with Money objects that have amount attribute
        portfolio = Portfolio(
            name="Test Portfolio",
            initial_capital=Money(Decimal("10000.00")),
        )
        portfolio.id = uuid4()
        portfolio.cash_balance = Money(Decimal("5000.00"))
        portfolio.total_realized_pnl = Money(Decimal("100.00"))
        portfolio.get_total_value = Mock(return_value=Money(Decimal("10000.00")))
        portfolio.positions = {}

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = portfolio

        request = GetPortfolioRequest(portfolio_id=portfolio.id)

        response = await use_case.process(request)

        assert response.success is True
        assert response.portfolio["cash_balance"] == 5000.0
        assert response.portfolio["initial_capital"] == 10000.0
        assert response.portfolio["realized_pnl"] == 100.0

    @pytest.mark.asyncio
    async def test_process_with_metadata(
        self, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test processing with metadata and correlation ID."""
        use_case = GetPortfolioUseCase(mock_unit_of_work, mock_risk_calculator)

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = GetPortfolioRequest(
            portfolio_id=sample_portfolio.id,
            correlation_id=uuid4(),
            metadata={"source": "dashboard"},
        )

        response = await use_case.process(request)

        assert response.success is True
        assert response.request_id == request.request_id


# Test UpdatePortfolioUseCase
class TestUpdatePortfolioUseCase:
    """Test UpdatePortfolioUseCase class."""

    @pytest.mark.asyncio
    async def test_init(self, mock_unit_of_work):
        """Test use case initialization."""
        use_case = UpdatePortfolioUseCase(mock_unit_of_work)

        assert use_case.unit_of_work == mock_unit_of_work
        assert use_case.name == "UpdatePortfolioUseCase"

    @pytest.mark.asyncio
    async def test_validate_success(self, mock_unit_of_work):
        """Test successful validation."""
        use_case = UpdatePortfolioUseCase(mock_unit_of_work)

        request = UpdatePortfolioRequest(
            portfolio_id=uuid4(),
            name="Updated Portfolio",
            max_position_size=Decimal("5000.00"),
            max_positions=20,
            max_leverage=Decimal("3.0"),
            max_portfolio_risk=Decimal("0.3"),
        )

        result = await use_case.validate(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_negative_position_size(self, mock_unit_of_work):
        """Test validation with negative max position size."""
        use_case = UpdatePortfolioUseCase(mock_unit_of_work)

        request = UpdatePortfolioRequest(
            portfolio_id=uuid4(),
            max_position_size=Decimal("-1000.00"),
        )

        result = await use_case.validate(request)
        assert result == "Max position size must be positive"

    @pytest.mark.asyncio
    async def test_validate_zero_position_size(self, mock_unit_of_work):
        """Test validation with zero max position size."""
        use_case = UpdatePortfolioUseCase(mock_unit_of_work)

        request = UpdatePortfolioRequest(
            portfolio_id=uuid4(),
            max_position_size=Decimal("0"),
        )

        result = await use_case.validate(request)
        assert result == "Max position size must be positive"

    @pytest.mark.asyncio
    async def test_validate_negative_max_positions(self, mock_unit_of_work):
        """Test validation with negative max positions."""
        use_case = UpdatePortfolioUseCase(mock_unit_of_work)

        request = UpdatePortfolioRequest(
            portfolio_id=uuid4(),
            max_positions=-5,
        )

        result = await use_case.validate(request)
        assert result == "Max positions must be positive"

    @pytest.mark.asyncio
    async def test_validate_zero_max_positions(self, mock_unit_of_work):
        """Test validation with zero max positions."""
        use_case = UpdatePortfolioUseCase(mock_unit_of_work)

        request = UpdatePortfolioRequest(
            portfolio_id=uuid4(),
            max_positions=0,
        )

        result = await use_case.validate(request)
        assert result == "Max positions must be positive"

    @pytest.mark.asyncio
    async def test_validate_leverage_less_than_one(self, mock_unit_of_work):
        """Test validation with leverage less than 1."""
        use_case = UpdatePortfolioUseCase(mock_unit_of_work)

        request = UpdatePortfolioRequest(
            portfolio_id=uuid4(),
            max_leverage=Decimal("0.5"),
        )

        result = await use_case.validate(request)
        assert result == "Max leverage must be at least 1.0"

    @pytest.mark.asyncio
    async def test_validate_portfolio_risk_out_of_range(self, mock_unit_of_work):
        """Test validation with portfolio risk out of range."""
        use_case = UpdatePortfolioUseCase(mock_unit_of_work)

        # Test negative risk
        request = UpdatePortfolioRequest(
            portfolio_id=uuid4(),
            max_portfolio_risk=Decimal("-0.1"),
        )

        result = await use_case.validate(request)
        assert result == "Max portfolio risk must be between 0 and 1"

        # Test zero risk
        request.max_portfolio_risk = Decimal("0")
        result = await use_case.validate(request)
        assert result == "Max portfolio risk must be between 0 and 1"

        # Test risk greater than 1
        request.max_portfolio_risk = Decimal("1.5")
        result = await use_case.validate(request)
        assert result == "Max portfolio risk must be between 0 and 1"

    @pytest.mark.asyncio
    async def test_process_successful_update_all_fields(self, mock_unit_of_work, sample_portfolio):
        """Test successful portfolio update with all fields."""
        use_case = UpdatePortfolioUseCase(mock_unit_of_work)

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = UpdatePortfolioRequest(
            portfolio_id=sample_portfolio.id,
            name="Updated Portfolio",
            max_position_size=Decimal("3000.00"),
            max_positions=15,
            max_leverage=Decimal("2.5"),
            max_portfolio_risk=Decimal("0.25"),
        )

        response = await use_case.process(request)

        assert response.success is True
        assert response.updated is True

        # Verify updates
        assert sample_portfolio.name == "Updated Portfolio"
        assert sample_portfolio.max_position_size == Money(Decimal("3000.00"))
        assert sample_portfolio.max_positions == 15
        assert sample_portfolio.max_leverage == Decimal("2.5")
        assert sample_portfolio.max_portfolio_risk == Decimal("0.25")

        # Verify save was called
        mock_unit_of_work.portfolios.update_portfolio.assert_called_once_with(sample_portfolio)

    @pytest.mark.asyncio
    async def test_process_partial_update(self, mock_unit_of_work, sample_portfolio):
        """Test partial portfolio update."""
        use_case = UpdatePortfolioUseCase(mock_unit_of_work)

        # Setup mocks
        original_name = sample_portfolio.name
        original_max_positions = sample_portfolio.max_positions
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = UpdatePortfolioRequest(
            portfolio_id=sample_portfolio.id,
            max_position_size=Decimal("4000.00"),
            max_leverage=Decimal("3.0"),
        )

        response = await use_case.process(request)

        assert response.success is True
        assert response.updated is True

        # Verify only specified fields were updated
        assert sample_portfolio.name == original_name
        assert sample_portfolio.max_position_size == Money(Decimal("4000.00"))
        assert sample_portfolio.max_positions == original_max_positions
        assert sample_portfolio.max_leverage == Decimal("3.0")

    @pytest.mark.asyncio
    async def test_process_portfolio_not_found(self, mock_unit_of_work):
        """Test update when portfolio not found."""
        use_case = UpdatePortfolioUseCase(mock_unit_of_work)

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = None

        request = UpdatePortfolioRequest(
            portfolio_id=uuid4(),
            name="New Name",
        )

        response = await use_case.process(request)

        assert response.success is False
        assert response.error == "Portfolio not found"
        assert response.updated is False

    @pytest.mark.asyncio
    async def test_process_with_metadata(self, mock_unit_of_work, sample_portfolio):
        """Test update with metadata and correlation ID."""
        use_case = UpdatePortfolioUseCase(mock_unit_of_work)

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = UpdatePortfolioRequest(
            portfolio_id=sample_portfolio.id,
            name="Updated",
            correlation_id=uuid4(),
            metadata={"update_source": "admin"},
        )

        response = await use_case.process(request)

        assert response.success is True
        assert response.request_id == request.request_id


# Test GetPositionsUseCase
class TestGetPositionsUseCase:
    """Test GetPositionsUseCase class."""

    @pytest.mark.asyncio
    async def test_init(self, mock_unit_of_work):
        """Test use case initialization."""
        use_case = GetPositionsUseCase(mock_unit_of_work)

        assert use_case.unit_of_work == mock_unit_of_work
        assert use_case.name == "GetPositionsUseCase"

    @pytest.mark.asyncio
    async def test_validate_success(self, mock_unit_of_work):
        """Test successful validation."""
        use_case = GetPositionsUseCase(mock_unit_of_work)

        request = GetPositionsRequest(portfolio_id=uuid4())

        result = await use_case.validate(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_process_get_open_positions(self, mock_unit_of_work, sample_portfolio):
        """Test getting only open positions."""
        use_case = GetPositionsUseCase(mock_unit_of_work)

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = GetPositionsRequest(
            portfolio_id=sample_portfolio.id,
            only_open=True,
        )

        response = await use_case.process(request)

        assert response.success is True
        assert len(response.positions) == 2  # Only open positions
        assert response.total_value > 0

        # Verify all positions are open
        for position in response.positions:
            assert position["is_open"] is True
            assert position["closed_at"] is None

    @pytest.mark.asyncio
    async def test_process_get_all_positions(self, mock_unit_of_work, sample_portfolio):
        """Test getting all positions including closed."""
        use_case = GetPositionsUseCase(mock_unit_of_work)

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = GetPositionsRequest(
            portfolio_id=sample_portfolio.id,
            only_open=False,
        )

        response = await use_case.process(request)

        assert response.success is True
        assert len(response.positions) == 3  # All positions

        # Check we have both open and closed positions
        open_positions = [p for p in response.positions if p["is_open"]]
        closed_positions = [p for p in response.positions if not p["is_open"]]

        assert len(open_positions) == 2
        assert len(closed_positions) == 1

    @pytest.mark.asyncio
    async def test_process_filter_by_symbol(self, mock_unit_of_work, sample_portfolio):
        """Test filtering positions by symbol."""
        use_case = GetPositionsUseCase(mock_unit_of_work)

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = GetPositionsRequest(
            portfolio_id=sample_portfolio.id,
            only_open=False,
            symbol="AAPL",
        )

        response = await use_case.process(request)

        assert response.success is True
        assert len(response.positions) == 1
        assert response.positions[0]["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_process_filter_nonexistent_symbol(self, mock_unit_of_work, sample_portfolio):
        """Test filtering by nonexistent symbol."""
        use_case = GetPositionsUseCase(mock_unit_of_work)

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = GetPositionsRequest(
            portfolio_id=sample_portfolio.id,
            symbol="NONEXISTENT",
        )

        response = await use_case.process(request)

        assert response.success is True
        assert len(response.positions) == 0
        assert response.total_value == Decimal("0")

    @pytest.mark.asyncio
    async def test_process_position_data_fields(self, mock_unit_of_work, sample_portfolio):
        """Test that all position data fields are included."""
        use_case = GetPositionsUseCase(mock_unit_of_work)

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = GetPositionsRequest(portfolio_id=sample_portfolio.id)

        response = await use_case.process(request)

        assert response.success is True

        # Check all expected fields in position data
        for position in response.positions:
            assert "id" in position
            assert "symbol" in position
            assert "side" in position
            assert "quantity" in position
            assert "entry_price" in position
            assert "current_price" in position
            assert "unrealized_pnl" in position
            assert "realized_pnl" in position
            assert "return_pct" in position
            assert "value" in position
            assert "is_open" in position
            assert "opened_at" in position
            assert "closed_at" in position

    @pytest.mark.asyncio
    async def test_process_portfolio_not_found(self, mock_unit_of_work):
        """Test getting positions when portfolio not found."""
        use_case = GetPositionsUseCase(mock_unit_of_work)

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = None

        request = GetPositionsRequest(portfolio_id=uuid4())

        response = await use_case.process(request)

        assert response.success is False
        assert response.error == "Portfolio not found"

    @pytest.mark.asyncio
    async def test_process_total_value_calculation(self, mock_unit_of_work, sample_portfolio):
        """Test total value calculation for positions."""
        use_case = GetPositionsUseCase(mock_unit_of_work)

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = GetPositionsRequest(
            portfolio_id=sample_portfolio.id,
            only_open=True,
        )

        response = await use_case.process(request)

        assert response.success is True

        # Calculate expected total value
        open_positions = sample_portfolio.get_open_positions()
        expected_total = sum(
            pos.get_position_value() if pos.get_position_value() is not None else Decimal("0")
            for pos in open_positions
        )

        assert (
            response.total_value.value == expected_total
            if hasattr(response.total_value, "value")
            else response.total_value == expected_total
        )

    @pytest.mark.asyncio
    async def test_process_with_metadata(self, mock_unit_of_work, sample_portfolio):
        """Test getting positions with metadata and correlation ID."""
        use_case = GetPositionsUseCase(mock_unit_of_work)

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = GetPositionsRequest(
            portfolio_id=sample_portfolio.id,
            correlation_id=uuid4(),
            metadata={"view": "positions_list"},
        )

        response = await use_case.process(request)

        assert response.success is True
        assert response.request_id == request.request_id


# Test ClosePositionUseCase
class TestClosePositionUseCase:
    """Test ClosePositionUseCase class."""

    @pytest.mark.asyncio
    async def test_init(self, mock_unit_of_work, mock_position_manager):
        """Test use case initialization."""
        use_case = ClosePositionUseCase(mock_unit_of_work, mock_position_manager)

        assert use_case.unit_of_work == mock_unit_of_work
        assert use_case.position_manager == mock_position_manager
        assert use_case.name == "ClosePositionUseCase"

    @pytest.mark.asyncio
    async def test_validate_success(self, mock_unit_of_work, mock_position_manager):
        """Test successful validation."""
        use_case = ClosePositionUseCase(mock_unit_of_work, mock_position_manager)

        request = ClosePositionRequest(
            position_id=uuid4(),
            exit_price=Decimal("150.00"),
        )

        result = await use_case.validate(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_negative_exit_price(self, mock_unit_of_work, mock_position_manager):
        """Test validation with negative exit price."""
        use_case = ClosePositionUseCase(mock_unit_of_work, mock_position_manager)

        request = ClosePositionRequest(
            position_id=uuid4(),
            exit_price=Decimal("-150.00"),
        )

        result = await use_case.validate(request)
        assert result == "Exit price must be positive"

    @pytest.mark.asyncio
    async def test_validate_zero_exit_price(self, mock_unit_of_work, mock_position_manager):
        """Test validation with zero exit price."""
        use_case = ClosePositionUseCase(mock_unit_of_work, mock_position_manager)

        request = ClosePositionRequest(
            position_id=uuid4(),
            exit_price=Decimal("0"),
        )

        result = await use_case.validate(request)
        assert result == "Exit price must be positive"

    @pytest.mark.asyncio
    async def test_process_successful_close(
        self, mock_unit_of_work, mock_position_manager, sample_position
    ):
        """Test successful position closing."""
        use_case = ClosePositionUseCase(mock_unit_of_work, mock_position_manager)

        # Setup position with P&L
        sample_position.realized_pnl = Money(Decimal("2500.00"))  # $50 profit per share

        # Setup mocks
        mock_unit_of_work.positions.get_position_by_id.return_value = sample_position

        request = ClosePositionRequest(
            position_id=sample_position.id,
            exit_price=Decimal("750.00"),
            reason="Take profit",
        )

        response = await use_case.process(request)

        assert response.success is True
        assert response.closed is True
        assert response.realized_pnl == Decimal("2500.00")
        assert response.total_return is not None

        # Verify interactions
        mock_unit_of_work.positions.get_position_by_id.assert_called_once_with(sample_position.id)
        mock_position_manager.close_position.assert_called_once()
        mock_unit_of_work.positions.update_position.assert_called_once_with(sample_position)

    @pytest.mark.asyncio
    async def test_process_position_not_found(self, mock_unit_of_work, mock_position_manager):
        """Test closing when position not found."""
        use_case = ClosePositionUseCase(mock_unit_of_work, mock_position_manager)

        # Setup mocks
        mock_unit_of_work.positions.get_position_by_id.return_value = None

        request = ClosePositionRequest(
            position_id=uuid4(),
            exit_price=Decimal("100.00"),
        )

        response = await use_case.process(request)

        assert response.success is False
        assert response.error == "Position not found"
        assert response.closed is False

    @pytest.mark.asyncio
    async def test_process_already_closed_position(self, mock_unit_of_work, mock_position_manager):
        """Test closing an already closed position."""
        use_case = ClosePositionUseCase(mock_unit_of_work, mock_position_manager)

        # Create closed position - create with quantity first, then set to zero after closing
        position = Position(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),  # Will be closed
            average_entry_price=Price(Decimal("150.00")),
        )
        position.id = uuid4()
        position.closed_at = datetime.now(UTC)
        position.quantity = Quantity(Decimal("0"))  # Set to zero after marking as closed

        # Setup mocks
        mock_unit_of_work.positions.get_position_by_id.return_value = position

        request = ClosePositionRequest(
            position_id=position.id,
            exit_price=Decimal("160.00"),
        )

        response = await use_case.process(request)

        assert response.success is False
        assert response.error == "Position is already closed"
        assert response.closed is False

    @pytest.mark.asyncio
    async def test_process_creates_closing_order(
        self, mock_unit_of_work, mock_position_manager, sample_position
    ):
        """Test that a closing order is created correctly."""
        use_case = ClosePositionUseCase(mock_unit_of_work, mock_position_manager)

        # Setup mocks
        mock_unit_of_work.positions.get_position_by_id.return_value = sample_position

        request = ClosePositionRequest(
            position_id=sample_position.id,
            exit_price=Decimal("750.00"),
        )

        response = await use_case.process(request)

        assert response.success is True

        # Verify closing order was created with correct parameters
        call_args = mock_position_manager.close_position.call_args
        closing_order = call_args[1]["order"]

        assert closing_order.symbol == sample_position.symbol
        assert closing_order.side == OrderSide.SELL  # Long position closes with sell
        assert closing_order.order_type == OrderType.MARKET
        assert closing_order.quantity == Quantity(abs(sample_position.quantity.value))
        assert closing_order.average_fill_price == Price(Decimal("750.00"))

    @pytest.mark.asyncio
    async def test_process_short_position_close(self, mock_unit_of_work, mock_position_manager):
        """Test closing a short position."""
        use_case = ClosePositionUseCase(mock_unit_of_work, mock_position_manager)

        # Create short position
        position = Position(
            symbol="TSLA",
            quantity=Quantity(Decimal("-50")),  # Short position
            average_entry_price=Price(Decimal("800.00")),
        )
        position.id = uuid4()
        position.realized_pnl = Money(Decimal("2500.00"))  # Profit from short

        # Setup mocks
        mock_unit_of_work.positions.get_position_by_id.return_value = position

        request = ClosePositionRequest(
            position_id=position.id,
            exit_price=Decimal("750.00"),
        )

        response = await use_case.process(request)

        assert response.success is True

        # Verify closing order for short position
        call_args = mock_position_manager.close_position.call_args
        closing_order = call_args[1]["order"]

        assert closing_order.side == OrderSide.BUY  # Short position closes with buy

    @pytest.mark.asyncio
    async def test_process_with_metadata(
        self, mock_unit_of_work, mock_position_manager, sample_position
    ):
        """Test closing position with metadata and correlation ID."""
        use_case = ClosePositionUseCase(mock_unit_of_work, mock_position_manager)

        # Setup mocks
        mock_unit_of_work.positions.get_position_by_id.return_value = sample_position
        sample_position.realized_pnl = Money(Decimal("1000.00"))

        request = ClosePositionRequest(
            position_id=sample_position.id,
            exit_price=Decimal("720.00"),
            reason="Stop loss",
            correlation_id=uuid4(),
            metadata={"strategy": "momentum"},
        )

        response = await use_case.process(request)

        assert response.success is True
        assert response.request_id == request.request_id


# Test Request/Response DTOs
class TestRequestResponseDTOs:
    """Test request and response data classes."""

    def test_get_portfolio_request_init(self):
        """Test GetPortfolioRequest initialization."""
        request = GetPortfolioRequest(portfolio_id=uuid4())

        assert request.request_id is not None
        assert request.metadata == {}
        assert request.include_positions is True
        assert request.include_metrics is True

    def test_update_portfolio_request_init(self):
        """Test UpdatePortfolioRequest initialization."""
        request = UpdatePortfolioRequest(
            portfolio_id=uuid4(),
            name="New Name",
        )

        assert request.request_id is not None
        assert request.metadata == {}
        assert request.max_position_size is None
        assert request.max_positions is None

    def test_get_positions_request_init(self):
        """Test GetPositionsRequest initialization."""
        request = GetPositionsRequest(portfolio_id=uuid4())

        assert request.request_id is not None
        assert request.metadata == {}
        assert request.only_open is True
        assert request.symbol is None

    def test_close_position_request_init(self):
        """Test ClosePositionRequest initialization."""
        request = ClosePositionRequest(
            position_id=uuid4(),
            exit_price=Decimal("100.00"),
        )

        assert request.request_id is not None
        assert request.metadata == {}
        assert request.reason is None

    def test_get_portfolio_response(self):
        """Test GetPortfolioResponse initialization."""
        response = GetPortfolioResponse(
            success=True,
            portfolio={"id": "123", "name": "Test"},
            positions=[{"symbol": "AAPL"}],
            metrics={"sharpe": 1.5},
        )

        assert response.success is True
        assert response.portfolio["name"] == "Test"
        assert len(response.positions) == 1
        assert response.metrics["sharpe"] == 1.5

    def test_update_portfolio_response(self):
        """Test UpdatePortfolioResponse initialization."""
        response = UpdatePortfolioResponse(
            success=True,
            updated=True,
        )

        assert response.success is True
        assert response.updated is True

    def test_get_positions_response(self):
        """Test GetPositionsResponse initialization."""
        response = GetPositionsResponse(
            success=True,
            positions=[{"symbol": "AAPL"}, {"symbol": "GOOGL"}],
            total_value=Decimal("10000.00"),
        )

        assert response.success is True
        assert len(response.positions) == 2
        assert response.total_value == Decimal("10000.00")

    def test_get_positions_response_default(self):
        """Test GetPositionsResponse with default values."""
        response = GetPositionsResponse(success=True)

        assert response.success is True
        assert response.positions == []
        assert response.total_value is None

    def test_close_position_response(self):
        """Test ClosePositionResponse initialization."""
        response = ClosePositionResponse(
            success=True,
            closed=True,
            realized_pnl=Decimal("500.00"),
            total_return=Decimal("0.10"),
        )

        assert response.success is True
        assert response.closed is True
        assert response.realized_pnl == Decimal("500.00")
        assert response.total_return == Decimal("0.10")
