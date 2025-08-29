"""
Comprehensive tests for portfolio management use cases with full coverage.

Tests all portfolio-related use cases including:
- GetPortfolioUseCase
- UpdatePortfolioUseCase
- GetPositionsUseCase
- ClosePositionUseCase

Covers all scenarios including success, failure, validation, and edge cases.
"""

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from src.application.interfaces.repositories import IPortfolioRepository, IPositionRepository
from src.application.interfaces.unit_of_work import IUnitOfWork
from src.application.use_cases.portfolio import (
    ClosePositionRequest,
    ClosePositionUseCase,
    GetPortfolioRequest,
    GetPortfolioResponse,
    GetPortfolioUseCase,
    GetPositionsRequest,
    GetPositionsUseCase,
    UpdatePortfolioRequest,
    UpdatePortfolioUseCase,
)
from src.domain.entities.order import OrderSide
from src.domain.entities.portfolio import Portfolio
from src.domain.entities.position import Position
from src.domain.services.position_manager import PositionManager
from src.domain.services.risk_calculator import RiskCalculator
from src.domain.value_objects.money import Money
from src.domain.value_objects.price import Price
from src.domain.value_objects.quantity import Quantity


@pytest.fixture
def mock_unit_of_work():
    """Create a mock unit of work."""
    uow = Mock(spec=IUnitOfWork)

    # Setup repository mocks
    uow.portfolios = AsyncMock(spec=IPortfolioRepository)
    uow.positions = AsyncMock(spec=IPositionRepository)
    uow.orders = AsyncMock()

    # Setup transaction methods
    uow.begin = AsyncMock()
    uow.commit = AsyncMock()
    uow.rollback = AsyncMock()

    # Make it work as async context manager
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock(return_value=None)

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
    """Create a sample portfolio for testing."""
    portfolio = Portfolio(
        id=uuid4(),
        name="Test Portfolio",
        initial_capital=Money(Decimal("100000")),
        cash_balance=Money(Decimal("50000")),
    )
    portfolio.total_realized_pnl = Money(Decimal("5000"))
    portfolio.max_position_size = Money(Decimal("10000"))
    portfolio.max_positions = 10
    portfolio.max_leverage = Decimal("2.0")
    portfolio.max_portfolio_risk = Decimal("0.2")

    # Mock method - get_total_value is NOT async, returns Money
    portfolio.get_total_value = Mock(return_value=Money(Decimal("155000")))

    # Add some positions
    position1 = Position(
        symbol="AAPL",
        quantity=Quantity(Decimal("100")),
        average_entry_price=Price(Decimal("150.00")),
    )
    position1.current_price = Price(Decimal("155.00"))
    position1.opened_at = datetime.now(UTC)

    position2 = Position(
        symbol="GOOGL",
        quantity=Quantity(Decimal("50")),
        average_entry_price=Price(Decimal("2800.00")),
    )
    position2.current_price = Price(Decimal("2850.00"))
    position2.opened_at = datetime.now(UTC)

    # Closed position - create with quantity first, then set to zero after closing
    position3 = Position(
        symbol="MSFT",
        quantity=Quantity(Decimal("50")),
        average_entry_price=Price(Decimal("300.00")),
    )
    position3.closed_at = datetime.now(UTC)
    position3.quantity = Quantity(Decimal("0"))  # Set to zero after marking as closed
    position3.realized_pnl = Money(Decimal("500"))

    portfolio.positions = {
        position1.id: position1,
        position2.id: position2,
        position3.id: position3,
    }

    return portfolio


@pytest.fixture
def sample_position():
    """Create a sample position for testing."""
    position = Position(
        symbol="AAPL",
        quantity=Quantity(Decimal("100")),
        average_entry_price=Price(Decimal("150.00")),
    )
    position.current_price = Price(Decimal("155.00"))
    position.opened_at = datetime.now(UTC)
    return position


class TestGetPortfolioUseCase:
    """Test GetPortfolioUseCase."""

    @pytest.mark.asyncio
    async def test_get_portfolio_with_all_data(
        self, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test getting portfolio with positions and metrics."""
        # Setup
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        use_case = GetPortfolioUseCase(
            unit_of_work=mock_unit_of_work, risk_calculator=mock_risk_calculator
        )

        request = GetPortfolioRequest(
            portfolio_id=sample_portfolio.id, include_positions=True, include_metrics=True
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.portfolio is not None
        assert response.positions is not None
        assert response.metrics is not None

        # Check portfolio data
        portfolio_data = response.portfolio
        assert portfolio_data["id"] == str(sample_portfolio.id)
        assert portfolio_data["name"] == "Test Portfolio"
        assert portfolio_data["cash_balance"] == 50000.0
        assert portfolio_data["total_value"] == 155000.0
        assert portfolio_data["open_positions_count"] == 2

        # Check positions data
        assert len(response.positions) == 2  # Only open positions
        position_symbols = [p["symbol"] for p in response.positions]
        assert "AAPL" in position_symbols
        assert "GOOGL" in position_symbols
        assert "MSFT" not in position_symbols  # Closed position excluded

        # Check metrics
        assert response.metrics["sharpe_ratio"] == 1.5

    @pytest.mark.asyncio
    async def test_get_portfolio_without_positions(
        self, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test getting portfolio without positions data."""
        # Setup
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        use_case = GetPortfolioUseCase(
            unit_of_work=mock_unit_of_work, risk_calculator=mock_risk_calculator
        )

        request = GetPortfolioRequest(
            portfolio_id=sample_portfolio.id, include_positions=False, include_metrics=True
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.portfolio is not None
        assert response.positions is None
        assert response.metrics is not None

    @pytest.mark.asyncio
    async def test_get_portfolio_without_metrics(
        self, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test getting portfolio without metrics."""
        # Setup
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        use_case = GetPortfolioUseCase(
            unit_of_work=mock_unit_of_work, risk_calculator=mock_risk_calculator
        )

        request = GetPortfolioRequest(
            portfolio_id=sample_portfolio.id, include_positions=True, include_metrics=False
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.portfolio is not None
        assert response.positions is not None
        assert response.metrics is None

        # Risk calculator should not be called
        mock_risk_calculator.calculate_sharpe_ratio.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_portfolio_not_found(self, mock_unit_of_work, mock_risk_calculator):
        """Test getting non-existent portfolio."""
        # Setup
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = None

        use_case = GetPortfolioUseCase(
            unit_of_work=mock_unit_of_work, risk_calculator=mock_risk_calculator
        )

        request = GetPortfolioRequest(portfolio_id=uuid4())

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Portfolio not found" in response.error
        assert response.portfolio is None
        assert response.positions is None
        assert response.metrics is None

    @pytest.mark.asyncio
    async def test_get_portfolio_position_data_formatting(
        self, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test position data formatting in response."""
        # Setup
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        use_case = GetPortfolioUseCase(
            unit_of_work=mock_unit_of_work, risk_calculator=mock_risk_calculator
        )

        request = GetPortfolioRequest(
            portfolio_id=sample_portfolio.id, include_positions=True, include_metrics=False
        )

        # Execute
        response = await use_case.execute(request)

        # Assert position data structure
        assert response.success is True
        position = response.positions[0]

        assert "id" in position
        assert "symbol" in position
        assert "side" in position
        assert "quantity" in position
        assert "entry_price" in position
        assert "current_price" in position
        assert "unrealized_pnl" in position
        assert "return_pct" in position
        assert "value" in position

        # Check side formatting
        aapl_position = next(p for p in response.positions if p["symbol"] == "AAPL")
        assert aapl_position["side"] == "long"  # Positive quantity
        assert aapl_position["quantity"] == 100  # Absolute value


class TestUpdatePortfolioUseCase:
    """Test UpdatePortfolioUseCase."""

    @pytest.mark.asyncio
    async def test_update_portfolio_all_fields(self, mock_unit_of_work, sample_portfolio):
        """Test updating all portfolio fields."""
        # Setup
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        use_case = UpdatePortfolioUseCase(unit_of_work=mock_unit_of_work)

        request = UpdatePortfolioRequest(
            portfolio_id=sample_portfolio.id,
            name="Updated Portfolio",
            max_position_size=Decimal("20000"),
            max_positions=20,
            max_leverage=Decimal("3.0"),
            max_portfolio_risk=Decimal("0.3"),
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.updated is True

        # Verify updates
        assert sample_portfolio.name == "Updated Portfolio"
        assert sample_portfolio.max_position_size == Money(Decimal("20000"))
        assert sample_portfolio.max_positions == 20
        assert sample_portfolio.max_leverage == Decimal("3.0")
        assert sample_portfolio.max_portfolio_risk == Decimal("0.3")

        mock_unit_of_work.portfolios.update_portfolio.assert_called_once_with(sample_portfolio)
        mock_unit_of_work.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_portfolio_partial_fields(self, mock_unit_of_work, sample_portfolio):
        """Test updating only some portfolio fields."""
        # Setup
        original_name = sample_portfolio.name
        original_max_positions = sample_portfolio.max_positions
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        use_case = UpdatePortfolioUseCase(unit_of_work=mock_unit_of_work)

        request = UpdatePortfolioRequest(
            portfolio_id=sample_portfolio.id,
            max_position_size=Decimal("15000"),
            max_leverage=Decimal("2.5"),
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.updated is True

        # Verify only specified fields were updated
        assert sample_portfolio.name == original_name
        assert sample_portfolio.max_position_size == Money(Decimal("15000"))
        assert sample_portfolio.max_positions == original_max_positions
        assert sample_portfolio.max_leverage == Decimal("2.5")

    @pytest.mark.asyncio
    async def test_update_portfolio_not_found(self, mock_unit_of_work):
        """Test updating non-existent portfolio."""
        # Setup
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = None

        use_case = UpdatePortfolioUseCase(unit_of_work=mock_unit_of_work)

        request = UpdatePortfolioRequest(portfolio_id=uuid4(), name="New Name")

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Portfolio not found" in response.error
        assert response.updated is False
        mock_unit_of_work.portfolios.update_portfolio.assert_not_called()

    @pytest.mark.asyncio
    async def test_validate_negative_max_position_size(self, mock_unit_of_work):
        """Test validation with negative max position size."""
        use_case = UpdatePortfolioUseCase(unit_of_work=mock_unit_of_work)

        request = UpdatePortfolioRequest(portfolio_id=uuid4(), max_position_size=Decimal("-1000"))

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Max position size must be positive" in response.error

    @pytest.mark.asyncio
    async def test_validate_negative_max_positions(self, mock_unit_of_work):
        """Test validation with negative max positions."""
        use_case = UpdatePortfolioUseCase(unit_of_work=mock_unit_of_work)

        request = UpdatePortfolioRequest(portfolio_id=uuid4(), max_positions=-5)

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Max positions must be positive" in response.error

    @pytest.mark.asyncio
    async def test_validate_invalid_max_leverage(self, mock_unit_of_work):
        """Test validation with invalid max leverage."""
        use_case = UpdatePortfolioUseCase(unit_of_work=mock_unit_of_work)

        request = UpdatePortfolioRequest(portfolio_id=uuid4(), max_leverage=Decimal("0.5"))

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Max leverage must be at least 1.0" in response.error

    @pytest.mark.asyncio
    async def test_validate_invalid_max_portfolio_risk(self, mock_unit_of_work):
        """Test validation with invalid max portfolio risk."""
        use_case = UpdatePortfolioUseCase(unit_of_work=mock_unit_of_work)

        # Test risk > 1
        request = UpdatePortfolioRequest(portfolio_id=uuid4(), max_portfolio_risk=Decimal("1.5"))

        response = await use_case.execute(request)
        assert response.success is False
        assert "Max portfolio risk must be between 0 and 1" in response.error

        # Test risk <= 0
        request = UpdatePortfolioRequest(portfolio_id=uuid4(), max_portfolio_risk=Decimal("0"))

        response = await use_case.execute(request)
        assert response.success is False
        assert "Max portfolio risk must be between 0 and 1" in response.error


class TestGetPositionsUseCase:
    """Test GetPositionsUseCase."""

    @pytest.mark.asyncio
    async def test_get_open_positions(self, mock_unit_of_work, sample_portfolio):
        """Test getting only open positions."""
        # Setup
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        use_case = GetPositionsUseCase(unit_of_work=mock_unit_of_work)

        request = GetPositionsRequest(portfolio_id=sample_portfolio.id, only_open=True)

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert len(response.positions) == 2  # Only AAPL and GOOGL

        symbols = [p["symbol"] for p in response.positions]
        assert "AAPL" in symbols
        assert "GOOGL" in symbols
        assert "MSFT" not in symbols

        # Check all positions are open
        for position in response.positions:
            assert position["is_open"] is True

    @pytest.mark.asyncio
    async def test_get_all_positions(self, mock_unit_of_work, sample_portfolio):
        """Test getting all positions including closed."""
        # Setup
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        use_case = GetPositionsUseCase(unit_of_work=mock_unit_of_work)

        request = GetPositionsRequest(portfolio_id=sample_portfolio.id, only_open=False)

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert len(response.positions) == 3  # All positions

        symbols = [p["symbol"] for p in response.positions]
        assert "AAPL" in symbols
        assert "GOOGL" in symbols
        assert "MSFT" in symbols

        # Check closed position
        msft_position = next(p for p in response.positions if p["symbol"] == "MSFT")
        assert msft_position["is_open"] is False
        assert msft_position["closed_at"] is not None

    @pytest.mark.asyncio
    async def test_get_positions_by_symbol(self, mock_unit_of_work, sample_portfolio):
        """Test filtering positions by symbol."""
        # Setup
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        use_case = GetPositionsUseCase(unit_of_work=mock_unit_of_work)

        request = GetPositionsRequest(
            portfolio_id=sample_portfolio.id, only_open=False, symbol="AAPL"
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert len(response.positions) == 1
        assert response.positions[0]["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_get_positions_portfolio_not_found(self, mock_unit_of_work):
        """Test getting positions for non-existent portfolio."""
        # Setup
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = None

        use_case = GetPositionsUseCase(unit_of_work=mock_unit_of_work)

        request = GetPositionsRequest(portfolio_id=uuid4())

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Portfolio not found" in response.error
        assert response.positions == []

    @pytest.mark.asyncio
    async def test_get_positions_total_value_calculation(self, mock_unit_of_work, sample_portfolio):
        """Test total value calculation for positions."""
        # Setup
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        use_case = GetPositionsUseCase(unit_of_work=mock_unit_of_work)

        request = GetPositionsRequest(portfolio_id=sample_portfolio.id, only_open=True)

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.total_value is not None

        # Calculate expected total
        # AAPL: 100 * 155 = 15,500
        # GOOGL: 50 * 2850 = 142,500
        # Total: 158,000
        expected_total = Decimal("158000")
        assert abs(response.total_value - expected_total) < Decimal("1")

    @pytest.mark.asyncio
    async def test_get_positions_data_formatting(self, mock_unit_of_work, sample_portfolio):
        """Test position data formatting in response."""
        # Setup
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        use_case = GetPositionsUseCase(unit_of_work=mock_unit_of_work)

        request = GetPositionsRequest(portfolio_id=sample_portfolio.id, only_open=True)

        # Execute
        response = await use_case.execute(request)

        # Assert data structure
        position = response.positions[0]

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


class TestClosePositionUseCase:
    """Test ClosePositionUseCase."""

    @pytest.mark.asyncio
    async def test_close_position_success(
        self, mock_unit_of_work, mock_position_manager, sample_position
    ):
        """Test successful position closure."""
        # Setup
        sample_position.realized_pnl = Money(Decimal("500"))
        mock_unit_of_work.positions.get_position_by_id.return_value = sample_position

        use_case = ClosePositionUseCase(
            unit_of_work=mock_unit_of_work, position_manager=mock_position_manager
        )

        request = ClosePositionRequest(
            position_id=sample_position.id, exit_price=Decimal("160.00"), reason="Take profit"
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.closed is True
        assert response.realized_pnl == Decimal("500")
        assert response.total_return is not None

        # Verify interactions
        mock_position_manager.close_position.assert_called_once()
        mock_unit_of_work.positions.update_position.assert_called_once_with(sample_position)
        mock_unit_of_work.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_position_not_found(self, mock_unit_of_work, mock_position_manager):
        """Test closing non-existent position."""
        # Setup
        mock_unit_of_work.positions.get_position_by_id.return_value = None

        use_case = ClosePositionUseCase(
            unit_of_work=mock_unit_of_work, position_manager=mock_position_manager
        )

        request = ClosePositionRequest(position_id=uuid4(), exit_price=Decimal("160.00"))

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Position not found" in response.error
        assert response.closed is False
        mock_position_manager.close_position.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_already_closed_position(
        self, mock_unit_of_work, mock_position_manager, sample_position
    ):
        """Test closing an already closed position."""
        # Setup
        sample_position.closed_at = datetime.now(UTC)
        sample_position.quantity = Quantity(Decimal("0"))
        mock_unit_of_work.positions.get_position_by_id.return_value = sample_position

        use_case = ClosePositionUseCase(
            unit_of_work=mock_unit_of_work, position_manager=mock_position_manager
        )

        request = ClosePositionRequest(position_id=sample_position.id, exit_price=Decimal("160.00"))

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Position is already closed" in response.error
        assert response.closed is False
        mock_position_manager.close_position.assert_not_called()

    @pytest.mark.asyncio
    async def test_validate_negative_exit_price(self, mock_unit_of_work, mock_position_manager):
        """Test validation with negative exit price."""
        use_case = ClosePositionUseCase(
            unit_of_work=mock_unit_of_work, position_manager=mock_position_manager
        )

        request = ClosePositionRequest(position_id=uuid4(), exit_price=Decimal("-10.00"))

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Exit price must be positive" in response.error

    @pytest.mark.asyncio
    async def test_close_position_with_reason(
        self, mock_unit_of_work, mock_position_manager, sample_position
    ):
        """Test closing position with reason."""
        # Setup
        mock_unit_of_work.positions.get_position_by_id.return_value = sample_position

        use_case = ClosePositionUseCase(
            unit_of_work=mock_unit_of_work, position_manager=mock_position_manager
        )

        request = ClosePositionRequest(
            position_id=sample_position.id,
            exit_price=Decimal("145.00"),
            reason="Stop loss triggered",
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.closed is True

        # Verify closing order was created correctly
        call_args = mock_position_manager.close_position.call_args
        closing_order = call_args[1]["order"]
        assert closing_order.side == OrderSide.SELL  # Long position, so sell to close
        assert closing_order.quantity == abs(sample_position.quantity)
        assert closing_order.average_fill_price == Price(Decimal("145.00"))


class TestPortfolioRequestResponseDTOs:
    """Test portfolio management request and response DTOs."""

    def test_get_portfolio_request_defaults(self):
        """Test GetPortfolioRequest default values."""
        portfolio_id = uuid4()
        request = GetPortfolioRequest(portfolio_id=portfolio_id)

        assert request.portfolio_id == portfolio_id
        assert request.include_positions is True
        assert request.include_metrics is True
        assert request.request_id is not None
        assert request.metadata == {}

    def test_update_portfolio_request_partial(self):
        """Test UpdatePortfolioRequest with partial updates."""
        portfolio_id = uuid4()
        request = UpdatePortfolioRequest(portfolio_id=portfolio_id, name="New Name")

        assert request.portfolio_id == portfolio_id
        assert request.name == "New Name"
        assert request.max_position_size is None
        assert request.max_positions is None
        assert request.max_leverage is None
        assert request.max_portfolio_risk is None

    def test_get_positions_request_filtering(self):
        """Test GetPositionsRequest with filtering options."""
        portfolio_id = uuid4()
        request = GetPositionsRequest(portfolio_id=portfolio_id, only_open=False, symbol="AAPL")

        assert request.portfolio_id == portfolio_id
        assert request.only_open is False
        assert request.symbol == "AAPL"

    def test_close_position_request_with_metadata(self):
        """Test ClosePositionRequest with metadata."""
        position_id = uuid4()
        request = ClosePositionRequest(
            position_id=position_id,
            exit_price=Decimal("150.00"),
            reason="Market exit",
            metadata={"strategy": "momentum"},
        )

        assert request.position_id == position_id
        assert request.exit_price == Decimal("150.00")
        assert request.reason == "Market exit"
        assert request.metadata == {"strategy": "momentum"}

    def test_portfolio_response_creation(self):
        """Test portfolio response creation."""
        response = GetPortfolioResponse(
            success=True,
            portfolio={"id": "123", "name": "Test"},
            positions=[{"symbol": "AAPL"}],
            metrics={"sharpe_ratio": 1.5},
            request_id=uuid4(),
        )

        assert response.success is True
        assert response.portfolio["name"] == "Test"
        assert len(response.positions) == 1
        assert response.metrics["sharpe_ratio"] == 1.5
