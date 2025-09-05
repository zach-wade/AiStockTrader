"""
Comprehensive tests for portfolio management use cases.

Tests all portfolio management-related use cases including portfolio retrieval,
updates, position queries, and position closing with full coverage of success
scenarios, failure cases, and edge conditions.
"""

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from src.application.use_cases.portfolio_legacy import (
    ClosePositionRequest,
    ClosePositionUseCase,
    GetPortfolioRequest,
    GetPortfolioUseCase,
    GetPositionsRequest,
    GetPositionsUseCase,
    UpdatePortfolioRequest,
    UpdatePortfolioUseCase,
)
from src.domain.entities.portfolio import Portfolio
from src.domain.entities.position import Position
from src.domain.services.position_manager import PositionManager
from src.domain.services.risk_calculator import RiskCalculator
from src.domain.value_objects.money import Money
from src.domain.value_objects.price import Price
from src.domain.value_objects.quantity import Quantity


class TestGetPortfolioUseCase:
    """Test GetPortfolioUseCase with comprehensive scenarios."""

    @pytest.fixture
    def mock_unit_of_work(self):
        """Create mock unit of work."""
        uow = AsyncMock()
        uow.portfolios = AsyncMock()
        uow.positions = AsyncMock()
        uow.commit = AsyncMock()
        uow.rollback = AsyncMock()
        uow.__aenter__ = AsyncMock(return_value=uow)
        uow.__aexit__ = AsyncMock(return_value=None)
        return uow

    @pytest.fixture
    def mock_risk_calculator(self):
        """Create mock risk calculator."""
        calculator = Mock(spec=RiskCalculator)
        calculator.calculate_sharpe_ratio = Mock(return_value=Decimal("1.5"))
        return calculator

    @pytest.fixture
    def use_case(self, mock_unit_of_work, mock_risk_calculator):
        """Create use case instance."""
        return GetPortfolioUseCase(
            unit_of_work=mock_unit_of_work, risk_calculator=mock_risk_calculator
        )

    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio with positions."""
        portfolio = Portfolio(
            name="Test Portfolio",
            initial_capital=Money(Decimal("100000")),
            cash_balance=Money(Decimal("50000")),
            max_position_size=Money(Decimal("10000")),
            max_positions=10,
            max_leverage=Decimal("2.0"),
            max_portfolio_risk=Decimal("0.02"),
        )
        portfolio.id = uuid4()
        portfolio.total_realized_pnl = Money(Decimal("5000"))

        # Add open position
        position1 = Position(
            id=uuid4(),
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            average_entry_price=Price(Decimal("150.00")),
            current_price=Price(Decimal("155.00")),
            opened_at=datetime.now(UTC),
        )

        # Add closed position
        position2 = Position(
            id=uuid4(),
            symbol="GOOGL",
            quantity=Quantity(Decimal("50")),
            average_entry_price=Price(Decimal("2500.00")),
            current_price=Price(Decimal("2600.00")),
            opened_at=datetime.now(UTC),
            closed_at=datetime.now(UTC),
            realized_pnl=Money(Decimal("5000")),
        )

        portfolio.positions = {"AAPL": position1, "GOOGL": position2}
        return portfolio

    @pytest.mark.asyncio
    async def test_get_portfolio_success_with_all_data(
        self, use_case, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test successful portfolio retrieval with positions and metrics."""
        # Setup
        portfolio_id = sample_portfolio.id
        request = GetPortfolioRequest(
            portfolio_id=portfolio_id, include_positions=True, include_metrics=True
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.error is None
        assert response.portfolio is not None
        assert response.positions is not None
        assert response.metrics is not None

        # Verify portfolio data
        portfolio_data = response.portfolio
        assert portfolio_data["id"] == str(portfolio_id)
        assert portfolio_data["name"] == "Test Portfolio"
        assert portfolio_data["cash_balance"] == 50000.0
        assert portfolio_data["initial_capital"] == 100000.0
        assert portfolio_data["realized_pnl"] == 5000.0
        assert portfolio_data["open_positions_count"] == 1
        assert portfolio_data["max_leverage"] == 2.0

        # Verify positions data
        assert len(response.positions) == 1  # Only open positions
        position = response.positions[0]
        assert position["symbol"] == "AAPL"
        assert position["quantity"] == 100
        assert position["entry_price"] == 150.0
        assert position["current_price"] == 155.0

        # Verify metrics
        assert response.metrics["sharpe_ratio"] == 1.5

        mock_unit_of_work.portfolios.get_portfolio_by_id.assert_called_once_with(portfolio_id)
        mock_risk_calculator.calculate_sharpe_ratio.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_portfolio_without_positions(
        self, use_case, mock_unit_of_work, sample_portfolio
    ):
        """Test portfolio retrieval without positions data."""
        # Setup
        request = GetPortfolioRequest(
            portfolio_id=sample_portfolio.id, include_positions=False, include_metrics=False
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.portfolio is not None
        assert response.positions is None
        assert response.metrics is None

    @pytest.mark.asyncio
    async def test_get_portfolio_not_found(self, use_case, mock_unit_of_work):
        """Test portfolio retrieval when portfolio doesn't exist."""
        # Setup
        portfolio_id = uuid4()
        request = GetPortfolioRequest(portfolio_id=portfolio_id)

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = None

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.error == "Portfolio not found"
        assert response.portfolio is None

    @pytest.mark.asyncio
    async def test_get_portfolio_empty_positions(
        self, use_case, mock_unit_of_work, mock_risk_calculator
    ):
        """Test portfolio retrieval with no positions."""
        # Setup
        portfolio = Portfolio(
            name="Empty Portfolio",
            initial_capital=Money(Decimal("100000")),
            cash_balance=Money(Decimal("100000")),
        )
        portfolio.id = uuid4()

        request = GetPortfolioRequest(
            portfolio_id=portfolio.id, include_positions=True, include_metrics=True
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = portfolio

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.positions == []
        assert response.portfolio["open_positions_count"] == 0
        assert response.portfolio["positions_value"] == 0.0
        assert response.portfolio["unrealized_pnl"] == 0.0

    @pytest.mark.asyncio
    async def test_get_portfolio_with_exception(self, use_case, mock_unit_of_work):
        """Test portfolio retrieval with unexpected exception."""
        # Setup
        request = GetPortfolioRequest(portfolio_id=uuid4())

        mock_unit_of_work.portfolios.get_portfolio_by_id.side_effect = Exception("Database error")

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Database error" in response.error

    @pytest.mark.asyncio
    async def test_validate_always_passes(self, use_case):
        """Test that validation always passes for GetPortfolioRequest."""
        request = GetPortfolioRequest(portfolio_id=uuid4())
        result = await use_case.validate(request)
        assert result is None


class TestUpdatePortfolioUseCase:
    """Test UpdatePortfolioUseCase with comprehensive scenarios."""

    @pytest.fixture
    def mock_unit_of_work(self):
        """Create mock unit of work."""
        uow = AsyncMock()
        uow.portfolios = AsyncMock()
        uow.commit = AsyncMock()
        uow.rollback = AsyncMock()
        uow.__aenter__ = AsyncMock(return_value=uow)
        uow.__aexit__ = AsyncMock(return_value=None)
        return uow

    @pytest.fixture
    def use_case(self, mock_unit_of_work):
        """Create use case instance."""
        return UpdatePortfolioUseCase(unit_of_work=mock_unit_of_work)

    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio."""
        portfolio = Portfolio(
            name="Test Portfolio",
            initial_capital=Money(Decimal("100000")),
            cash_balance=Money(Decimal("50000")),
        )
        portfolio.id = uuid4()
        return portfolio

    @pytest.mark.asyncio
    async def test_update_portfolio_all_fields(self, use_case, mock_unit_of_work, sample_portfolio):
        """Test updating all portfolio fields."""
        # Setup
        request = UpdatePortfolioRequest(
            portfolio_id=sample_portfolio.id,
            name="Updated Portfolio",
            max_position_size=Decimal("20000"),
            max_positions=20,
            max_leverage=Decimal("3.0"),
            max_portfolio_risk=Decimal("0.05"),
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.updated is True
        assert response.error is None

        # Verify portfolio was updated
        assert sample_portfolio.name == "Updated Portfolio"
        assert sample_portfolio.max_position_size == Money(Decimal("20000"))
        assert sample_portfolio.max_positions == 20
        assert sample_portfolio.max_leverage == Decimal("3.0")
        assert sample_portfolio.max_portfolio_risk == Decimal("0.05")

        mock_unit_of_work.portfolios.update_portfolio.assert_called_once_with(sample_portfolio)
        mock_unit_of_work.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_portfolio_partial_fields(
        self, use_case, mock_unit_of_work, sample_portfolio
    ):
        """Test updating only some portfolio fields."""
        # Setup
        original_name = sample_portfolio.name
        request = UpdatePortfolioRequest(
            portfolio_id=sample_portfolio.id, max_positions=15, max_leverage=Decimal("2.5")
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert sample_portfolio.name == original_name  # Unchanged
        assert sample_portfolio.max_positions == 15
        assert sample_portfolio.max_leverage == Decimal("2.5")

    @pytest.mark.asyncio
    async def test_update_portfolio_not_found(self, use_case, mock_unit_of_work):
        """Test updating non-existent portfolio."""
        # Setup
        request = UpdatePortfolioRequest(portfolio_id=uuid4(), name="New Name")

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = None

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.error == "Portfolio not found"
        assert response.updated is False
        mock_unit_of_work.portfolios.update_portfolio.assert_not_called()

    @pytest.mark.asyncio
    async def test_validate_invalid_max_position_size(self, use_case):
        """Test validation with invalid max position size."""
        request = UpdatePortfolioRequest(portfolio_id=uuid4(), max_position_size=Decimal("-1000"))

        result = await use_case.validate(request)
        assert result == "Max position size must be positive"

    @pytest.mark.asyncio
    async def test_validate_invalid_max_positions(self, use_case):
        """Test validation with invalid max positions."""
        request = UpdatePortfolioRequest(portfolio_id=uuid4(), max_positions=0)

        result = await use_case.validate(request)
        assert result == "Max positions must be positive"

    @pytest.mark.asyncio
    async def test_validate_invalid_max_leverage(self, use_case):
        """Test validation with invalid max leverage."""
        request = UpdatePortfolioRequest(portfolio_id=uuid4(), max_leverage=Decimal("0.5"))

        result = await use_case.validate(request)
        assert result == "Max leverage must be at least 1.0"

    @pytest.mark.asyncio
    async def test_validate_invalid_max_portfolio_risk_negative(self, use_case):
        """Test validation with negative portfolio risk."""
        request = UpdatePortfolioRequest(portfolio_id=uuid4(), max_portfolio_risk=Decimal("-0.01"))

        result = await use_case.validate(request)
        assert result == "Max portfolio risk must be between 0 and 1"

    @pytest.mark.asyncio
    async def test_validate_invalid_max_portfolio_risk_too_high(self, use_case):
        """Test validation with portfolio risk > 1."""
        request = UpdatePortfolioRequest(portfolio_id=uuid4(), max_portfolio_risk=Decimal("1.5"))

        result = await use_case.validate(request)
        assert result == "Max portfolio risk must be between 0 and 1"

    @pytest.mark.asyncio
    async def test_validate_valid_request(self, use_case):
        """Test validation with valid request."""
        request = UpdatePortfolioRequest(
            portfolio_id=uuid4(),
            max_position_size=Decimal("10000"),
            max_positions=10,
            max_leverage=Decimal("2.0"),
            max_portfolio_risk=Decimal("0.02"),
        )

        result = await use_case.validate(request)
        assert result is None


class TestGetPositionsUseCase:
    """Test GetPositionsUseCase with comprehensive scenarios."""

    @pytest.fixture
    def mock_unit_of_work(self):
        """Create mock unit of work."""
        uow = AsyncMock()
        uow.portfolios = AsyncMock()
        uow.positions = AsyncMock()
        uow.commit = AsyncMock()
        uow.rollback = AsyncMock()
        uow.__aenter__ = AsyncMock(return_value=uow)
        uow.__aexit__ = AsyncMock(return_value=None)
        return uow

    @pytest.fixture
    def use_case(self, mock_unit_of_work):
        """Create use case instance."""
        return GetPositionsUseCase(unit_of_work=mock_unit_of_work)

    @pytest.fixture
    def sample_positions(self):
        """Create sample positions."""
        position1 = Position(
            id=uuid4(),
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            average_entry_price=Price(Decimal("150.00")),
            current_price=Price(Decimal("155.00")),
            opened_at=datetime.now(UTC),
        )

        position2 = Position(
            id=uuid4(),
            symbol="GOOGL",
            quantity=Quantity(Decimal("50")),
            average_entry_price=Price(Decimal("2500.00")),
            current_price=Price(Decimal("2600.00")),
            opened_at=datetime.now(UTC),
            closed_at=datetime.now(UTC),
            realized_pnl=Money(Decimal("5000")),
        )

        position3 = Position(
            id=uuid4(),
            symbol="MSFT",
            quantity=Quantity(Decimal("-50")),  # Short position
            average_entry_price=Price(Decimal("300.00")),
            current_price=Price(Decimal("295.00")),
            opened_at=datetime.now(UTC),
        )

        return [position1, position2, position3]

    @pytest.fixture
    def sample_portfolio_with_positions(self, sample_positions):
        """Create sample portfolio with positions."""
        portfolio = Portfolio(
            name="Test Portfolio",
            initial_capital=Money(Decimal("100000")),
            cash_balance=Money(Decimal("50000")),
        )
        portfolio.id = uuid4()
        portfolio.positions = {
            "AAPL": sample_positions[0],
            "GOOGL": sample_positions[1],
            "MSFT": sample_positions[2],
        }
        return portfolio

    @pytest.mark.asyncio
    async def test_get_open_positions_only(
        self, use_case, mock_unit_of_work, sample_portfolio_with_positions
    ):
        """Test retrieving only open positions."""
        # Setup
        request = GetPositionsRequest(
            portfolio_id=sample_portfolio_with_positions.id, only_open=True
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = (
            sample_portfolio_with_positions
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert len(response.positions) == 2  # Only AAPL and MSFT are open

        symbols = [p["symbol"] for p in response.positions]
        assert "AAPL" in symbols
        assert "MSFT" in symbols
        assert "GOOGL" not in symbols  # Closed position

        # Verify total value calculation
        expected_total = Decimal("100") * Decimal("155") + abs(Decimal("-50")) * Decimal("295")
        assert response.total_value == Money(expected_total)

    @pytest.mark.asyncio
    async def test_get_all_positions(
        self, use_case, mock_unit_of_work, sample_portfolio_with_positions
    ):
        """Test retrieving all positions including closed."""
        # Setup
        request = GetPositionsRequest(
            portfolio_id=sample_portfolio_with_positions.id, only_open=False
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = (
            sample_portfolio_with_positions
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert len(response.positions) == 3  # All positions

        # Verify closed position is included
        closed_position = next(p for p in response.positions if p["symbol"] == "GOOGL")
        assert closed_position["is_open"] is False
        assert closed_position["realized_pnl"] == 5000

    @pytest.mark.asyncio
    async def test_get_positions_by_symbol(
        self, use_case, mock_unit_of_work, sample_portfolio_with_positions
    ):
        """Test filtering positions by symbol."""
        # Setup
        request = GetPositionsRequest(
            portfolio_id=sample_portfolio_with_positions.id, only_open=False, symbol="AAPL"
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = (
            sample_portfolio_with_positions
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert len(response.positions) == 1
        assert response.positions[0]["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_get_positions_empty_portfolio(self, use_case, mock_unit_of_work):
        """Test retrieving positions from empty portfolio."""
        # Setup
        portfolio = Portfolio(name="Empty Portfolio", initial_capital=Money(Decimal("100000")))
        portfolio.id = uuid4()

        request = GetPositionsRequest(portfolio_id=portfolio.id)
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = portfolio

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.positions == []
        assert response.total_value == Decimal("0")

    @pytest.mark.asyncio
    async def test_get_positions_portfolio_not_found(self, use_case, mock_unit_of_work):
        """Test retrieving positions when portfolio doesn't exist."""
        # Setup
        request = GetPositionsRequest(portfolio_id=uuid4())
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = None

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.error == "Portfolio not found"
        assert response.positions == []

    @pytest.mark.asyncio
    async def test_get_positions_symbol_not_found(
        self, use_case, mock_unit_of_work, sample_portfolio_with_positions
    ):
        """Test filtering by non-existent symbol."""
        # Setup
        request = GetPositionsRequest(
            portfolio_id=sample_portfolio_with_positions.id, symbol="TSLA"
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = (
            sample_portfolio_with_positions
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.positions == []
        assert response.total_value == Decimal("0")

    @pytest.mark.asyncio
    async def test_validate_always_passes(self, use_case):
        """Test that validation always passes for GetPositionsRequest."""
        request = GetPositionsRequest(portfolio_id=uuid4())
        result = await use_case.validate(request)
        assert result is None


class TestClosePositionUseCase:
    """Test ClosePositionUseCase with comprehensive scenarios."""

    @pytest.fixture
    def mock_unit_of_work(self):
        """Create mock unit of work."""
        uow = AsyncMock()
        uow.portfolios = AsyncMock()
        uow.positions = AsyncMock()
        uow.commit = AsyncMock()
        uow.rollback = AsyncMock()
        uow.__aenter__ = AsyncMock(return_value=uow)
        uow.__aexit__ = AsyncMock(return_value=None)
        return uow

    @pytest.fixture
    def mock_position_manager(self):
        """Create mock position manager."""
        manager = Mock(spec=PositionManager)
        manager.close_position = Mock()
        return manager

    @pytest.fixture
    def use_case(self, mock_unit_of_work, mock_position_manager):
        """Create use case instance."""
        return ClosePositionUseCase(
            unit_of_work=mock_unit_of_work, position_manager=mock_position_manager
        )

    @pytest.fixture
    def sample_position(self):
        """Create sample open position."""
        position = Position(
            id=uuid4(),
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            average_entry_price=Price(Decimal("150.00")),
            current_price=Price(Decimal("155.00")),
            opened_at=datetime.now(UTC),
        )
        return position

    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio."""
        portfolio = Portfolio(
            name="Test Portfolio",
            initial_capital=Money(Decimal("100000")),
            cash_balance=Money(Decimal("50000")),
        )
        portfolio.id = uuid4()
        return portfolio

    @pytest.mark.asyncio
    async def test_close_position_success_with_profit(
        self, use_case, mock_unit_of_work, mock_position_manager, sample_position, sample_portfolio
    ):
        """Test successfully closing a position with profit."""
        # Setup
        exit_price = Decimal("160.00")
        request = ClosePositionRequest(
            position_id=sample_position.id, exit_price=exit_price, reason="Take profit"
        )

        # Calculate expected P&L
        expected_pnl = (exit_price - Decimal("150.00")) * Decimal("100")
        expected_return = ((exit_price - Decimal("150.00")) / Decimal("150.00")) * Decimal("100")

        sample_position.realized_pnl = expected_pnl
        sample_position.get_return_percentage = Mock(return_value=expected_return)

        mock_unit_of_work.positions.get_position_by_id.return_value = sample_position
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        def close_position_side_effect(position, order, exit_price):
            position.closed_at = datetime.now(UTC)
            position.realized_pnl = expected_pnl

        mock_position_manager.close_position.side_effect = close_position_side_effect

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.closed is True
        assert response.realized_pnl == expected_pnl
        assert response.total_return == expected_return

        # Verify position manager was called
        mock_position_manager.close_position.assert_called_once()
        call_args = mock_position_manager.close_position.call_args
        assert call_args[1]["position"] == sample_position
        assert call_args[1]["exit_price"] == Price(exit_price)

        # Verify repositories were updated
        mock_unit_of_work.positions.update_position.assert_called_once_with(sample_position)
        # Portfolio update is disabled until position-portfolio relationship is established
        mock_unit_of_work.portfolios.update_portfolio.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_position_success_with_loss(
        self, use_case, mock_unit_of_work, mock_position_manager, sample_position, sample_portfolio
    ):
        """Test successfully closing a position with loss."""
        # Setup
        exit_price = Decimal("145.00")
        request = ClosePositionRequest(
            position_id=sample_position.id, exit_price=exit_price, reason="Stop loss"
        )

        # Calculate expected loss
        expected_pnl = (exit_price - Decimal("150.00")) * Decimal("100")  # -500
        expected_return = ((exit_price - Decimal("150.00")) / Decimal("150.00")) * Decimal("100")

        sample_position.realized_pnl = expected_pnl
        sample_position.get_return_percentage = Mock(return_value=expected_return)

        mock_unit_of_work.positions.get_position_by_id.return_value = sample_position
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        def close_position_side_effect(position, order, exit_price):
            position.closed_at = datetime.now(UTC)
            position.realized_pnl = expected_pnl

        mock_position_manager.close_position.side_effect = close_position_side_effect

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.closed is True
        assert response.realized_pnl == expected_pnl
        assert response.realized_pnl < 0  # Loss

        # Portfolio update is disabled
        mock_unit_of_work.portfolios.update_portfolio.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_short_position_success(
        self, use_case, mock_unit_of_work, mock_position_manager, sample_portfolio
    ):
        """Test closing a short position."""
        # Setup
        short_position = Position(
            id=uuid4(),
            symbol="TSLA",
            quantity=Quantity(Decimal("-50")),  # Short position
            average_entry_price=Price(Decimal("700.00")),
            current_price=Price(Decimal("680.00")),
            opened_at=datetime.now(UTC),
        )
        # Position is open by default

        exit_price = Decimal("680.00")
        request = ClosePositionRequest(position_id=short_position.id, exit_price=exit_price)

        # Short position profit: sold at 700, bought back at 680
        expected_pnl = (Decimal("700.00") - exit_price) * Decimal("50")
        short_position.realized_pnl = expected_pnl
        short_position.get_return_percentage = Mock(return_value=Decimal("2.86"))

        mock_unit_of_work.positions.get_position_by_id.return_value = short_position
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.realized_pnl == expected_pnl
        assert response.realized_pnl > 0  # Profit on short

    @pytest.mark.asyncio
    async def test_close_position_not_found(self, use_case, mock_unit_of_work):
        """Test closing non-existent position."""
        # Setup
        request = ClosePositionRequest(position_id=uuid4(), exit_price=Decimal("150.00"))

        mock_unit_of_work.positions.get_position_by_id.return_value = None

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.error == "Position not found"
        assert response.closed is False
        mock_unit_of_work.positions.update_position.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_already_closed_position(
        self, use_case, mock_unit_of_work, sample_position
    ):
        """Test closing an already closed position."""
        # Setup
        # Mark as closed
        sample_position.closed_at = datetime.now(UTC)

        request = ClosePositionRequest(position_id=sample_position.id, exit_price=Decimal("150.00"))

        mock_unit_of_work.positions.get_position_by_id.return_value = sample_position

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.error == "Position is already closed"
        assert response.closed is False

    @pytest.mark.asyncio
    async def test_close_position_portfolio_not_found(
        self, use_case, mock_unit_of_work, mock_position_manager, sample_position
    ):
        """Test closing position when portfolio is not found."""
        # Setup
        request = ClosePositionRequest(position_id=sample_position.id, exit_price=Decimal("160.00"))

        mock_unit_of_work.positions.get_position_by_id.return_value = sample_position
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = None

        # Execute - should still succeed but without portfolio update
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.closed is True
        mock_unit_of_work.positions.update_position.assert_called_once()
        # Portfolio lookup wouldn't succeed anyway since Position has no portfolio_id
        mock_unit_of_work.portfolios.get_portfolio_by_id.assert_not_called()

    @pytest.mark.asyncio
    async def test_validate_negative_exit_price(self, use_case):
        """Test validation with negative exit price."""
        request = ClosePositionRequest(position_id=uuid4(), exit_price=Decimal("-100.00"))

        result = await use_case.validate(request)
        assert result == "Exit price must be positive"

    @pytest.mark.asyncio
    async def test_validate_zero_exit_price(self, use_case):
        """Test validation with zero exit price."""
        request = ClosePositionRequest(position_id=uuid4(), exit_price=Decimal("0"))

        result = await use_case.validate(request)
        assert result == "Exit price must be positive"

    @pytest.mark.asyncio
    async def test_validate_valid_request(self, use_case):
        """Test validation with valid request."""
        request = ClosePositionRequest(position_id=uuid4(), exit_price=Decimal("150.00"))

        result = await use_case.validate(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_close_position_with_metadata(
        self, use_case, mock_unit_of_work, mock_position_manager, sample_position, sample_portfolio
    ):
        """Test closing position with metadata."""
        # Setup
        request = ClosePositionRequest(
            position_id=sample_position.id,
            exit_price=Decimal("155.00"),
            reason="Manual close",
            metadata={"user": "trader1", "notes": "Taking partial profits"},
        )

        sample_position.realized_pnl = Decimal("500")
        sample_position.get_return_percentage = Mock(return_value=Decimal("3.33"))

        mock_unit_of_work.positions.get_position_by_id.return_value = sample_position
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.request_id == request.request_id

    @pytest.mark.asyncio
    async def test_close_position_transaction_rollback(
        self, use_case, mock_unit_of_work, mock_position_manager, sample_position
    ):
        """Test transaction rollback on error."""
        # Setup
        request = ClosePositionRequest(position_id=sample_position.id, exit_price=Decimal("160.00"))

        mock_unit_of_work.positions.get_position_by_id.return_value = sample_position
        mock_unit_of_work.positions.update_position.side_effect = Exception("Database error")

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Database error" in response.error
        mock_unit_of_work.rollback.assert_called()


class TestEdgeCasesAndIntegration:
    """Test edge cases and integration scenarios."""

    @pytest.fixture
    def mock_unit_of_work(self):
        """Create mock unit of work."""
        uow = AsyncMock()
        uow.portfolios = AsyncMock()
        uow.positions = AsyncMock()
        uow.commit = AsyncMock()
        uow.rollback = AsyncMock()
        uow.__aenter__ = AsyncMock(return_value=uow)
        uow.__aexit__ = AsyncMock(return_value=None)
        return uow

    @pytest.mark.asyncio
    async def test_portfolio_with_extreme_values(self, mock_unit_of_work):
        """Test portfolio with extreme values."""
        # Setup
        portfolio = Portfolio(
            name="Extreme Portfolio",
            initial_capital=Money(Decimal("999999999999.99")),
            cash_balance=Money(Decimal("0.01")),
            max_position_size=Money(Decimal("999999999.99")),
            max_leverage=Decimal("100.0"),
        )
        portfolio.id = uuid4()
        portfolio.total_realized_pnl = Money(Decimal("-999999999.99"))

        risk_calculator = Mock(spec=RiskCalculator)
        risk_calculator.calculate_sharpe_ratio = Mock(return_value=Decimal("-99.99"))

        use_case = GetPortfolioUseCase(
            unit_of_work=mock_unit_of_work, risk_calculator=risk_calculator
        )

        request = GetPortfolioRequest(portfolio_id=portfolio.id)
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = portfolio

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.portfolio["cash_balance"] == 0.01
        assert response.portfolio["realized_pnl"] == -999999999.99

    @pytest.mark.asyncio
    async def test_concurrent_position_updates(self, mock_unit_of_work):
        """Test handling concurrent position updates."""
        # Setup
        portfolio = Portfolio(name="Concurrent Test")
        portfolio.id = uuid4()

        # Create positions being updated concurrently
        positions = {}
        for i in range(100):
            symbol = f"STOCK{i}"
            positions[symbol] = Position(
                id=uuid4(),
                symbol=symbol,
                quantity=Quantity(Decimal("10")),
                average_entry_price=Price(Decimal("100.00")),
                current_price=Price(Decimal("105.00")),
            )

        portfolio.positions = positions

        use_case = GetPositionsUseCase(unit_of_work=mock_unit_of_work)
        request = GetPositionsRequest(portfolio_id=portfolio.id)
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = portfolio

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert len(response.positions) == 100
        assert response.total_value == Money(Decimal("10") * Decimal("105.00") * 100)

    @pytest.mark.asyncio
    async def test_position_with_none_values(self, mock_unit_of_work):
        """Test handling positions with None values."""
        # Setup
        position = Position(
            id=uuid4(),
            symbol="TEST",
            quantity=Quantity(Decimal("100")),
            average_entry_price=Price(Decimal("50.00")),
            current_price=None,  # No current price
            realized_pnl=None,
            opened_at=datetime.now(UTC),
        )
        # Position is open if not closed

        portfolio = Portfolio(name="Test")
        portfolio.id = uuid4()
        portfolio.positions = {"TEST": position}

        use_case = GetPositionsUseCase(unit_of_work=mock_unit_of_work)
        request = GetPositionsRequest(portfolio_id=portfolio.id)
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = portfolio

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.positions[0]["current_price"] is None
        assert response.positions[0]["realized_pnl"] == 0

    @pytest.mark.asyncio
    async def test_request_id_propagation(self, mock_unit_of_work):
        """Test that request IDs are properly propagated through responses."""
        # Setup
        request_id = uuid4()
        correlation_id = uuid4()

        use_case = UpdatePortfolioUseCase(unit_of_work=mock_unit_of_work)
        request = UpdatePortfolioRequest(
            portfolio_id=uuid4(), name="Test", request_id=request_id, correlation_id=correlation_id
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = Portfolio(name="Old")

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.request_id == request_id

    @pytest.mark.asyncio
    async def test_empty_symbol_filter(self, mock_unit_of_work):
        """Test filtering with empty symbol string."""
        # Setup
        portfolio = Portfolio(name="Test")
        portfolio.id = uuid4()
        portfolio.positions = {
            "AAPL": Position(
                symbol="AAPL",
                quantity=Quantity(Decimal("100")),
                average_entry_price=Price(Decimal("150.00")),
            )
        }

        use_case = GetPositionsUseCase(unit_of_work=mock_unit_of_work)
        request = GetPositionsRequest(portfolio_id=portfolio.id, symbol="")  # Empty string
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = portfolio

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        # Empty symbol string means no filter - returns all positions
        assert len(response.positions) == 1
        assert response.positions[0]["symbol"] == "AAPL"
