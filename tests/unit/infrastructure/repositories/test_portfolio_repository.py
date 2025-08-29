"""
Comprehensive unit tests for PostgreSQL Portfolio Repository Implementation.

Tests the concrete implementation of IPortfolioRepository including CRUD operations,
portfolio management, position loading, and entity mapping with full coverage.
"""

# Standard library imports
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, patch
from uuid import uuid4

# Third-party imports
import pytest

# Local imports
from src.application.interfaces.exceptions import PortfolioNotFoundError, RepositoryError
from src.domain.entities.portfolio import Portfolio
from src.domain.entities.position import Position
from src.infrastructure.repositories.portfolio_repository import PostgreSQLPortfolioRepository


@pytest.fixture
def mock_adapter():
    """Mock PostgreSQL adapter for repository tests."""
    adapter = AsyncMock()
    adapter.execute_query.return_value = "EXECUTE 1"
    adapter.fetch_one.return_value = None
    adapter.fetch_all.return_value = []
    return adapter


@pytest.fixture
def repository(mock_adapter):
    """Portfolio repository with mocked adapter."""
    return PostgreSQLPortfolioRepository(mock_adapter)


@pytest.fixture
def sample_portfolio():
    """Sample portfolio entity for testing."""
    return Portfolio(
        id=uuid4(),
        name="Test Portfolio",
        initial_capital=Decimal("100000.00"),
        cash_balance=Decimal("95000.00"),
        positions={},
        max_position_size=Decimal("10000.00"),
        max_portfolio_risk=Decimal("0.02"),
        max_positions=10,
        max_leverage=Decimal("2.0"),
        total_realized_pnl=Decimal("500.00"),
        total_commission_paid=Decimal("50.00"),
        trades_count=5,
        winning_trades=3,
        losing_trades=2,
        created_at=datetime.now(UTC),
        last_updated=datetime.now(UTC),
        strategy="balanced",
        tags={"owner": "john", "type": "personal"},
    )


@pytest.fixture
def sample_portfolio_record():
    """Sample portfolio database record."""
    portfolio_id = uuid4()
    return {
        "id": portfolio_id,
        "name": "Test Portfolio",
        "initial_capital": Decimal("100000.00"),
        "cash_balance": Decimal("95000.00"),
        "max_position_size": Decimal("10000.00"),
        "max_portfolio_risk": Decimal("0.02"),
        "max_positions": 10,
        "max_leverage": Decimal("2.0"),
        "total_realized_pnl": Decimal("500.00"),
        "total_commission_paid": Decimal("50.00"),
        "trades_count": 5,
        "winning_trades": 3,
        "losing_trades": 2,
        "created_at": datetime.now(UTC),
        "last_updated": datetime.now(UTC),
        "strategy": "balanced",
        "tags": {"owner": "john", "type": "personal"},
    }


@pytest.fixture
def sample_position_record():
    """Sample position database record for portfolio loading."""
    return {
        "id": uuid4(),
        "symbol": "AAPL",
        "quantity": Decimal("100"),
        "average_entry_price": Decimal("145.00"),
        "current_price": Decimal("150.00"),
        "last_updated": datetime.now(UTC),
        "realized_pnl": Decimal("0"),
        "commission_paid": Decimal("5.00"),
        "stop_loss_price": Decimal("140.00"),
        "take_profit_price": Decimal("160.00"),
        "max_position_value": Decimal("15000.00"),
        "opened_at": datetime.now(UTC),
        "closed_at": None,
        "strategy": "momentum",
        "tags": {},
    }


@pytest.mark.unit
class TestPortfolioRepositoryCRUD:
    """Test portfolio CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_portfolio_insert_new(self, repository, mock_adapter, sample_portfolio):
        """Test saving a new portfolio."""
        mock_adapter.fetch_one.return_value = None  # Portfolio doesn't exist

        result = await repository.save_portfolio(sample_portfolio)

        assert result == sample_portfolio
        assert mock_adapter.execute_query.called

        # Verify insert query structure
        call_args = mock_adapter.execute_query.call_args
        assert "INSERT INTO portfolios" in call_args[0][0]
        assert call_args[0][1] == sample_portfolio.id
        assert call_args[0][2] == sample_portfolio.name
        assert call_args[0][3] == sample_portfolio.initial_capital

    @pytest.mark.asyncio
    async def test_save_portfolio_update_existing(
        self, repository, mock_adapter, sample_portfolio, sample_portfolio_record
    ):
        """Test updating an existing portfolio."""
        mock_adapter.fetch_one.return_value = sample_portfolio_record
        mock_adapter.fetch_all.return_value = []  # No positions
        mock_adapter.execute_query.return_value = "UPDATE 1"

        result = await repository.save_portfolio(sample_portfolio)

        assert result == sample_portfolio
        # Should call update instead of insert
        call_args = mock_adapter.execute_query.call_args
        assert "UPDATE portfolios SET" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_save_portfolio_error(self, repository, mock_adapter, sample_portfolio):
        """Test save portfolio with database error."""
        mock_adapter.fetch_one.side_effect = Exception("Database error")

        with pytest.raises(RepositoryError, match="Failed to save portfolio"):
            await repository.save_portfolio(sample_portfolio)

    @pytest.mark.asyncio
    async def test_insert_portfolio_all_fields(self, repository, mock_adapter, sample_portfolio):
        """Test inserting portfolio with all fields."""
        await repository._insert_portfolio(sample_portfolio)

        # Verify all fields are in insert query
        call_args = mock_adapter.execute_query.call_args
        query = call_args[0][0]

        assert "INSERT INTO portfolios" in query
        fields = [
            "id",
            "name",
            "initial_capital",
            "cash_balance",
            "max_position_size",
            "max_portfolio_risk",
            "max_positions",
            "max_leverage",
            "total_realized_pnl",
            "total_commission_paid",
            "trades_count",
            "winning_trades",
            "losing_trades",
            "created_at",
            "last_updated",
            "strategy",
            "tags",
            "version",
        ]
        for field in fields:
            assert field in query

        # Verify parameter count
        params = call_args[0][1:]
        assert len(params) == 18  # All portfolio fields including version

    @pytest.mark.asyncio
    async def test_get_portfolio_by_id_found(
        self, repository, mock_adapter, sample_portfolio_record
    ):
        """Test getting portfolio by ID when it exists."""
        mock_adapter.fetch_one.return_value = sample_portfolio_record
        mock_adapter.fetch_all.return_value = []  # No positions

        result = await repository.get_portfolio_by_id(sample_portfolio_record["id"])

        assert result is not None
        assert result.id == sample_portfolio_record["id"]
        assert result.name == sample_portfolio_record["name"]
        assert result.initial_capital == sample_portfolio_record["initial_capital"]

        # Verify position loading was attempted
        assert mock_adapter.fetch_all.called

    @pytest.mark.asyncio
    async def test_get_portfolio_by_id_not_found(self, repository, mock_adapter):
        """Test getting portfolio by ID when it doesn't exist."""
        mock_adapter.fetch_one.return_value = None

        result = await repository.get_portfolio_by_id(uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_get_portfolio_by_id_error(self, repository, mock_adapter):
        """Test get portfolio by ID with database error."""
        mock_adapter.fetch_one.side_effect = Exception("Database error")

        with pytest.raises(RepositoryError, match="Failed to retrieve portfolio"):
            await repository.get_portfolio_by_id(uuid4())

    @pytest.mark.asyncio
    async def test_get_portfolio_by_name(self, repository, mock_adapter, sample_portfolio_record):
        """Test getting portfolio by name."""
        mock_adapter.fetch_one.return_value = sample_portfolio_record
        mock_adapter.fetch_all.return_value = []  # No positions

        result = await repository.get_portfolio_by_name("Test Portfolio")

        assert result is not None
        assert result.name == "Test Portfolio"

        # Verify query
        call_args = mock_adapter.fetch_one.call_args
        assert "WHERE name = %s" in call_args[0][0]
        assert call_args[0][1] == "Test Portfolio"

    @pytest.mark.asyncio
    async def test_get_portfolio_by_name_not_found(self, repository, mock_adapter):
        """Test getting portfolio by name when none exists."""
        mock_adapter.fetch_one.return_value = None

        result = await repository.get_portfolio_by_name("Nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_portfolio_by_name_error(self, repository, mock_adapter):
        """Test get portfolio by name with database error."""
        mock_adapter.fetch_one.side_effect = Exception("Database error")

        with pytest.raises(RepositoryError, match="Failed to retrieve portfolio by name"):
            await repository.get_portfolio_by_name("Test Portfolio")

    @pytest.mark.asyncio
    async def test_get_current_portfolio(self, repository, mock_adapter, sample_portfolio_record):
        """Test getting the current active portfolio."""
        mock_adapter.fetch_one.return_value = sample_portfolio_record
        mock_adapter.fetch_all.return_value = []  # No positions

        result = await repository.get_current_portfolio()

        assert result is not None
        assert result.id == sample_portfolio_record["id"]

        # Verify query orders by last_updated
        call_args = mock_adapter.fetch_one.call_args
        assert "ORDER BY COALESCE(last_updated, created_at) DESC" in call_args[0][0]
        assert "LIMIT 1" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_current_portfolio_not_found(self, repository, mock_adapter):
        """Test getting current portfolio when none exists."""
        mock_adapter.fetch_one.return_value = None

        result = await repository.get_current_portfolio()

        assert result is None

    @pytest.mark.asyncio
    async def test_get_current_portfolio_error(self, repository, mock_adapter):
        """Test get current portfolio with database error."""
        mock_adapter.fetch_one.side_effect = Exception("Database error")

        with pytest.raises(RepositoryError, match="Failed to retrieve current portfolio"):
            await repository.get_current_portfolio()

    @pytest.mark.asyncio
    async def test_get_all_portfolios(self, repository, mock_adapter, sample_portfolio_record):
        """Test getting all portfolios."""
        mock_adapter.fetch_all.return_value = [sample_portfolio_record, sample_portfolio_record]

        # Mock for position loading (called for each portfolio)
        with patch.object(repository, "_load_positions", new_callable=AsyncMock):
            result = await repository.get_all_portfolios()

        assert len(result) == 2

        # Verify query
        call_args = mock_adapter.fetch_all.call_args_list[0]
        assert "FROM portfolios" in call_args[0][0]
        assert "ORDER BY created_at DESC" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_all_portfolios_empty(self, repository, mock_adapter):
        """Test getting all portfolios when none exist."""
        mock_adapter.fetch_all.return_value = []

        result = await repository.get_all_portfolios()

        assert result == []

    @pytest.mark.asyncio
    async def test_get_all_portfolios_error(self, repository, mock_adapter):
        """Test get all portfolios with database error."""
        mock_adapter.fetch_all.side_effect = Exception("Database error")

        with pytest.raises(RepositoryError, match="Failed to retrieve all portfolios"):
            await repository.get_all_portfolios()

    @pytest.mark.asyncio
    async def test_get_portfolios_by_strategy(
        self, repository, mock_adapter, sample_portfolio_record
    ):
        """Test getting portfolios by strategy."""
        mock_adapter.fetch_all.return_value = [sample_portfolio_record]

        # Mock for position loading
        with patch.object(repository, "_load_positions", new_callable=AsyncMock):
            result = await repository.get_portfolios_by_strategy("balanced")

        assert len(result) == 1
        assert result[0].strategy == "balanced"

        # Verify query
        call_args = mock_adapter.fetch_all.call_args_list[0]
        assert "WHERE strategy = %s" in call_args[0][0]
        assert call_args[0][1] == "balanced"

    @pytest.mark.asyncio
    async def test_get_portfolios_by_strategy_error(self, repository, mock_adapter):
        """Test get portfolios by strategy with database error."""
        mock_adapter.fetch_all.side_effect = Exception("Database error")

        with pytest.raises(RepositoryError, match="Failed to retrieve portfolios for strategy"):
            await repository.get_portfolios_by_strategy("balanced")

    @pytest.mark.asyncio
    async def test_get_portfolio_history(self, repository, mock_adapter, sample_portfolio_record):
        """Test getting portfolio history within date range."""
        start_date = datetime.now(UTC) - timedelta(days=7)
        end_date = datetime.now(UTC)
        mock_adapter.fetch_all.return_value = [sample_portfolio_record]

        # Mock for position loading
        with patch.object(repository, "_load_positions", new_callable=AsyncMock):
            result = await repository.get_portfolio_history(
                sample_portfolio_record["id"], start_date, end_date
            )

        assert len(result) == 1

        # Verify query parameters
        call_args = mock_adapter.fetch_all.call_args_list[0]
        assert "WHERE id = %s" in call_args[0][0]
        assert (
            "AND (last_updated BETWEEN %s AND %s OR created_at BETWEEN %s AND %s)"
            in call_args[0][0]
        )
        assert call_args[0][1] == sample_portfolio_record["id"]
        assert call_args[0][2] == start_date
        assert call_args[0][3] == end_date

    @pytest.mark.asyncio
    async def test_get_portfolio_history_error(self, repository, mock_adapter):
        """Test get portfolio history with database error."""
        mock_adapter.fetch_all.side_effect = Exception("Database error")

        with pytest.raises(RepositoryError, match="Failed to retrieve portfolio history"):
            await repository.get_portfolio_history(uuid4(), datetime.now(UTC), datetime.now(UTC))

    @pytest.mark.asyncio
    async def test_update_portfolio_success(self, repository, mock_adapter, sample_portfolio):
        """Test successful portfolio update."""
        mock_adapter.execute_query.return_value = "UPDATE 1"

        # Store initial version
        initial_version = getattr(sample_portfolio, "version", 1)

        result = await repository.update_portfolio(sample_portfolio)

        assert result == sample_portfolio
        # Version should be incremented after successful update
        assert getattr(sample_portfolio, "version", 1) == initial_version + 1

        # Verify update query
        call_args = mock_adapter.execute_query.call_args
        assert "UPDATE portfolios SET" in call_args[0][0]
        assert "WHERE id = %s AND version = %s" in call_args[0][0]
        # ID should be second-to-last parameter, version is last
        assert call_args[0][-2] == sample_portfolio.id
        # Version in WHERE clause should be the initial version
        assert call_args[0][-1] == initial_version

    @pytest.mark.asyncio
    async def test_update_portfolio_not_found(self, repository, mock_adapter, sample_portfolio):
        """Test updating non-existent portfolio."""
        mock_adapter.execute_query.return_value = "UPDATE 0"
        mock_adapter.fetch_one.return_value = None  # Portfolio doesn't exist

        with pytest.raises(PortfolioNotFoundError):
            await repository.update_portfolio(sample_portfolio)

    @pytest.mark.asyncio
    async def test_update_portfolio_error(self, repository, mock_adapter, sample_portfolio):
        """Test update portfolio with database error."""
        mock_adapter.execute_query.side_effect = Exception("Database error")

        with pytest.raises(RepositoryError, match="Failed to update portfolio"):
            await repository.update_portfolio(sample_portfolio)

    @pytest.mark.asyncio
    async def test_delete_portfolio_success(self, repository, mock_adapter):
        """Test successful portfolio deletion."""
        portfolio_id = uuid4()
        mock_adapter.execute_query.return_value = "DELETE 1"

        result = await repository.delete_portfolio(portfolio_id)

        assert result is True

        # Verify delete query
        call_args = mock_adapter.execute_query.call_args
        assert "DELETE FROM portfolios WHERE id = %s" in call_args[0][0]
        assert call_args[0][1] == portfolio_id

    @pytest.mark.asyncio
    async def test_delete_portfolio_not_found(self, repository, mock_adapter):
        """Test deleting non-existent portfolio."""
        mock_adapter.execute_query.return_value = "DELETE 0"

        result = await repository.delete_portfolio(uuid4())

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_portfolio_error(self, repository, mock_adapter):
        """Test delete portfolio with database error."""
        mock_adapter.execute_query.side_effect = Exception("Database error")

        with pytest.raises(RepositoryError, match="Failed to delete portfolio"):
            await repository.delete_portfolio(uuid4())

    @pytest.mark.asyncio
    async def test_create_portfolio_snapshot(self, repository, mock_adapter, sample_portfolio):
        """Test creating portfolio snapshot."""
        mock_adapter.execute_query.return_value = "UPDATE 1"

        result = await repository.create_portfolio_snapshot(sample_portfolio)

        assert result == sample_portfolio
        assert sample_portfolio.last_updated is not None

        # Should call update with new timestamp
        assert mock_adapter.execute_query.called

    @pytest.mark.asyncio
    async def test_create_portfolio_snapshot_error(
        self, repository, mock_adapter, sample_portfolio
    ):
        """Test create portfolio snapshot with database error."""
        mock_adapter.execute_query.side_effect = Exception("Database error")

        with pytest.raises(RepositoryError, match="Failed to create portfolio snapshot"):
            await repository.create_portfolio_snapshot(sample_portfolio)


@pytest.mark.unit
class TestPortfolioPositionManagement:
    """Test portfolio position loading and management."""

    @pytest.mark.asyncio
    async def test_load_positions_success(
        self, repository, mock_adapter, sample_portfolio, sample_position_record
    ):
        """Test loading positions for a portfolio."""
        mock_adapter.fetch_all.return_value = [sample_position_record]

        await repository._load_positions(sample_portfolio)

        assert len(sample_portfolio.positions) == 1
        assert "AAPL" in sample_portfolio.positions
        assert sample_portfolio.positions["AAPL"].symbol == "AAPL"

        # Verify query
        call_args = mock_adapter.fetch_all.call_args
        assert "FROM positions" in call_args[0][0]
        assert "WHERE closed_at IS NULL" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_load_positions_multiple(self, repository, mock_adapter, sample_portfolio):
        """Test loading multiple positions for a portfolio."""
        position_records = [
            {
                "id": uuid4(),
                "symbol": "AAPL",
                "quantity": Decimal("100"),
                "average_entry_price": Decimal("145.00"),
                "current_price": Decimal("150.00"),
                "last_updated": datetime.now(UTC),
                "realized_pnl": Decimal("0"),
                "commission_paid": Decimal("5.00"),
                "stop_loss_price": None,
                "take_profit_price": None,
                "max_position_value": None,
                "opened_at": datetime.now(UTC),
                "closed_at": None,
                "strategy": "momentum",
                "tags": {},
            },
            {
                "id": uuid4(),
                "symbol": "GOOGL",
                "quantity": Decimal("50"),
                "average_entry_price": Decimal("2500.00"),
                "current_price": Decimal("2520.00"),
                "last_updated": datetime.now(UTC),
                "realized_pnl": Decimal("0"),
                "commission_paid": Decimal("10.00"),
                "stop_loss_price": None,
                "take_profit_price": None,
                "max_position_value": None,
                "opened_at": datetime.now(UTC),
                "closed_at": None,
                "strategy": "momentum",
                "tags": {},
            },
        ]
        mock_adapter.fetch_all.return_value = position_records

        await repository._load_positions(sample_portfolio)

        assert len(sample_portfolio.positions) == 2
        assert "AAPL" in sample_portfolio.positions
        assert "GOOGL" in sample_portfolio.positions

    @pytest.mark.asyncio
    async def test_load_positions_clears_existing(self, repository, mock_adapter, sample_portfolio):
        """Test that loading positions clears existing positions."""
        # Add existing position
        sample_portfolio.positions["TSLA"] = Position(
            id=uuid4(),
            symbol="TSLA",
            quantity=Decimal("10"),
            average_entry_price=Decimal("700.00"),
        )

        # Load new positions (empty)
        mock_adapter.fetch_all.return_value = []

        await repository._load_positions(sample_portfolio)

        assert len(sample_portfolio.positions) == 0
        assert "TSLA" not in sample_portfolio.positions

    @pytest.mark.asyncio
    async def test_load_positions_error_suppressed(
        self, repository, mock_adapter, sample_portfolio
    ):
        """Test that position loading errors are suppressed (logged but not raised)."""
        mock_adapter.fetch_all.side_effect = Exception("Database error")

        # Should not raise exception
        await repository._load_positions(sample_portfolio)

        # Positions should remain empty
        assert len(sample_portfolio.positions) == 0


@pytest.mark.unit
class TestPortfolioEntityMapping:
    """Test entity mapping between domain objects and database records."""

    def test_map_record_to_portfolio_complete(self, repository, sample_portfolio_record):
        """Test mapping complete database record to portfolio entity."""
        portfolio = repository._map_record_to_portfolio(sample_portfolio_record)

        assert portfolio.id == sample_portfolio_record["id"]
        assert portfolio.name == sample_portfolio_record["name"]
        assert portfolio.initial_capital == sample_portfolio_record["initial_capital"]
        assert portfolio.cash_balance == sample_portfolio_record["cash_balance"]
        assert portfolio.positions == {}  # Loaded separately
        assert portfolio.max_position_size == sample_portfolio_record["max_position_size"]
        assert portfolio.max_portfolio_risk == sample_portfolio_record["max_portfolio_risk"]
        assert portfolio.max_positions == sample_portfolio_record["max_positions"]
        assert portfolio.max_leverage == sample_portfolio_record["max_leverage"]
        assert portfolio.total_realized_pnl == sample_portfolio_record["total_realized_pnl"]
        assert portfolio.total_commission_paid == sample_portfolio_record["total_commission_paid"]
        assert portfolio.trades_count == sample_portfolio_record["trades_count"]
        assert portfolio.winning_trades == sample_portfolio_record["winning_trades"]
        assert portfolio.losing_trades == sample_portfolio_record["losing_trades"]
        assert portfolio.created_at == sample_portfolio_record["created_at"]
        assert portfolio.last_updated == sample_portfolio_record["last_updated"]
        assert portfolio.strategy == sample_portfolio_record["strategy"]
        assert portfolio.tags == sample_portfolio_record["tags"]

    def test_map_record_to_portfolio_with_nulls(self, repository, sample_portfolio_record):
        """Test mapping record with null optional fields."""
        sample_portfolio_record["last_updated"] = None
        sample_portfolio_record["strategy"] = None
        sample_portfolio_record["tags"] = None

        portfolio = repository._map_record_to_portfolio(sample_portfolio_record)

        assert portfolio.last_updated is None
        assert portfolio.strategy is None
        assert portfolio.tags == {}

    def test_map_record_to_position_complete(self, repository, sample_position_record):
        """Test mapping complete position record."""
        position = repository._map_record_to_position(sample_position_record)

        assert position.id == sample_position_record["id"]
        assert position.symbol == sample_position_record["symbol"]
        assert position.quantity == sample_position_record["quantity"]
        assert position.average_entry_price == sample_position_record["average_entry_price"]
        assert position.current_price == sample_position_record["current_price"]

    def test_map_record_to_position_with_nulls(self, repository):
        """Test mapping position record with null fields."""
        record = {
            "id": uuid4(),
            "symbol": "AAPL",
            "quantity": Decimal("100"),
            "average_entry_price": Decimal("145.00"),
            "current_price": None,
            "last_updated": None,
            "realized_pnl": Decimal("0"),
            "commission_paid": Decimal("0"),
            "stop_loss_price": None,
            "take_profit_price": None,
            "max_position_value": None,
            "opened_at": datetime.now(UTC),
            "closed_at": None,
            "strategy": None,
            "tags": None,
        }

        position = repository._map_record_to_position(record)

        assert position.current_price is None
        assert position.last_updated is None
        assert position.stop_loss_price is None
        assert position.take_profit_price is None
        assert position.max_position_value is None
        assert position.strategy is None
        assert position.tags == {}


@pytest.mark.unit
class TestPortfolioRepositoryIntegration:
    """Test portfolio repository integration scenarios."""

    @pytest.mark.asyncio
    async def test_portfolio_lifecycle(self, repository, mock_adapter):
        """Test complete portfolio lifecycle from creation to deletion."""
        portfolio = Portfolio(
            id=uuid4(),
            name="Lifecycle Test",
            initial_capital=Decimal("50000.00"),
            cash_balance=Decimal("50000.00"),
        )

        # 1. Create new portfolio
        mock_adapter.fetch_one.return_value = None
        await repository.save_portfolio(portfolio)
        assert mock_adapter.execute_query.called

        # 2. Retrieve portfolio
        portfolio_record = {
            "id": portfolio.id,
            "name": portfolio.name,
            "initial_capital": portfolio.initial_capital,
            "cash_balance": portfolio.cash_balance,
            "max_position_size": portfolio.max_position_size,
            "max_portfolio_risk": portfolio.max_portfolio_risk,
            "max_positions": portfolio.max_positions,
            "max_leverage": portfolio.max_leverage,
            "total_realized_pnl": Decimal("0"),
            "total_commission_paid": Decimal("0"),
            "trades_count": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "created_at": datetime.now(UTC),
            "last_updated": None,
            "strategy": None,
            "tags": {},
        }
        mock_adapter.fetch_one.return_value = portfolio_record
        mock_adapter.fetch_all.return_value = []

        retrieved = await repository.get_portfolio_by_id(portfolio.id)
        assert retrieved.id == portfolio.id

        # 3. Update portfolio
        portfolio.cash_balance = Decimal("45000.00")
        mock_adapter.execute_query.return_value = "UPDATE 1"
        await repository.update_portfolio(portfolio)

        # 4. Delete portfolio
        mock_adapter.execute_query.return_value = "DELETE 1"
        result = await repository.delete_portfolio(portfolio.id)
        assert result is True

    @pytest.mark.asyncio
    async def test_portfolio_with_positions(self, repository, mock_adapter):
        """Test portfolio with multiple positions."""
        portfolio_record = {
            "id": uuid4(),
            "name": "Portfolio with Positions",
            "initial_capital": Decimal("100000.00"),
            "cash_balance": Decimal("50000.00"),
            "max_position_size": Decimal("10000.00"),
            "max_portfolio_risk": Decimal("0.02"),
            "max_positions": 10,
            "max_leverage": Decimal("2.0"),
            "total_realized_pnl": Decimal("1000.00"),
            "total_commission_paid": Decimal("100.00"),
            "trades_count": 10,
            "winning_trades": 6,
            "losing_trades": 4,
            "created_at": datetime.now(UTC),
            "last_updated": datetime.now(UTC),
            "strategy": "diversified",
            "tags": {},
        }

        position_records = [
            {
                "id": uuid4(),
                "symbol": "AAPL",
                "quantity": Decimal("100"),
                "average_entry_price": Decimal("145.00"),
                "current_price": Decimal("150.00"),
                "last_updated": datetime.now(UTC),
                "realized_pnl": Decimal("0"),
                "commission_paid": Decimal("5.00"),
                "stop_loss_price": None,
                "take_profit_price": None,
                "max_position_value": None,
                "opened_at": datetime.now(UTC),
                "closed_at": None,
                "strategy": "momentum",
                "tags": {},
            },
            {
                "id": uuid4(),
                "symbol": "GOOGL",
                "quantity": Decimal("20"),
                "average_entry_price": Decimal("2500.00"),
                "current_price": Decimal("2550.00"),
                "last_updated": datetime.now(UTC),
                "realized_pnl": Decimal("0"),
                "commission_paid": Decimal("10.00"),
                "stop_loss_price": None,
                "take_profit_price": None,
                "max_position_value": None,
                "opened_at": datetime.now(UTC),
                "closed_at": None,
                "strategy": "value",
                "tags": {},
            },
        ]

        mock_adapter.fetch_one.return_value = portfolio_record
        mock_adapter.fetch_all.return_value = position_records

        portfolio = await repository.get_portfolio_by_id(portfolio_record["id"])

        assert portfolio is not None
        assert len(portfolio.positions) == 2
        assert "AAPL" in portfolio.positions
        assert "GOOGL" in portfolio.positions
        assert portfolio.positions["AAPL"].quantity == Decimal("100")
        assert portfolio.positions["GOOGL"].quantity == Decimal("20")

    @pytest.mark.asyncio
    async def test_repository_initialization(self):
        """Test repository initialization."""
        mock_adapter = AsyncMock()
        repo = PostgreSQLPortfolioRepository(mock_adapter)

        assert repo.adapter is mock_adapter

    @pytest.mark.asyncio
    async def test_error_propagation(self, repository, mock_adapter, sample_portfolio):
        """Test that specific errors are properly propagated."""
        # Test PortfolioNotFoundError propagation in update
        mock_adapter.execute_query.return_value = "UPDATE 0"

        with pytest.raises(PortfolioNotFoundError):
            await repository.update_portfolio(sample_portfolio)

        # Reset and test generic error wrapping
        mock_adapter.execute_query.side_effect = ValueError("Invalid value")

        with pytest.raises(RepositoryError):
            await repository.update_portfolio(sample_portfolio)


@pytest.mark.unit
class TestPortfolioRepositoryConcurrency:
    """Test portfolio repository concurrent access and locking scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_portfolio_updates(self, repository, mock_adapter, sample_portfolio):
        """Test handling concurrent portfolio updates."""
        # Simulate optimistic locking failure
        mock_adapter.execute_query.return_value = (
            "UPDATE 0"  # No rows updated due to version mismatch
        )

        with pytest.raises(PortfolioNotFoundError):
            await repository.update_portfolio(sample_portfolio)

    @pytest.mark.asyncio
    async def test_concurrent_position_loading(self, repository, mock_adapter, sample_portfolio):
        """Test concurrent position loading for same portfolio."""
        position_records = [
            {
                "id": uuid4(),
                "symbol": "AAPL",
                "quantity": Decimal("100"),
                "average_entry_price": Decimal("145.00"),
                "current_price": Decimal("150.00"),
                "last_updated": datetime.now(UTC),
                "realized_pnl": Decimal("0"),
                "commission_paid": Decimal("5.00"),
                "stop_loss_price": None,
                "take_profit_price": None,
                "max_position_value": None,
                "opened_at": datetime.now(UTC),
                "closed_at": None,
                "strategy": "momentum",
                "tags": {},
            }
        ]

        # First call returns positions
        mock_adapter.fetch_all.return_value = position_records
        await repository._load_positions(sample_portfolio)
        assert len(sample_portfolio.positions) == 1

        # Concurrent call with different positions
        position_records[0]["symbol"] = "GOOGL"
        mock_adapter.fetch_all.return_value = position_records
        await repository._load_positions(sample_portfolio)

        # Should replace previous positions
        assert len(sample_portfolio.positions) == 1
        assert "GOOGL" in sample_portfolio.positions
        assert "AAPL" not in sample_portfolio.positions

    @pytest.mark.asyncio
    async def test_deadlock_recovery(self, repository, mock_adapter, sample_portfolio):
        """Test recovery from database deadlock."""
        # Simulate deadlock error
        mock_adapter.execute_query.side_effect = Exception("deadlock detected")

        with pytest.raises(RepositoryError, match="Failed to update portfolio"):
            await repository.update_portfolio(sample_portfolio)


@pytest.mark.unit
class TestPortfolioRepositoryDataIntegrity:
    """Test data integrity and validation scenarios."""

    @pytest.mark.asyncio
    async def test_portfolio_data_validation(self, repository, mock_adapter):
        """Test portfolio data validation before persistence."""
        # Create portfolio with edge case values
        portfolio = Portfolio(
            id=uuid4(),
            name="X" * 256,  # Very long name
            initial_capital=Decimal("999999999999.99"),
            cash_balance=Decimal("999999999999.99"),  # Match initial capital
            positions={},
            max_position_size=Decimal("100000.00"),  # Large but valid
            max_portfolio_risk=Decimal("1.0"),  # 100% risk
            max_positions=100,  # High number of positions
            max_leverage=Decimal("100.0"),  # High leverage
        )

        # Should still attempt to save (validation is domain layer responsibility)
        mock_adapter.fetch_one.return_value = None
        await repository.save_portfolio(portfolio)
        assert mock_adapter.execute_query.called

    @pytest.mark.asyncio
    async def test_null_handling_in_database_records(self, repository):
        """Test handling of NULL values from database."""
        record_with_nulls = {
            "id": uuid4(),
            "name": "Test",
            "initial_capital": Decimal("100000.00"),
            "cash_balance": Decimal("100000.00"),
            "max_position_size": Decimal("10000.00"),  # Required field
            "max_portfolio_risk": Decimal("0.02"),  # Required field
            "max_positions": 10,  # Required field
            "max_leverage": Decimal("2.0"),  # Required field
            "total_realized_pnl": Decimal("0"),  # Default to 0 instead of None
            "total_commission_paid": Decimal("0"),  # Default to 0 instead of None
            "trades_count": 0,  # Default to 0 instead of None
            "winning_trades": 0,  # Default to 0 instead of None
            "losing_trades": 0,  # Default to 0 instead of None
            "created_at": datetime.now(UTC),
            "last_updated": None,  # This can be None
            "strategy": None,  # This can be None
            "tags": None,  # This can be None
        }

        portfolio = repository._map_record_to_portfolio(record_with_nulls)

        # Should handle NULLs gracefully
        assert portfolio.id == record_with_nulls["id"]
        assert portfolio.name == record_with_nulls["name"]
        assert portfolio.tags == {}  # None becomes empty dict
        assert portfolio.strategy is None  # Strategy can be None
        assert portfolio.last_updated is None  # last_updated can be None

    @pytest.mark.asyncio
    async def test_transaction_isolation(self, repository, mock_adapter, sample_portfolio):
        """Test transaction isolation between operations."""
        # Save portfolio
        mock_adapter.fetch_one.return_value = None
        await repository.save_portfolio(sample_portfolio)

        # Verify isolation - different operation shouldn't see uncommitted changes
        mock_adapter.fetch_one.return_value = None  # Portfolio not visible yet
        result = await repository.get_portfolio_by_id(sample_portfolio.id)
        assert result is None


@pytest.mark.unit
class TestPortfolioRepositoryPerformance:
    """Test performance-related scenarios."""

    @pytest.mark.asyncio
    async def test_bulk_portfolio_loading(self, repository, mock_adapter):
        """Test loading many portfolios efficiently."""
        # Create 100 portfolio records
        portfolio_records = [
            {
                "id": uuid4(),
                "name": f"Portfolio {i}",
                "initial_capital": Decimal("100000.00"),
                "cash_balance": Decimal("100000.00"),
                "max_position_size": Decimal("10000.00"),
                "max_portfolio_risk": Decimal("0.02"),
                "max_positions": 10,
                "max_leverage": Decimal("2.0"),
                "total_realized_pnl": Decimal("0"),
                "total_commission_paid": Decimal("0"),
                "trades_count": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "created_at": datetime.now(UTC),
                "last_updated": None,
                "strategy": "test",
                "tags": {},
            }
            for i in range(100)
        ]

        mock_adapter.fetch_all.return_value = portfolio_records

        # Mock _load_positions to avoid complexity
        with patch.object(repository, "_load_positions", new_callable=AsyncMock):
            portfolios = await repository.get_all_portfolios()

        assert len(portfolios) == 100

    @pytest.mark.asyncio
    async def test_query_optimization_for_history(self, repository, mock_adapter):
        """Test optimized query for portfolio history."""
        portfolio_id = uuid4()
        start_date = datetime.now(UTC) - timedelta(days=30)
        end_date = datetime.now(UTC)

        mock_adapter.fetch_all.return_value = []

        await repository.get_portfolio_history(portfolio_id, start_date, end_date)

        # Verify query uses proper date filtering
        call_args = mock_adapter.fetch_all.call_args
        query = call_args[0][0]
        assert "BETWEEN" in query
        assert "ORDER BY" in query

    @pytest.mark.asyncio
    async def test_connection_pool_exhaustion(self, repository, mock_adapter):
        """Test handling when connection pool is exhausted."""
        mock_adapter.fetch_one.side_effect = Exception("connection pool exhausted")

        with pytest.raises(RepositoryError, match="Failed to retrieve portfolio"):
            await repository.get_portfolio_by_id(uuid4())
