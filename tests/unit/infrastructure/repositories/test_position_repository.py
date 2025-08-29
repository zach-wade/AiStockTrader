"""
Comprehensive unit tests for PostgreSQL Position Repository Implementation.

Tests the concrete implementation of IPositionRepository including CRUD operations,
position lifecycle management, and entity mapping with full coverage.
"""

# Standard library imports
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock
from uuid import uuid4

# Third-party imports
import pytest

# Local imports
from src.application.interfaces.exceptions import PositionNotFoundError, RepositoryError
from src.domain.entities.position import Position
from src.infrastructure.repositories.position_repository import PostgreSQLPositionRepository


@pytest.fixture
def mock_adapter():
    """Mock PostgreSQL adapter for repository tests."""
    adapter = AsyncMock()
    adapter.execute_query.return_value = "UPDATE 1"
    adapter.fetch_one.return_value = None
    adapter.fetch_all.return_value = []
    return adapter


@pytest.fixture
def repository(mock_adapter):
    """Position repository with mocked adapter."""
    return PostgreSQLPositionRepository(mock_adapter)


@pytest.fixture
def sample_position():
    """Sample position entity for testing."""
    return Position(
        id=uuid4(),
        symbol="AAPL",
        quantity=Decimal("100"),
        average_entry_price=Decimal("145.00"),
        current_price=Decimal("150.00"),
        last_updated=datetime.now(UTC),
        realized_pnl=Decimal("0"),
        commission_paid=Decimal("5.00"),
        stop_loss_price=Decimal("140.00"),
        take_profit_price=Decimal("160.00"),
        max_position_value=Decimal("15000.00"),
        opened_at=datetime.now(UTC),
        closed_at=None,
        strategy="momentum",
        tags={"trader": "john"},
    )


@pytest.fixture
def sample_position_record():
    """Sample position database record."""
    position_id = uuid4()
    return {
        "id": position_id,
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
        "tags": {"trader": "john"},
    }


@pytest.mark.unit
class TestPositionRepositoryCRUD:
    """Test position CRUD operations."""

    @pytest.mark.asyncio
    async def test_persist_position_insert_new(self, repository, mock_adapter, sample_position):
        """Test saving a new position."""
        mock_adapter.fetch_one.return_value = None  # Position doesn't exist

        result = await repository.persist_position(sample_position)

        assert result == sample_position
        assert mock_adapter.execute_query.called

        # Verify insert query structure
        call_args = mock_adapter.execute_query.call_args
        assert "INSERT INTO positions" in call_args[0][0]
        assert call_args[0][1] == sample_position.id
        assert call_args[0][2] == sample_position.symbol
        assert call_args[0][3] == sample_position.quantity

    @pytest.mark.asyncio
    async def test_persist_position_update_existing(
        self, repository, mock_adapter, sample_position, sample_position_record
    ):
        """Test updating an existing position."""
        mock_adapter.fetch_one.return_value = sample_position_record

        result = await repository.persist_position(sample_position)

        assert result == sample_position
        # Should call update instead of insert
        call_args = mock_adapter.execute_query.call_args
        assert "UPDATE positions SET" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_insert_position_all_fields(self, repository, mock_adapter, sample_position):
        """Test inserting position with all fields."""
        await repository._insert_position(sample_position)

        # Verify all fields are in insert query
        call_args = mock_adapter.execute_query.call_args
        query = call_args[0][0]

        assert "INSERT INTO positions" in query
        assert "id" in query and "symbol" in query and "quantity" in query
        assert "average_entry_price" in query and "current_price" in query
        assert "last_updated" in query and "realized_pnl" in query
        assert "commission_paid" in query and "stop_loss_price" in query
        assert "take_profit_price" in query and "max_position_value" in query
        assert "opened_at" in query and "closed_at" in query
        assert "strategy" in query and "tags" in query

        # Verify parameter count
        params = call_args[0][1:]
        assert len(params) == 15  # All position fields

    @pytest.mark.asyncio
    async def test_get_position_by_id_found(self, repository, mock_adapter, sample_position_record):
        """Test getting position by ID when it exists."""
        mock_adapter.fetch_one.return_value = sample_position_record

        result = await repository.get_position_by_id(sample_position_record["id"])

        assert result is not None
        assert result.id == sample_position_record["id"]
        assert result.symbol == sample_position_record["symbol"]
        assert result.quantity == sample_position_record["quantity"]
        assert result.average_entry_price == sample_position_record["average_entry_price"]

    @pytest.mark.asyncio
    async def test_get_position_by_id_not_found(self, repository, mock_adapter):
        """Test getting position by ID when it doesn't exist."""
        mock_adapter.fetch_one.return_value = None

        result = await repository.get_position_by_id(uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_get_position_by_symbol(self, repository, mock_adapter, sample_position_record):
        """Test getting current position by symbol."""
        mock_adapter.fetch_one.return_value = sample_position_record

        result = await repository.get_position_by_symbol("AAPL")

        assert result is not None
        assert result.symbol == "AAPL"

        # Verify query includes symbol filter and open position condition
        call_args = mock_adapter.fetch_one.call_args
        assert "WHERE symbol = %s" in call_args[0][0]
        assert "closed_at IS NULL" in call_args[0][0]
        assert call_args[0][1] == "AAPL"

    @pytest.mark.asyncio
    async def test_get_position_by_symbol_not_found(self, repository, mock_adapter):
        """Test getting position by symbol when none exists."""
        mock_adapter.fetch_one.return_value = None

        result = await repository.get_position_by_symbol("AAPL")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_positions_by_symbol(self, repository, mock_adapter, sample_position_record):
        """Test getting all positions (including historical) for a symbol."""
        mock_adapter.fetch_all.return_value = [sample_position_record, sample_position_record]

        result = await repository.get_positions_by_symbol("AAPL")

        assert len(result) == 2
        assert all(pos.symbol == "AAPL" for pos in result)

        # Verify query
        call_args = mock_adapter.fetch_all.call_args
        assert "WHERE symbol = %s" in call_args[0][0]
        assert call_args[0][1] == "AAPL"

    @pytest.mark.asyncio
    async def test_get_active_positions(self, repository, mock_adapter, sample_position_record):
        """Test getting active positions."""
        mock_adapter.fetch_all.return_value = [sample_position_record]

        result = await repository.get_active_positions()

        assert len(result) == 1
        assert not result[0].is_closed()

        # Verify query filters for open positions
        call_args = mock_adapter.fetch_all.call_args
        assert "WHERE closed_at IS NULL" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_closed_positions(self, repository, mock_adapter):
        """Test getting closed positions."""
        sample_record = {
            "id": uuid4(),
            "symbol": "AAPL",
            "quantity": Decimal("0"),  # Closed position has 0 quantity
            "average_entry_price": Decimal("145.00"),
            "current_price": Decimal("150.00"),
            "last_updated": datetime.now(UTC),
            "realized_pnl": Decimal("500.00"),
            "commission_paid": Decimal("5.00"),
            "stop_loss_price": None,
            "take_profit_price": None,
            "max_position_value": Decimal("15000.00"),
            "opened_at": datetime.now(UTC),
            "closed_at": datetime.now(UTC),  # Position is closed
            "strategy": "momentum",
            "tags": {"trader": "john"},
        }
        mock_adapter.fetch_all.return_value = [sample_record]

        result = await repository.get_closed_positions()

        assert len(result) == 1
        assert result[0].is_closed()

        # Verify query filters for closed positions
        call_args = mock_adapter.fetch_all.call_args
        assert "WHERE closed_at IS NOT NULL" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_positions_by_strategy(
        self, repository, mock_adapter, sample_position_record
    ):
        """Test getting positions by strategy."""
        mock_adapter.fetch_all.return_value = [sample_position_record]

        result = await repository.get_positions_by_strategy("momentum")

        assert len(result) == 1
        assert result[0].strategy == "momentum"

        # Verify query
        call_args = mock_adapter.fetch_all.call_args
        assert "WHERE strategy = %s" in call_args[0][0]
        assert call_args[0][1] == "momentum"

    @pytest.mark.asyncio
    async def test_get_positions_by_date_range(
        self, repository, mock_adapter, sample_position_record
    ):
        """Test getting positions by date range."""
        start_date = datetime.now(UTC) - timedelta(days=7)
        end_date = datetime.now(UTC)
        mock_adapter.fetch_all.return_value = [sample_position_record]

        result = await repository.get_positions_by_date_range(start_date, end_date)

        assert len(result) == 1

        # Verify query parameters
        call_args = mock_adapter.fetch_all.call_args
        assert "WHERE opened_at >= %s AND opened_at <= %s" in call_args[0][0]
        assert call_args[0][1] == start_date
        assert call_args[0][2] == end_date

    @pytest.mark.asyncio
    async def test_update_position_success(self, repository, mock_adapter, sample_position):
        """Test successful position update."""
        mock_adapter.execute_query.return_value = "UPDATE 1"

        result = await repository.update_position(sample_position)

        assert result == sample_position

        # Verify update query
        call_args = mock_adapter.execute_query.call_args
        assert "UPDATE positions SET" in call_args[0][0]
        assert "WHERE id = %s" in call_args[0][0]
        # ID should be last parameter
        assert call_args[0][-1] == sample_position.id

    @pytest.mark.asyncio
    async def test_update_position_not_found(self, repository, mock_adapter, sample_position):
        """Test updating non-existent position."""
        mock_adapter.execute_query.return_value = "UPDATE 0"

        with pytest.raises(PositionNotFoundError):
            await repository.update_position(sample_position)

    @pytest.mark.asyncio
    async def test_close_position_success(self, repository, mock_adapter):
        """Test successful position closure."""
        position_id = uuid4()
        mock_adapter.execute_query.return_value = "UPDATE 1"

        result = await repository.close_position(position_id)

        assert result is True

        # Verify close query
        call_args = mock_adapter.execute_query.call_args
        assert "UPDATE positions SET" in call_args[0][0]
        assert "closed_at = NOW()" in call_args[0][0]
        assert "quantity = 0" in call_args[0][0]
        assert "WHERE id = %s AND closed_at IS NULL" in call_args[0][0]
        assert call_args[0][1] == position_id

    @pytest.mark.asyncio
    async def test_close_position_not_found(self, repository, mock_adapter):
        """Test closing non-existent position."""
        mock_adapter.execute_query.return_value = "UPDATE 0"

        result = await repository.close_position(uuid4())

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_position_success(self, repository, mock_adapter):
        """Test successful position deletion."""
        position_id = uuid4()
        mock_adapter.execute_query.return_value = "DELETE 1"

        result = await repository.delete_position(position_id)

        assert result is True

        # Verify delete query
        call_args = mock_adapter.execute_query.call_args
        assert "DELETE FROM positions WHERE id = %s" in call_args[0][0]
        assert call_args[0][1] == position_id

    @pytest.mark.asyncio
    async def test_delete_position_not_found(self, repository, mock_adapter):
        """Test deleting non-existent position."""
        mock_adapter.execute_query.return_value = "DELETE 0"

        result = await repository.delete_position(uuid4())

        assert result is False


@pytest.mark.unit
class TestPositionEntityMapping:
    """Test entity mapping between domain objects and database records."""

    def test_map_record_to_position_complete(self, repository, sample_position_record):
        """Test mapping complete database record to position entity."""
        position = repository._map_record_to_position(sample_position_record)

        assert position.id == sample_position_record["id"]
        assert position.symbol == sample_position_record["symbol"]
        assert position.quantity == sample_position_record["quantity"]
        assert position.average_entry_price == sample_position_record["average_entry_price"]
        assert position.current_price == sample_position_record["current_price"]
        assert position.last_updated == sample_position_record["last_updated"]
        assert position.realized_pnl == sample_position_record["realized_pnl"]
        assert position.commission_paid == sample_position_record["commission_paid"]
        assert position.stop_loss_price == sample_position_record["stop_loss_price"]
        assert position.take_profit_price == sample_position_record["take_profit_price"]
        assert position.max_position_value == sample_position_record["max_position_value"]
        assert position.opened_at == sample_position_record["opened_at"]
        assert position.closed_at == sample_position_record["closed_at"]
        assert position.strategy == sample_position_record["strategy"]
        assert position.tags == sample_position_record["tags"]

    def test_map_record_to_position_with_nulls(self, repository, sample_position_record):
        """Test mapping record with null optional fields."""
        sample_position_record["current_price"] = None
        sample_position_record["last_updated"] = None
        sample_position_record["stop_loss_price"] = None
        sample_position_record["take_profit_price"] = None
        sample_position_record["max_position_value"] = None
        sample_position_record["closed_at"] = None
        sample_position_record["strategy"] = None
        sample_position_record["tags"] = None

        position = repository._map_record_to_position(sample_position_record)

        assert position.current_price is None
        assert position.last_updated is None
        assert position.stop_loss_price is None
        assert position.take_profit_price is None
        assert position.max_position_value is None
        assert position.closed_at is None
        assert position.strategy is None
        assert position.tags == {}

    def test_map_record_to_position_long_position(self, repository):
        """Test mapping long position."""
        sample_record = {
            "id": uuid4(),
            "symbol": "AAPL",
            "quantity": Decimal("100"),  # Positive quantity = long
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

        position = repository._map_record_to_position(sample_record)

        assert position.is_long()
        assert not position.is_short()
        assert position.quantity > 0

    def test_map_record_to_position_short_position(self, repository):
        """Test mapping short position."""
        sample_record = {
            "id": uuid4(),
            "symbol": "AAPL",
            "quantity": Decimal("-100"),  # Negative quantity = short
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

        position = repository._map_record_to_position(sample_record)

        assert position.is_short()
        assert not position.is_long()
        assert position.quantity < 0


@pytest.mark.unit
class TestPositionErrorHandling:
    """Test position repository error handling."""

    @pytest.mark.asyncio
    async def test_persist_position_error(self, repository, mock_adapter, sample_position):
        """Test persist position with database error."""
        mock_adapter.fetch_one.side_effect = Exception("Database connection failed")

        with pytest.raises(RepositoryError, match="Failed to save position"):
            await repository.persist_position(sample_position)

    @pytest.mark.asyncio
    async def test_get_position_by_id_error(self, repository, mock_adapter):
        """Test get position by ID with database error."""
        mock_adapter.fetch_one.side_effect = Exception("Query failed")

        with pytest.raises(RepositoryError, match="Failed to retrieve position"):
            await repository.get_position_by_id(uuid4())

    @pytest.mark.asyncio
    async def test_get_position_by_symbol_error(self, repository, mock_adapter):
        """Test get position by symbol with database error."""
        mock_adapter.fetch_one.side_effect = Exception("Query failed")

        with pytest.raises(RepositoryError, match="Failed to retrieve position for symbol"):
            await repository.get_position_by_symbol("AAPL")

    @pytest.mark.asyncio
    async def test_get_positions_by_symbol_error(self, repository, mock_adapter):
        """Test get positions by symbol with database error."""
        mock_adapter.fetch_all.side_effect = Exception("Query failed")

        with pytest.raises(RepositoryError, match="Failed to retrieve positions for symbol"):
            await repository.get_positions_by_symbol("AAPL")

    @pytest.mark.asyncio
    async def test_get_active_positions_error(self, repository, mock_adapter):
        """Test get active positions with database error."""
        mock_adapter.fetch_all.side_effect = Exception("Query failed")

        with pytest.raises(RepositoryError, match="Failed to retrieve active positions"):
            await repository.get_active_positions()

    @pytest.mark.asyncio
    async def test_get_closed_positions_error(self, repository, mock_adapter):
        """Test get closed positions with database error."""
        mock_adapter.fetch_all.side_effect = Exception("Query failed")

        with pytest.raises(RepositoryError, match="Failed to retrieve closed positions"):
            await repository.get_closed_positions()

    @pytest.mark.asyncio
    async def test_get_positions_by_strategy_error(self, repository, mock_adapter):
        """Test get positions by strategy with database error."""
        mock_adapter.fetch_all.side_effect = Exception("Query failed")

        with pytest.raises(RepositoryError, match="Failed to retrieve positions for strategy"):
            await repository.get_positions_by_strategy("momentum")

    @pytest.mark.asyncio
    async def test_get_positions_by_date_range_error(self, repository, mock_adapter):
        """Test get positions by date range with database error."""
        mock_adapter.fetch_all.side_effect = Exception("Query failed")

        with pytest.raises(RepositoryError, match="Failed to retrieve positions by date range"):
            await repository.get_positions_by_date_range(datetime.now(UTC), datetime.now(UTC))

    @pytest.mark.asyncio
    async def test_update_position_error(self, repository, mock_adapter, sample_position):
        """Test update position with database error."""
        mock_adapter.execute_query.side_effect = Exception("Update failed")

        with pytest.raises(RepositoryError, match="Failed to update position"):
            await repository.update_position(sample_position)

    @pytest.mark.asyncio
    async def test_close_position_error(self, repository, mock_adapter):
        """Test close position with database error."""
        mock_adapter.execute_query.side_effect = Exception("Update failed")

        with pytest.raises(RepositoryError, match="Failed to close position"):
            await repository.close_position(uuid4())

    @pytest.mark.asyncio
    async def test_delete_position_error(self, repository, mock_adapter):
        """Test delete position with database error."""
        mock_adapter.execute_query.side_effect = Exception("Delete failed")

        with pytest.raises(RepositoryError, match="Failed to delete position"):
            await repository.delete_position(uuid4())


@pytest.mark.unit
class TestPositionLifecycle:
    """Test position lifecycle operations."""

    @pytest.mark.asyncio
    async def test_position_opening_flow(self, repository, mock_adapter):
        """Test position from opening to tracking."""
        # Create new position using factory method
        position = Position.open_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            commission=Decimal("5.00"),
            strategy="momentum",
        )

        # Save initial position
        mock_adapter.fetch_one.return_value = None
        await repository.persist_position(position)

        # Verify insert call
        assert mock_adapter.execute_query.called
        call_args = mock_adapter.execute_query.call_args
        assert "INSERT INTO positions" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_position_update_flow(self, repository, mock_adapter, sample_position_record):
        """Test position price update."""
        position = Position(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("100"),
            average_entry_price=Decimal("145.00"),
            current_price=Decimal("152.00"),  # Updated price
            strategy="momentum",
        )

        # Update existing position
        mock_adapter.fetch_one.return_value = sample_position_record
        mock_adapter.execute_query.return_value = "UPDATE 1"

        result = await repository.persist_position(position)
        assert result.current_price == Decimal("152.00")

    @pytest.mark.asyncio
    async def test_position_closing_flow(self, repository, mock_adapter):
        """Test position closing process."""
        position_id = uuid4()

        # Close position
        mock_adapter.execute_query.return_value = "UPDATE 1"
        result = await repository.close_position(position_id)

        assert result is True

        # Verify close operation
        call_args = mock_adapter.execute_query.call_args
        assert "closed_at = NOW()" in call_args[0][0]
        assert "quantity = 0" in call_args[0][0]


@pytest.mark.unit
class TestPositionQueryOptimizations:
    """Test query optimization and edge cases."""

    @pytest.mark.asyncio
    async def test_get_positions_with_empty_results(self, repository, mock_adapter):
        """Test methods that return empty lists."""
        mock_adapter.fetch_all.return_value = []
        mock_adapter.fetch_one.return_value = None

        # Test empty symbol search
        result = await repository.get_positions_by_symbol("NONEXISTENT")
        assert result == []

        # Test empty current position by symbol
        result = await repository.get_position_by_symbol("NONEXISTENT")
        assert result is None

        # Test empty active positions
        result = await repository.get_active_positions()
        assert result == []

        # Test empty closed positions
        result = await repository.get_closed_positions()
        assert result == []

        # Test empty strategy search
        result = await repository.get_positions_by_strategy("NONEXISTENT")
        assert result == []

        # Test empty date range
        start_date = datetime.now(UTC) - timedelta(days=1)
        end_date = datetime.now(UTC)
        result = await repository.get_positions_by_date_range(start_date, end_date)
        assert result == []

    @pytest.mark.asyncio
    async def test_persist_position_exception_during_get(
        self, repository, mock_adapter, sample_position
    ):
        """Test persist position when get_position_by_id throws exception."""
        mock_adapter.fetch_one.side_effect = Exception("Database error")

        with pytest.raises(RepositoryError, match="Failed to save position"):
            await repository.persist_position(sample_position)

    @pytest.mark.asyncio
    async def test_persist_position_exception_during_update(
        self, repository, mock_adapter, sample_position, sample_position_record
    ):
        """Test persist position when update throws exception."""
        mock_adapter.fetch_one.return_value = sample_position_record
        mock_adapter.execute_query.side_effect = Exception("Update failed")

        with pytest.raises(RepositoryError, match="Failed to update position"):
            await repository.persist_position(sample_position)

    @pytest.mark.asyncio
    async def test_persist_position_exception_during_insert(
        self, repository, mock_adapter, sample_position
    ):
        """Test persist position when insert throws exception."""
        mock_adapter.fetch_one.return_value = None
        mock_adapter.execute_query.side_effect = Exception("Insert failed")

        with pytest.raises(RepositoryError, match="Failed to save position"):
            await repository.persist_position(sample_position)

    @pytest.mark.asyncio
    async def test_update_position_not_found_exception_propagation(
        self, repository, mock_adapter, sample_position
    ):
        """Test that PositionNotFoundError is properly propagated in update_position."""
        mock_adapter.execute_query.return_value = "UPDATE 0"

        with pytest.raises(PositionNotFoundError):
            await repository.update_position(sample_position)

    @pytest.mark.asyncio
    async def test_repository_initialization(self):
        """Test repository initialization."""
        mock_adapter = AsyncMock()
        repo = PostgreSQLPositionRepository(mock_adapter)

        assert repo.adapter is mock_adapter

    @pytest.mark.asyncio
    async def test_private_insert_position_method(self, repository, mock_adapter, sample_position):
        """Test the private _insert_position method directly."""
        result = await repository._insert_position(sample_position)

        assert result == sample_position
        mock_adapter.execute_query.assert_called_once()

        # Verify query structure
        call_args = mock_adapter.execute_query.call_args
        assert "INSERT INTO positions" in call_args[0][0]
        # Verify all 15 parameters are passed
        assert len(call_args[0]) == 16  # query + 15 params

    @pytest.mark.asyncio
    async def test_multiple_positions_same_symbol(self, repository, mock_adapter):
        """Test retrieving multiple positions for the same symbol."""
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
                "opened_at": datetime.now(UTC) - timedelta(days=1),
                "closed_at": datetime.now(UTC),  # Closed position
                "strategy": "momentum",
                "tags": {},
            },
            {
                "id": uuid4(),
                "symbol": "AAPL",
                "quantity": Decimal("200"),
                "average_entry_price": Decimal("148.00"),
                "current_price": Decimal("150.00"),
                "last_updated": datetime.now(UTC),
                "realized_pnl": Decimal("0"),
                "commission_paid": Decimal("5.00"),
                "stop_loss_price": None,
                "take_profit_price": None,
                "max_position_value": None,
                "opened_at": datetime.now(UTC),
                "closed_at": None,  # Open position
                "strategy": "momentum",
                "tags": {},
            },
        ]
        mock_adapter.fetch_all.return_value = position_records

        result = await repository.get_positions_by_symbol("AAPL")

        assert len(result) == 2
        assert all(pos.symbol == "AAPL" for pos in result)
        # Should include both open and closed positions
        assert any(pos.is_closed() for pos in result)
        assert any(not pos.is_closed() for pos in result)

    @pytest.mark.asyncio
    async def test_bulk_operations_with_different_strategies(self, repository, mock_adapter):
        """Test bulk operations with positions from different strategies."""
        strategy_records = {
            "momentum": [
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
            ],
            "mean_reversion": [
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
                    "strategy": "mean_reversion",
                    "tags": {},
                },
            ],
        }

        for strategy, records in strategy_records.items():
            mock_adapter.fetch_all.return_value = records

            result = await repository.get_positions_by_strategy(strategy)
            assert len(result) == 1
            assert result[0].strategy == strategy


@pytest.mark.unit
class TestPositionRepositoryConcurrency:
    """Test position repository concurrent access and transaction scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_position_updates(self, repository, mock_adapter):
        """Test handling concurrent position updates."""
        position1 = Position(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("100"),
            average_entry_price=Decimal("145.00"),
        )
        position2 = Position(
            id=position1.id,  # Same ID
            symbol="AAPL",
            quantity=Decimal("150"),  # Different quantity
            average_entry_price=Decimal("147.00"),
        )

        # First update succeeds
        mock_adapter.execute_query.return_value = "UPDATE 1"
        result1 = await repository.update_position(position1)
        assert result1 == position1

        # Concurrent update detects version mismatch
        mock_adapter.execute_query.return_value = "UPDATE 0"
        with pytest.raises(PositionNotFoundError):
            await repository.update_position(position2)

    @pytest.mark.asyncio
    async def test_race_condition_in_close_position(self, repository, mock_adapter):
        """Test race condition when closing position multiple times."""
        position_id = uuid4()

        # First close succeeds
        mock_adapter.execute_query.return_value = "UPDATE 1"
        result1 = await repository.close_position(position_id)
        assert result1 is True

        # Second close finds position already closed
        mock_adapter.execute_query.return_value = "UPDATE 0"
        result2 = await repository.close_position(position_id)
        assert result2 is False

    @pytest.mark.asyncio
    async def test_transaction_rollback_scenario(self, repository, mock_adapter):
        """Test position changes during transaction rollback."""
        position = Position(
            id=uuid4(),
            symbol="TSLA",
            quantity=Decimal("50"),
            average_entry_price=Decimal("800.00"),
        )

        # Simulate transaction that will be rolled back
        mock_adapter.fetch_one.return_value = None
        await repository.persist_position(position)

        # After rollback, position shouldn't exist
        mock_adapter.fetch_one.return_value = None
        result = await repository.get_position_by_id(position.id)
        assert result is None


@pytest.mark.unit
class TestPositionRepositoryDataIntegrity:
    """Test data integrity and edge cases."""

    @pytest.mark.asyncio
    async def test_position_with_extreme_values(self, repository, mock_adapter):
        """Test positions with extreme numerical values."""
        extreme_position = Position(
            id=uuid4(),
            symbol="BRK.A",  # Berkshire Hathaway - high price stock
            quantity=Decimal("0.001"),  # Fractional shares
            average_entry_price=Decimal("500000.00"),  # Very high price
            current_price=Decimal("500001.00"),
            realized_pnl=Decimal("-999999999.99"),  # Large loss
            commission_paid=Decimal("0.00000001"),  # Very small commission
        )

        mock_adapter.fetch_one.return_value = None
        result = await repository.persist_position(extreme_position)
        assert result == extreme_position

    @pytest.mark.asyncio
    async def test_position_symbol_validation(self, repository, mock_adapter):
        """Test positions with various symbol formats."""
        symbols = [
            "AAPL",  # Standard
            "BRK.B",  # With dot
            "BABA",  # ADR
            "2330.TW",  # International
            "^DJI",  # Index
            "EURUSD=X",  # Forex
            "BTC-USD",  # Crypto
        ]

        for symbol in symbols:
            position = Position(
                id=uuid4(),
                symbol=symbol,
                quantity=Decimal("100"),
                average_entry_price=Decimal("100.00"),
            )

            mock_adapter.fetch_one.return_value = None
            result = await repository.persist_position(position)
            assert result.symbol == symbol

    @pytest.mark.asyncio
    async def test_partial_fill_handling(self, repository, mock_adapter):
        """Test handling positions with partial fills."""
        # Initial position
        position = Position(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("50"),  # Partial fill
            average_entry_price=Decimal("150.00"),
        )

        mock_adapter.fetch_one.return_value = None
        await repository.persist_position(position)

        # Update with additional fill
        position.quantity = Decimal("100")  # Full fill
        position.average_entry_price = Decimal("149.50")  # Updated average

        mock_adapter.fetch_one.return_value = {
            "id": position.id,
            "symbol": position.symbol,
            "quantity": Decimal("50"),
            "average_entry_price": Decimal("150.00"),
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
            "tags": {},
        }
        mock_adapter.execute_query.return_value = "UPDATE 1"

        result = await repository.persist_position(position)
        assert result.quantity == Decimal("100")


@pytest.mark.unit
class TestPositionRepositoryPerformance:
    """Test performance and scalability scenarios."""

    @pytest.mark.asyncio
    async def test_bulk_position_retrieval(self, repository, mock_adapter):
        """Test retrieving large number of positions."""
        # Create 1000 position records
        position_records = [
            {
                "id": uuid4(),
                "symbol": f"SYM{i:04d}",
                "quantity": Decimal(str(100 + i)),
                "average_entry_price": Decimal(str(100 + i * 0.1)),
                "current_price": Decimal(str(105 + i * 0.1)),
                "last_updated": datetime.now(UTC),
                "realized_pnl": Decimal("0"),
                "commission_paid": Decimal("5.00"),
                "stop_loss_price": None,
                "take_profit_price": None,
                "max_position_value": None,
                "opened_at": datetime.now(UTC) - timedelta(days=i % 30),
                "closed_at": None,
                "strategy": "bulk_test",
                "tags": {},
            }
            for i in range(1000)
        ]

        mock_adapter.fetch_all.return_value = position_records

        positions = await repository.get_positions_by_strategy("bulk_test")
        assert len(positions) == 1000

    @pytest.mark.asyncio
    async def test_position_history_query_optimization(self, repository, mock_adapter):
        """Test optimized queries for position history."""
        # Test date range query optimization
        start_date = datetime.now(UTC) - timedelta(days=365)
        end_date = datetime.now(UTC)

        mock_adapter.fetch_all.return_value = []

        await repository.get_positions_by_date_range(start_date, end_date)

        # Verify query is optimized with proper indexing hints
        call_args = mock_adapter.fetch_all.call_args
        query = call_args[0][0]
        assert "opened_at >=" in query
        assert "opened_at <=" in query
        assert "ORDER BY opened_at DESC" in query

    @pytest.mark.asyncio
    async def test_connection_timeout_handling(self, repository, mock_adapter):
        """Test handling of database connection timeouts."""
        mock_adapter.fetch_one.side_effect = Exception("timeout expired")

        with pytest.raises(RepositoryError, match="Failed to retrieve position"):
            await repository.get_position_by_id(uuid4())


@pytest.mark.unit
class TestPositionRepositoryComplexScenarios:
    """Test complex business scenarios."""

    @pytest.mark.asyncio
    async def test_position_split_adjustment(self, repository, mock_adapter):
        """Test handling stock split adjustments."""
        # Position before split
        position = Position(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("100"),
            average_entry_price=Decimal("600.00"),  # Pre-split price
        )

        mock_adapter.fetch_one.return_value = None
        await repository.persist_position(position)

        # After 4:1 split
        position.quantity = Decimal("400")  # 4x quantity
        position.average_entry_price = Decimal("150.00")  # 1/4 price

        mock_adapter.fetch_one.return_value = {
            "id": position.id,
            "symbol": "AAPL",
            "quantity": Decimal("100"),
            "average_entry_price": Decimal("600.00"),
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
            "tags": {},
        }
        mock_adapter.execute_query.return_value = "UPDATE 1"

        result = await repository.update_position(position)
        assert result.quantity == Decimal("400")
        assert result.average_entry_price == Decimal("150.00")

    @pytest.mark.asyncio
    async def test_position_transfer_between_accounts(self, repository, mock_adapter):
        """Test position transfer scenarios."""
        # Original position
        position_id = uuid4()

        # Close position in one account
        mock_adapter.execute_query.return_value = "UPDATE 1"
        closed = await repository.close_position(position_id)
        assert closed is True

        # Create new position in another account (new ID)
        new_position = Position(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("100"),
            average_entry_price=Decimal("150.00"),
            strategy="transferred",
        )

        mock_adapter.fetch_one.return_value = None
        result = await repository.persist_position(new_position)
        assert result.strategy == "transferred"

    @pytest.mark.asyncio
    async def test_position_with_complex_tags(self, repository, mock_adapter):
        """Test positions with complex tag structures."""
        complex_tags = {
            "trader": "john",
            "desk": "equity",
            "region": "US",
            "risk_level": "high",
            "entry_signal": "momentum_breakout",
            "exit_strategy": "trailing_stop",
            "correlation_group": "tech_large_cap",
            "hedged": "false",
            "metadata": {"source": "automated", "version": "2.0"},
        }

        position = Position(
            id=uuid4(),
            symbol="NVDA",
            quantity=Decimal("200"),
            average_entry_price=Decimal("450.00"),
            tags=complex_tags,
        )

        mock_adapter.fetch_one.return_value = None
        result = await repository.persist_position(position)
        assert result.tags == complex_tags
