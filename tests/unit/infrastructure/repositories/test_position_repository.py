"""
Unit tests for PostgreSQL Position Repository Implementation.

Tests the concrete implementation of IPositionRepository including CRUD operations,
position lifecycle management, and entity mapping.
"""

# Standard library imports
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock
from uuid import uuid4

# Third-party imports
import pytest

# Local imports
from src.application.interfaces.exceptions import PositionNotFoundError, RepositoryError
from src.domain.entities.position import Position, PositionSide
from src.infrastructure.repositories.position_repository import PostgreSQLPositionRepository


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
    """Position repository with mocked adapter."""
    return PostgreSQLPositionRepository(mock_adapter)


@pytest.fixture
def sample_position():
    """Sample position entity for testing."""
    return Position(
        id=uuid4(),
        symbol="AAPL",
        quantity=Decimal("100"),
        side=PositionSide.LONG,
        average_price=Decimal("145.00"),
        current_price=Decimal("150.00"),
        opened_at=datetime.now(UTC),
        strategy="test_strategy",
    )


@pytest.fixture
def sample_position_record():
    """Sample position database record."""
    position_id = uuid4()
    return {
        "id": position_id,
        "symbol": "AAPL",
        "quantity": Decimal("100"),
        "side": "long",
        "average_price": Decimal("145.00"),
        "current_price": Decimal("150.00"),
        "unrealized_pnl": Decimal("500.00"),
        "realized_pnl": Decimal("0.00"),
        "opened_at": datetime.now(UTC),
        "closed_at": None,
        "strategy": "test_strategy",
        "tags": {},
    }


@pytest.mark.unit
class TestPositionRepositoryCRUD:
    """Test position CRUD operations."""

    async def test_save_position_success(self, repository, mock_adapter, sample_position):
        """Test saving a position successfully."""
        mock_adapter.fetch_one.return_value = None  # Position doesn't exist

        result = await repository.save_position(sample_position)

        assert result == sample_position
        mock_adapter.execute_query.assert_called_once()

        # Verify insert was called with correct parameters
        call_args = mock_adapter.execute_query.call_args
        assert "INSERT INTO positions" in call_args[0][0]
        assert call_args[0][1] == sample_position.id
        assert call_args[0][2] == sample_position.symbol

    async def test_get_position_by_id_found(self, repository, mock_adapter, sample_position_record):
        """Test getting position by ID when position exists."""
        mock_adapter.fetch_one.return_value = sample_position_record

        result = await repository.get_position_by_id(sample_position_record["id"])

        assert result is not None
        assert result.id == sample_position_record["id"]
        assert result.symbol == sample_position_record["symbol"]
        assert result.side == PositionSide(sample_position_record["side"])

        mock_adapter.fetch_one.assert_called_once()

    async def test_get_position_by_id_not_found(self, repository, mock_adapter):
        """Test getting position by ID when position doesn't exist."""
        mock_adapter.fetch_one.return_value = None
        position_id = uuid4()

        result = await repository.get_position_by_id(position_id)

        assert result is None

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

    async def test_get_active_positions(self, repository, mock_adapter, sample_position_record):
        """Test getting active positions."""
        mock_adapter.fetch_all.return_value = [sample_position_record]

        result = await repository.get_active_positions()

        assert len(result) == 1
        assert not result[0].is_closed()

        # Verify query filters for open positions
        call_args = mock_adapter.fetch_all.call_args
        assert "WHERE closed_at IS NULL" in call_args[0][0]

    async def test_close_position_success(self, repository, mock_adapter):
        """Test successful position closure."""
        position_id = uuid4()
        mock_adapter.execute_query.return_value = "UPDATE 1"

        result = await repository.close_position(position_id)

        assert result is True

        # Verify update query sets closed_at
        call_args = mock_adapter.execute_query.call_args
        assert "UPDATE positions SET" in call_args[0][0]
        assert "closed_at = %s" in call_args[0][0]
        assert "WHERE id = %s" in call_args[0][0]

    async def test_update_position_success(self, repository, mock_adapter, sample_position):
        """Test successful position update."""
        mock_adapter.execute_query.return_value = "UPDATE 1"

        result = await repository.update_position(sample_position)

        assert result == sample_position
        mock_adapter.execute_query.assert_called_once()

    async def test_update_position_not_found(self, repository, mock_adapter, sample_position):
        """Test update position when position doesn't exist."""
        mock_adapter.execute_query.return_value = "UPDATE 0"

        with pytest.raises(PositionNotFoundError):
            await repository.update_position(sample_position)


@pytest.mark.unit
class TestPositionEntityMapping:
    """Test entity mapping between domain objects and database records."""

    def test_map_record_to_position_complete(self, repository, sample_position_record):
        """Test mapping complete database record to position entity."""
        position = repository._map_record_to_position(sample_position_record)

        assert position.id == sample_position_record["id"]
        assert position.symbol == sample_position_record["symbol"]
        assert position.quantity == sample_position_record["quantity"]
        assert position.side == PositionSide(sample_position_record["side"])
        assert position.average_price == sample_position_record["average_price"]
        assert position.current_price == sample_position_record["current_price"]
        assert position.opened_at == sample_position_record["opened_at"]
        assert position.closed_at == sample_position_record["closed_at"]
        assert position.strategy == sample_position_record["strategy"]

    def test_map_record_to_position_with_null_tags(self, repository, sample_position_record):
        """Test mapping record with null tags field."""
        sample_position_record["tags"] = None

        position = repository._map_record_to_position(sample_position_record)

        assert position.tags == {}


@pytest.mark.unit
class TestPositionErrorHandling:
    """Test position repository error handling."""

    async def test_save_position_adapter_error(self, repository, mock_adapter, sample_position):
        """Test save position with adapter error."""
        mock_adapter.fetch_one.side_effect = Exception("Database error")

        with pytest.raises(RepositoryError, match="Failed to save position"):
            await repository.save_position(sample_position)

    async def test_get_position_adapter_error(self, repository, mock_adapter):
        """Test get position with adapter error."""
        mock_adapter.fetch_one.side_effect = Exception("Database error")

        with pytest.raises(RepositoryError, match="Failed to retrieve position"):
            await repository.get_position_by_id(uuid4())


@pytest.mark.unit
class TestPositionQueries:
    """Test position-specific query operations."""

    async def test_get_positions_by_strategy(
        self, repository, mock_adapter, sample_position_record
    ):
        """Test getting positions by strategy."""
        mock_adapter.fetch_all.return_value = [sample_position_record]

        result = await repository.get_positions_by_strategy("test_strategy")

        assert len(result) == 1
        assert result[0].strategy == "test_strategy"

        # Verify query
        call_args = mock_adapter.fetch_all.call_args
        assert "WHERE strategy = %s" in call_args[0][0]
        assert call_args[0][1] == "test_strategy"

    async def test_get_closed_positions(self, repository, mock_adapter):
        """Test getting closed positions."""
        closed_position_record = {
            "id": uuid4(),
            "symbol": "AAPL",
            "quantity": Decimal("100"),
            "side": "long",
            "average_price": Decimal("145.00"),
            "current_price": Decimal("150.00"),
            "unrealized_pnl": Decimal("0.00"),
            "realized_pnl": Decimal("500.00"),
            "opened_at": datetime.now(UTC),
            "closed_at": datetime.now(UTC),  # Position is closed
            "strategy": "test_strategy",
            "tags": {},
        }
        mock_adapter.fetch_all.return_value = [closed_position_record]

        result = await repository.get_closed_positions()

        assert len(result) == 1
        assert result[0].is_closed()

        # Verify query filters for closed positions
        call_args = mock_adapter.fetch_all.call_args
        assert "WHERE closed_at IS NOT NULL" in call_args[0][0]

    async def test_get_positions_by_date_range(
        self, repository, mock_adapter, sample_position_record
    ):
        """Test getting positions by date range."""
        start_date = datetime(2023, 1, 1, tzinfo=UTC)
        end_date = datetime(2023, 12, 31, tzinfo=UTC)
        mock_adapter.fetch_all.return_value = [sample_position_record]

        result = await repository.get_positions_by_date_range(start_date, end_date)

        assert len(result) == 1

        # Verify query
        call_args = mock_adapter.fetch_all.call_args
        assert "WHERE opened_at >= %s AND opened_at <= %s" in call_args[0][0]
        assert call_args[0][1] == start_date
        assert call_args[0][2] == end_date
