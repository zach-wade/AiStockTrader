"""
Working tests for repository classes.

Tests the actual repository methods with proper parameter names and mocked database calls.
"""

# Standard library imports
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock
from uuid import uuid4

# Third-party imports
import pytest

# Local imports
from src.application.interfaces.exceptions import (
    OrderNotFoundError,
    PositionNotFoundError,
    RepositoryError,
)
from src.domain.entities.order import (
    Order,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from src.domain.entities.position import Position
from src.infrastructure.repositories.order_repository import PostgreSQLOrderRepository
from src.infrastructure.repositories.position_repository import PostgreSQLPositionRepository


@pytest.fixture
def mock_adapter():
    """Create a mock database adapter."""
    adapter = AsyncMock()
    adapter.execute_query.return_value = "EXECUTE 1"
    adapter.fetch_one.return_value = None
    adapter.fetch_all.return_value = []
    adapter.fetch_values.return_value = []
    return adapter


@pytest.fixture
def order_repo(mock_adapter):
    """Create OrderRepository with mock adapter."""
    return PostgreSQLOrderRepository(mock_adapter)


@pytest.fixture
def position_repo(mock_adapter):
    """Create PositionRepository with mock adapter."""
    return PostgreSQLPositionRepository(mock_adapter)


@pytest.fixture
def sample_order():
    """Create a sample order for testing."""
    request = OrderRequest(
        symbol="AAPL",
        quantity=Decimal("100"),
        side=OrderSide.BUY,
        limit_price=Decimal("150.00"),
        reason="Test order",
    )
    return Order.create_limit_order(request)


@pytest.fixture
def sample_position():
    """Create a sample position for testing."""
    return Position.open_position(
        symbol="AAPL",
        quantity=Decimal("100"),
        entry_price=Decimal("145.00"),
        strategy="test_strategy",
    )


@pytest.fixture
def order_db_record():
    """Sample order database record."""
    return {
        "id": uuid4(),
        "symbol": "AAPL",
        "side": "buy",
        "order_type": "limit",
        "status": "pending",
        "quantity": Decimal("100"),
        "limit_price": Decimal("150.00"),
        "stop_price": None,
        "time_in_force": "day",
        "broker_order_id": None,
        "filled_quantity": Decimal("0"),
        "average_fill_price": None,
        "created_at": datetime.now(UTC),
        "submitted_at": None,
        "filled_at": None,
        "cancelled_at": None,
        "reason": "Test order",
        "tags": {},
    }


@pytest.fixture
def position_db_record():
    """Sample position database record."""
    return {
        "id": uuid4(),
        "symbol": "AAPL",
        "quantity": Decimal("100"),
        "average_entry_price": Decimal("145.00"),
        "current_price": Decimal("150.00"),
        "last_updated": datetime.now(UTC),
        "realized_pnl": Decimal("0.00"),
        "commission_paid": Decimal("0.00"),
        "stop_loss_price": None,
        "take_profit_price": None,
        "max_position_value": None,
        "opened_at": datetime.now(UTC),
        "closed_at": None,
        "strategy": "test_strategy",
        "tags": {},
    }


class TestPostgreSQLOrderRepository:
    """Test PostgreSQLOrderRepository methods."""

    def test_init(self, mock_adapter):
        """Test repository initialization."""
        repo = PostgreSQLOrderRepository(mock_adapter)
        assert repo.adapter == mock_adapter

    @pytest.mark.asyncio
    async def test_save_order_new(self, order_repo, sample_order, mock_adapter):
        """Test saving a new order."""
        # Mock that order doesn't exist
        mock_adapter.fetch_one.return_value = None
        mock_adapter.execute_query.return_value = "EXECUTE 1"

        result = await order_repo.save_order(sample_order)

        assert result == sample_order
        # Verify insert was called
        mock_adapter.execute_query.assert_called()
        call_args = mock_adapter.execute_query.call_args
        assert "INSERT INTO orders" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_save_order_existing(
        self, order_repo, sample_order, mock_adapter, order_db_record
    ):
        """Test saving an existing order (update)."""
        # Mock that order exists
        mock_adapter.fetch_one.return_value = order_db_record
        mock_adapter.execute_query.return_value = "UPDATE 1"

        result = await order_repo.save_order(sample_order)

        assert result == sample_order
        # Verify update was called
        mock_adapter.execute_query.assert_called()
        call_args = mock_adapter.execute_query.call_args
        assert "UPDATE orders SET" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_save_order_error(self, order_repo, sample_order, mock_adapter):
        """Test save order with database error."""
        mock_adapter.fetch_one.side_effect = Exception("Database error")

        with pytest.raises(RepositoryError, match="Failed to save order"):
            await order_repo.save_order(sample_order)

    @pytest.mark.asyncio
    async def test_get_order_by_id_found(self, order_repo, mock_adapter, order_db_record):
        """Test getting order by ID when found."""
        mock_adapter.fetch_one.return_value = order_db_record

        result = await order_repo.get_order_by_id(order_db_record["id"])

        assert result is not None
        assert result.id == order_db_record["id"]
        assert result.symbol == order_db_record["symbol"]
        assert result.side == OrderSide(order_db_record["side"])
        assert result.quantity == order_db_record["quantity"]

    @pytest.mark.asyncio
    async def test_get_order_by_id_not_found(self, order_repo, mock_adapter):
        """Test getting order by ID when not found."""
        mock_adapter.fetch_one.return_value = None
        order_id = uuid4()

        result = await order_repo.get_order_by_id(order_id)

        assert result is None
        mock_adapter.fetch_one.assert_called_once()
        call_args = mock_adapter.fetch_one.call_args
        assert "WHERE id = %s" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_order_by_id_error(self, order_repo, mock_adapter):
        """Test get order by ID with database error."""
        mock_adapter.fetch_one.side_effect = Exception("Database error")
        order_id = uuid4()

        with pytest.raises(RepositoryError, match="Failed to retrieve order"):
            await order_repo.get_order_by_id(order_id)

    @pytest.mark.asyncio
    async def test_get_orders_by_symbol(self, order_repo, mock_adapter, order_db_record):
        """Test getting orders by symbol."""
        mock_adapter.fetch_all.return_value = [order_db_record]

        result = await order_repo.get_orders_by_symbol("AAPL")

        assert len(result) == 1
        assert result[0].symbol == "AAPL"
        mock_adapter.fetch_all.assert_called_once()
        call_args = mock_adapter.fetch_all.call_args
        assert "WHERE symbol = %s" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_orders_by_status(self, order_repo, mock_adapter, order_db_record):
        """Test getting orders by status."""
        mock_adapter.fetch_all.return_value = [order_db_record]

        result = await order_repo.get_orders_by_status(OrderStatus.PENDING)

        assert len(result) == 1
        assert result[0].status == OrderStatus.PENDING
        mock_adapter.fetch_all.assert_called_once()
        call_args = mock_adapter.fetch_all.call_args
        assert "WHERE status = %s" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_active_orders(self, order_repo, mock_adapter, order_db_record):
        """Test getting active orders."""
        mock_adapter.fetch_all.return_value = [order_db_record]

        result = await order_repo.get_active_orders()

        assert len(result) == 1
        mock_adapter.fetch_all.assert_called_once()
        call_args = mock_adapter.fetch_all.call_args
        assert "WHERE status IN" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_orders_by_date_range(self, order_repo, mock_adapter, order_db_record):
        """Test getting orders by date range."""
        start_date = datetime.now(UTC)
        end_date = datetime.now(UTC)
        mock_adapter.fetch_all.return_value = [order_db_record]

        result = await order_repo.get_orders_by_date_range(start_date, end_date)

        assert len(result) == 1
        mock_adapter.fetch_all.assert_called_once()
        call_args = mock_adapter.fetch_all.call_args
        assert "WHERE created_at >= %s AND created_at <= %s" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_update_order_success(self, order_repo, sample_order, mock_adapter):
        """Test successful order update."""
        mock_adapter.execute_query.return_value = "UPDATE 1"

        result = await order_repo.update_order(sample_order)

        assert result == sample_order
        mock_adapter.execute_query.assert_called_once()
        call_args = mock_adapter.execute_query.call_args
        assert "UPDATE orders SET" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_update_order_not_found(self, order_repo, sample_order, mock_adapter):
        """Test update order when not found."""
        mock_adapter.execute_query.return_value = "UPDATE 0"

        with pytest.raises(OrderNotFoundError):
            await order_repo.update_order(sample_order)

    @pytest.mark.asyncio
    async def test_delete_order_success(self, order_repo, mock_adapter):
        """Test successful order deletion."""
        mock_adapter.execute_query.return_value = "DELETE 1"
        order_id = uuid4()

        result = await order_repo.delete_order(order_id)

        assert result is True
        mock_adapter.execute_query.assert_called_once()
        call_args = mock_adapter.execute_query.call_args
        assert "DELETE FROM orders WHERE id = %s" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_delete_order_not_found(self, order_repo, mock_adapter):
        """Test delete order when not found."""
        mock_adapter.execute_query.return_value = "DELETE 0"
        order_id = uuid4()

        result = await order_repo.delete_order(order_id)

        assert result is False

    @pytest.mark.asyncio
    async def test_get_orders_by_broker_id(self, order_repo, mock_adapter, order_db_record):
        """Test getting orders by broker ID."""
        order_db_record["broker_order_id"] = "BROKER123"
        mock_adapter.fetch_all.return_value = [order_db_record]

        result = await order_repo.get_orders_by_broker_id("BROKER123")

        assert len(result) == 1
        assert result[0].broker_order_id == "BROKER123"
        mock_adapter.fetch_all.assert_called_once()
        call_args = mock_adapter.fetch_all.call_args
        assert "WHERE broker_order_id = %s" in call_args[0][0]

    def test_map_record_to_order(self, order_repo, order_db_record):
        """Test mapping database record to Order entity."""
        result = order_repo._map_record_to_order(order_db_record)

        assert isinstance(result, Order)
        assert result.id == order_db_record["id"]
        assert result.symbol == order_db_record["symbol"]
        assert result.side == OrderSide(order_db_record["side"])
        assert result.order_type == OrderType(order_db_record["order_type"])
        assert result.status == OrderStatus(order_db_record["status"])
        assert result.quantity == order_db_record["quantity"]
        assert result.limit_price == order_db_record["limit_price"]
        assert result.time_in_force == TimeInForce(order_db_record["time_in_force"])


class TestPostgreSQLPositionRepository:
    """Test PostgreSQLPositionRepository methods."""

    def test_init(self, mock_adapter):
        """Test repository initialization."""
        repo = PostgreSQLPositionRepository(mock_adapter)
        assert repo.adapter == mock_adapter

    @pytest.mark.asyncio
    async def test_persist_position_new(self, position_repo, sample_position, mock_adapter):
        """Test persisting a new position."""
        # Mock that position doesn't exist
        mock_adapter.fetch_one.return_value = None
        mock_adapter.execute_query.return_value = "EXECUTE 1"

        result = await position_repo.persist_position(sample_position)

        assert result == sample_position
        # Verify insert was called
        mock_adapter.execute_query.assert_called()
        call_args = mock_adapter.execute_query.call_args
        assert "INSERT INTO positions" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_persist_position_existing(
        self, position_repo, sample_position, mock_adapter, position_db_record
    ):
        """Test persisting an existing position (update)."""
        # Mock that position exists
        mock_adapter.fetch_one.return_value = position_db_record
        mock_adapter.execute_query.return_value = "UPDATE 1"

        result = await position_repo.persist_position(sample_position)

        assert result == sample_position
        # Verify update was called
        mock_adapter.execute_query.assert_called()
        call_args = mock_adapter.execute_query.call_args
        assert "UPDATE positions SET" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_persist_position_error(self, position_repo, sample_position, mock_adapter):
        """Test persist position with database error."""
        mock_adapter.fetch_one.side_effect = Exception("Database error")

        with pytest.raises(RepositoryError, match="Failed to save position"):
            await position_repo.persist_position(sample_position)

    @pytest.mark.asyncio
    async def test_get_position_by_id_found(self, position_repo, mock_adapter, position_db_record):
        """Test getting position by ID when found."""
        mock_adapter.fetch_one.return_value = position_db_record

        result = await position_repo.get_position_by_id(position_db_record["id"])

        assert result is not None
        assert result.id == position_db_record["id"]
        assert result.symbol == position_db_record["symbol"]
        assert result.quantity == position_db_record["quantity"]

    @pytest.mark.asyncio
    async def test_get_position_by_id_not_found(self, position_repo, mock_adapter):
        """Test getting position by ID when not found."""
        mock_adapter.fetch_one.return_value = None
        position_id = uuid4()

        result = await position_repo.get_position_by_id(position_id)

        assert result is None
        mock_adapter.fetch_one.assert_called_once()
        call_args = mock_adapter.fetch_one.call_args
        assert "WHERE id = %s" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_position_by_symbol(self, position_repo, mock_adapter, position_db_record):
        """Test getting current position by symbol."""
        mock_adapter.fetch_one.return_value = position_db_record

        result = await position_repo.get_position_by_symbol("AAPL")

        assert result is not None
        assert result.symbol == "AAPL"
        mock_adapter.fetch_one.assert_called_once()
        call_args = mock_adapter.fetch_one.call_args
        assert "WHERE symbol = %s AND closed_at IS NULL" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_positions_by_symbol(self, position_repo, mock_adapter, position_db_record):
        """Test getting all positions by symbol."""
        mock_adapter.fetch_all.return_value = [position_db_record]

        result = await position_repo.get_positions_by_symbol("AAPL")

        assert len(result) == 1
        assert result[0].symbol == "AAPL"
        mock_adapter.fetch_all.assert_called_once()
        call_args = mock_adapter.fetch_all.call_args
        assert "WHERE symbol = %s" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_active_positions(self, position_repo, mock_adapter, position_db_record):
        """Test getting active positions."""
        mock_adapter.fetch_all.return_value = [position_db_record]

        result = await position_repo.get_active_positions()

        assert len(result) == 1
        mock_adapter.fetch_all.assert_called_once()
        call_args = mock_adapter.fetch_all.call_args
        assert "WHERE closed_at IS NULL" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_closed_positions(self, position_repo, mock_adapter, position_db_record):
        """Test getting closed positions."""
        position_db_record["closed_at"] = datetime.now(UTC)
        mock_adapter.fetch_all.return_value = [position_db_record]

        result = await position_repo.get_closed_positions()

        assert len(result) == 1
        mock_adapter.fetch_all.assert_called_once()
        call_args = mock_adapter.fetch_all.call_args
        assert "WHERE closed_at IS NOT NULL" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_positions_by_strategy(self, position_repo, mock_adapter, position_db_record):
        """Test getting positions by strategy."""
        mock_adapter.fetch_all.return_value = [position_db_record]

        result = await position_repo.get_positions_by_strategy("test_strategy")

        assert len(result) == 1
        assert result[0].strategy == "test_strategy"
        mock_adapter.fetch_all.assert_called_once()
        call_args = mock_adapter.fetch_all.call_args
        assert "WHERE strategy = %s" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_positions_by_date_range(
        self, position_repo, mock_adapter, position_db_record
    ):
        """Test getting positions by date range."""
        start_date = datetime.now(UTC)
        end_date = datetime.now(UTC)
        mock_adapter.fetch_all.return_value = [position_db_record]

        result = await position_repo.get_positions_by_date_range(start_date, end_date)

        assert len(result) == 1
        mock_adapter.fetch_all.assert_called_once()
        call_args = mock_adapter.fetch_all.call_args
        assert "WHERE opened_at >= %s AND opened_at <= %s" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_update_position_success(self, position_repo, sample_position, mock_adapter):
        """Test successful position update."""
        mock_adapter.execute_query.return_value = "UPDATE 1"

        result = await position_repo.update_position(sample_position)

        assert result == sample_position
        mock_adapter.execute_query.assert_called_once()
        call_args = mock_adapter.execute_query.call_args
        assert "UPDATE positions SET" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_update_position_not_found(self, position_repo, sample_position, mock_adapter):
        """Test update position when not found."""
        mock_adapter.execute_query.return_value = "UPDATE 0"

        with pytest.raises(PositionNotFoundError):
            await position_repo.update_position(sample_position)

    @pytest.mark.asyncio
    async def test_close_position_success(self, position_repo, mock_adapter):
        """Test successful position close."""
        mock_adapter.execute_query.return_value = "UPDATE 1"
        position_id = uuid4()

        result = await position_repo.close_position(position_id)

        assert result is True
        mock_adapter.execute_query.assert_called_once()
        call_args = mock_adapter.execute_query.call_args
        assert "UPDATE positions SET" in call_args[0][0]
        assert "closed_at = NOW()" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_close_position_not_found(self, position_repo, mock_adapter):
        """Test close position when not found."""
        mock_adapter.execute_query.return_value = "UPDATE 0"
        position_id = uuid4()

        result = await position_repo.close_position(position_id)

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_position_success(self, position_repo, mock_adapter):
        """Test successful position deletion."""
        mock_adapter.execute_query.return_value = "DELETE 1"
        position_id = uuid4()

        result = await position_repo.delete_position(position_id)

        assert result is True
        mock_adapter.execute_query.assert_called_once()
        call_args = mock_adapter.execute_query.call_args
        assert "DELETE FROM positions WHERE id = %s" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_delete_position_not_found(self, position_repo, mock_adapter):
        """Test delete position when not found."""
        mock_adapter.execute_query.return_value = "DELETE 0"
        position_id = uuid4()

        result = await position_repo.delete_position(position_id)

        assert result is False

    def test_map_record_to_position(self, position_repo, position_db_record):
        """Test mapping database record to Position entity."""
        result = position_repo._map_record_to_position(position_db_record)

        assert isinstance(result, Position)
        assert result.id == position_db_record["id"]
        assert result.symbol == position_db_record["symbol"]
        assert result.quantity == position_db_record["quantity"]
        assert result.average_entry_price == position_db_record["average_entry_price"]
        assert result.current_price == position_db_record["current_price"]
        assert result.realized_pnl == position_db_record["realized_pnl"]
        assert result.commission_paid == position_db_record["commission_paid"]
        assert result.strategy == position_db_record["strategy"]
