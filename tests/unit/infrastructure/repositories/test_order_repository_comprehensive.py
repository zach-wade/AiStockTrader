"""
Comprehensive unit tests for PostgreSQL Order Repository Implementation.

Tests the concrete implementation of IOrderRepository including CRUD operations,
order lifecycle management, status transitions, and entity mapping with full coverage.
"""

# Standard library imports
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock
from uuid import uuid4

# Third-party imports
import pytest

# Local imports
from src.application.interfaces.exceptions import OrderNotFoundError, RepositoryError
from src.domain.entities.order import Order, OrderSide, OrderStatus, OrderType, TimeInForce
from src.infrastructure.repositories.order_repository import PostgreSQLOrderRepository


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
    """Order repository with mocked adapter."""
    return PostgreSQLOrderRepository(mock_adapter)


@pytest.fixture
def sample_order():
    """Sample order entity for testing."""
    return Order(
        id=uuid4(),
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        status=OrderStatus.PENDING,
        quantity=Decimal("100"),
        limit_price=Decimal("150.00"),
        stop_price=None,
        time_in_force=TimeInForce.DAY,
        broker_order_id="BROKER123",
        filled_quantity=Decimal("0"),
        average_fill_price=None,
        created_at=datetime.now(UTC),
        submitted_at=None,
        filled_at=None,
        cancelled_at=None,
        reason=None,
        tags={"strategy": "momentum"},
    )


@pytest.fixture
def sample_order_record():
    """Sample order database record."""
    order_id = uuid4()
    return {
        "id": order_id,
        "symbol": "AAPL",
        "side": "buy",
        "order_type": "limit",
        "status": "pending",
        "quantity": Decimal("100"),
        "limit_price": Decimal("150.00"),
        "stop_price": None,
        "time_in_force": "day",
        "broker_order_id": "BROKER123",
        "filled_quantity": Decimal("0"),
        "average_fill_price": None,
        "created_at": datetime.now(UTC),
        "submitted_at": None,
        "filled_at": None,
        "cancelled_at": None,
        "reason": None,
        "tags": {"strategy": "momentum"},
    }


@pytest.mark.unit
class TestOrderRepositoryCRUD:
    """Test order CRUD operations."""

    async def test_save_order_insert_new(self, repository, mock_adapter, sample_order):
        """Test saving a new order."""
        mock_adapter.fetch_one.return_value = None  # Order doesn't exist

        result = await repository.save_order(sample_order)

        assert result == sample_order
        assert mock_adapter.execute_query.called

        # Verify insert query structure
        call_args = mock_adapter.execute_query.call_args
        assert "INSERT INTO orders" in call_args[0][0]
        assert call_args[0][1] == sample_order.id
        assert call_args[0][2] == sample_order.symbol
        assert call_args[0][3] == sample_order.side

    async def test_save_order_update_existing(
        self, repository, mock_adapter, sample_order, sample_order_record
    ):
        """Test updating an existing order."""
        mock_adapter.fetch_one.return_value = sample_order_record

        result = await repository.save_order(sample_order)

        assert result == sample_order
        # Should call update instead of insert
        call_args = mock_adapter.execute_query.call_args
        assert "UPDATE orders SET" in call_args[0][0]

    async def test_insert_order_all_fields(self, repository, mock_adapter, sample_order):
        """Test inserting order with all fields."""
        await repository._insert_order(sample_order)

        # Verify all fields are in insert query
        call_args = mock_adapter.execute_query.call_args
        query = call_args[0][0]

        assert "INSERT INTO orders" in query
        assert "id" in query and "symbol" in query and "side" in query
        assert "order_type" in query and "status" in query and "quantity" in query
        assert "limit_price" in query and "stop_price" in query
        assert "time_in_force" in query and "broker_order_id" in query
        assert "filled_quantity" in query and "average_fill_price" in query
        assert "created_at" in query and "submitted_at" in query
        assert "filled_at" in query and "cancelled_at" in query
        assert "reason" in query and "tags" in query

        # Verify parameter count
        params = call_args[0][1:]
        assert len(params) == 18  # All order fields

    async def test_get_order_by_id_found(self, repository, mock_adapter, sample_order_record):
        """Test getting order by ID when it exists."""
        mock_adapter.fetch_one.return_value = sample_order_record

        result = await repository.get_order_by_id(sample_order_record["id"])

        assert result is not None
        assert result.id == sample_order_record["id"]
        assert result.symbol == sample_order_record["symbol"]
        assert result.side == OrderSide(sample_order_record["side"])
        assert result.order_type == OrderType(sample_order_record["order_type"])
        assert result.status == OrderStatus(sample_order_record["status"])

    async def test_get_order_by_id_not_found(self, repository, mock_adapter):
        """Test getting order by ID when it doesn't exist."""
        mock_adapter.fetch_one.return_value = None

        result = await repository.get_order_by_id(uuid4())

        assert result is None

    async def test_get_orders_by_symbol(self, repository, mock_adapter, sample_order_record):
        """Test getting orders by symbol."""
        mock_adapter.fetch_all.return_value = [sample_order_record, sample_order_record]

        result = await repository.get_orders_by_symbol("AAPL")

        assert len(result) == 2
        assert all(order.symbol == "AAPL" for order in result)

        # Verify query
        call_args = mock_adapter.fetch_all.call_args
        assert "WHERE symbol = %s" in call_args[0][0]
        assert call_args[0][1] == "AAPL"

    async def test_get_orders_by_status(self, repository, mock_adapter, sample_order_record):
        """Test getting orders by status."""
        mock_adapter.fetch_all.return_value = [sample_order_record]

        result = await repository.get_orders_by_status(OrderStatus.PENDING)

        assert len(result) == 1
        assert result[0].status == OrderStatus.PENDING

        # Verify query
        call_args = mock_adapter.fetch_all.call_args
        assert "WHERE status = %s" in call_args[0][0]
        assert call_args[0][1] == "pending"

    async def test_get_active_orders(self, repository, mock_adapter):
        """Test getting active orders."""
        sample_record = {
            "id": uuid4(),
            "symbol": "AAPL",
            "side": "buy",
            "order_type": "limit",
            "status": "pending",
            "quantity": Decimal("100"),
            "limit_price": Decimal("150.00"),
            "stop_price": None,
            "time_in_force": "day",
            "broker_order_id": "BROKER123",
            "filled_quantity": Decimal("0"),
            "average_fill_price": None,
            "created_at": datetime.now(UTC),
            "submitted_at": None,
            "filled_at": None,
            "cancelled_at": None,
            "reason": None,
            "tags": {"strategy": "momentum"},
        }
        active_order_records = [
            {**sample_record, "status": "pending"},
            {**sample_record, "status": "submitted"},
            {**sample_record, "status": "partially_filled"},
        ]
        mock_adapter.fetch_all.return_value = active_order_records

        result = await repository.get_active_orders()

        assert len(result) == 3
        assert all(
            order.status
            in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
            for order in result
        )

        # Verify query
        call_args = mock_adapter.fetch_all.call_args
        assert "WHERE status IN ('pending', 'submitted', 'partially_filled')" in call_args[0][0]

    async def test_get_orders_by_date_range(self, repository, mock_adapter, sample_order_record):
        """Test getting orders by date range."""
        start_date = datetime.now(UTC) - timedelta(days=7)
        end_date = datetime.now(UTC)
        mock_adapter.fetch_all.return_value = [sample_order_record]

        result = await repository.get_orders_by_date_range(start_date, end_date)

        assert len(result) == 1

        # Verify query parameters
        call_args = mock_adapter.fetch_all.call_args
        assert "WHERE created_at >= %s AND created_at <= %s" in call_args[0][0]
        assert call_args[0][1] == start_date
        assert call_args[0][2] == end_date

    async def test_get_orders_by_broker_id(self, repository, mock_adapter, sample_order_record):
        """Test getting orders by broker order ID."""
        mock_adapter.fetch_all.return_value = [sample_order_record]

        result = await repository.get_orders_by_broker_id("BROKER123")

        assert len(result) == 1
        assert result[0].broker_order_id == "BROKER123"

        # Verify query
        call_args = mock_adapter.fetch_all.call_args
        assert "WHERE broker_order_id = %s" in call_args[0][0]
        assert call_args[0][1] == "BROKER123"

    async def test_update_order_success(self, repository, mock_adapter, sample_order):
        """Test successful order update."""
        mock_adapter.execute_query.return_value = "UPDATE 1"

        result = await repository.update_order(sample_order)

        assert result == sample_order

        # Verify update query
        call_args = mock_adapter.execute_query.call_args
        assert "UPDATE orders SET" in call_args[0][0]
        assert "WHERE id = %s" in call_args[0][0]
        # ID should be last parameter
        assert call_args[0][-1] == sample_order.id

    async def test_update_order_not_found(self, repository, mock_adapter, sample_order):
        """Test updating non-existent order."""
        mock_adapter.execute_query.return_value = "UPDATE 0"

        with pytest.raises(OrderNotFoundError):
            await repository.update_order(sample_order)

    async def test_delete_order_success(self, repository, mock_adapter):
        """Test successful order deletion."""
        order_id = uuid4()
        mock_adapter.execute_query.return_value = "DELETE 1"

        result = await repository.delete_order(order_id)

        assert result is True

        # Verify delete query
        call_args = mock_adapter.execute_query.call_args
        assert "DELETE FROM orders WHERE id = %s" in call_args[0][0]
        assert call_args[0][1] == order_id

    async def test_delete_order_not_found(self, repository, mock_adapter):
        """Test deleting non-existent order."""
        mock_adapter.execute_query.return_value = "DELETE 0"

        result = await repository.delete_order(uuid4())

        assert result is False


@pytest.mark.unit
class TestOrderEntityMapping:
    """Test entity mapping between domain objects and database records."""

    def test_map_record_to_order_complete(self, repository, sample_order_record):
        """Test mapping complete database record to order entity."""
        order = repository._map_record_to_order(sample_order_record)

        assert order.id == sample_order_record["id"]
        assert order.symbol == sample_order_record["symbol"]
        assert order.side == OrderSide(sample_order_record["side"])
        assert order.order_type == OrderType(sample_order_record["order_type"])
        assert order.status == OrderStatus(sample_order_record["status"])
        assert order.quantity == sample_order_record["quantity"]
        assert order.limit_price == sample_order_record["limit_price"]
        assert order.stop_price == sample_order_record["stop_price"]
        assert order.time_in_force == TimeInForce(sample_order_record["time_in_force"])
        assert order.broker_order_id == sample_order_record["broker_order_id"]
        assert order.filled_quantity == sample_order_record["filled_quantity"]
        assert order.average_fill_price == sample_order_record["average_fill_price"]
        assert order.tags == sample_order_record["tags"]

    def test_map_record_to_order_with_nulls(self, repository, sample_order_record):
        """Test mapping record with null optional fields."""
        sample_order_record["stop_price"] = None
        sample_order_record["average_fill_price"] = None
        sample_order_record["submitted_at"] = None
        sample_order_record["filled_at"] = None
        sample_order_record["cancelled_at"] = None
        sample_order_record["reason"] = None
        sample_order_record["tags"] = None

        order = repository._map_record_to_order(sample_order_record)

        assert order.stop_price is None
        assert order.average_fill_price is None
        assert order.submitted_at is None
        assert order.filled_at is None
        assert order.cancelled_at is None
        assert order.reason is None
        assert order.tags == {}

    def test_map_record_to_order_different_types(self, repository):
        """Test mapping different order types and statuses."""
        sample_record = {
            "id": uuid4(),
            "symbol": "AAPL",
            "side": "buy",
            "order_type": "market",
            "status": "pending",
            "quantity": Decimal("100"),
            "limit_price": None,
            "stop_price": None,
            "time_in_force": "day",
            "broker_order_id": "BROKER123",
            "filled_quantity": Decimal("0"),
            "average_fill_price": None,
            "created_at": datetime.now(UTC),
            "submitted_at": None,
            "filled_at": None,
            "cancelled_at": None,
            "reason": None,
            "tags": {"strategy": "momentum"},
        }
        # Market order
        order = repository._map_record_to_order(sample_record)
        assert order.order_type == OrderType.MARKET
        assert order.limit_price is None

        # Stop order
        stop_order_record = {
            **sample_record,
            "order_type": "stop",
            "stop_price": Decimal("145.00"),
        }
        order = repository._map_record_to_order(stop_order_record)
        assert order.order_type == OrderType.STOP
        assert order.stop_price == Decimal("145.00")

        # Different statuses
        for status in ["submitted", "filled", "cancelled", "rejected", "partially_filled"]:
            record = {**sample_record, "status": status}
            order = repository._map_record_to_order(record)
            assert order.status == OrderStatus(status)

    def test_map_record_to_order_different_sides(self, repository):
        """Test mapping different order sides."""
        sample_record = {
            "id": uuid4(),
            "symbol": "AAPL",
            "side": "sell",
            "order_type": "limit",
            "status": "pending",
            "quantity": Decimal("100"),
            "limit_price": Decimal("150.00"),
            "stop_price": None,
            "time_in_force": "day",
            "broker_order_id": "BROKER123",
            "filled_quantity": Decimal("0"),
            "average_fill_price": None,
            "created_at": datetime.now(UTC),
            "submitted_at": None,
            "filled_at": None,
            "cancelled_at": None,
            "reason": None,
            "tags": {"strategy": "momentum"},
        }
        # Sell order
        order = repository._map_record_to_order(sample_record)
        assert order.side == OrderSide.SELL

        # Buy order
        buy_order_record = {**sample_record, "side": "buy"}
        order = repository._map_record_to_order(buy_order_record)
        assert order.side == OrderSide.BUY

    def test_map_record_to_order_time_in_force(self, repository):
        """Test mapping different time in force values."""
        sample_record = {
            "id": uuid4(),
            "symbol": "AAPL",
            "side": "buy",
            "order_type": "limit",
            "status": "pending",
            "quantity": Decimal("100"),
            "limit_price": Decimal("150.00"),
            "stop_price": None,
            "time_in_force": "day",
            "broker_order_id": "BROKER123",
            "filled_quantity": Decimal("0"),
            "average_fill_price": None,
            "created_at": datetime.now(UTC),
            "submitted_at": None,
            "filled_at": None,
            "cancelled_at": None,
            "reason": None,
            "tags": {"strategy": "momentum"},
        }
        for tif in ["day", "gtc", "ioc", "fok"]:
            record = {**sample_record, "time_in_force": tif}
            order = repository._map_record_to_order(record)
            assert order.time_in_force == TimeInForce(tif)


@pytest.mark.unit
class TestOrderErrorHandling:
    """Test order repository error handling."""

    async def test_save_order_error(self, repository, mock_adapter, sample_order):
        """Test save order with database error."""
        mock_adapter.fetch_one.side_effect = Exception("Database connection failed")

        with pytest.raises(RepositoryError, match="Failed to save order"):
            await repository.save_order(sample_order)

    async def test_get_order_by_id_error(self, repository, mock_adapter):
        """Test get order by ID with database error."""
        mock_adapter.fetch_one.side_effect = Exception("Query failed")

        with pytest.raises(RepositoryError, match="Failed to retrieve order"):
            await repository.get_order_by_id(uuid4())

    async def test_get_orders_by_symbol_error(self, repository, mock_adapter):
        """Test get orders by symbol with database error."""
        mock_adapter.fetch_all.side_effect = Exception("Query failed")

        with pytest.raises(RepositoryError, match="Failed to retrieve orders for symbol"):
            await repository.get_orders_by_symbol("AAPL")

    async def test_get_orders_by_status_error(self, repository, mock_adapter):
        """Test get orders by status with database error."""
        mock_adapter.fetch_all.side_effect = Exception("Query failed")

        with pytest.raises(RepositoryError, match="Failed to retrieve orders by status"):
            await repository.get_orders_by_status(OrderStatus.PENDING)

    async def test_get_active_orders_error(self, repository, mock_adapter):
        """Test get active orders with database error."""
        mock_adapter.fetch_all.side_effect = Exception("Query failed")

        with pytest.raises(RepositoryError, match="Failed to retrieve active orders"):
            await repository.get_active_orders()

    async def test_get_orders_by_date_range_error(self, repository, mock_adapter):
        """Test get orders by date range with database error."""
        mock_adapter.fetch_all.side_effect = Exception("Query failed")

        with pytest.raises(RepositoryError, match="Failed to retrieve orders by date range"):
            await repository.get_orders_by_date_range(datetime.now(UTC), datetime.now(UTC))

    async def test_get_orders_by_broker_id_error(self, repository, mock_adapter):
        """Test get orders by broker ID with database error."""
        mock_adapter.fetch_all.side_effect = Exception("Query failed")

        with pytest.raises(RepositoryError, match="Failed to retrieve orders by broker ID"):
            await repository.get_orders_by_broker_id("BROKER123")

    async def test_update_order_error(self, repository, mock_adapter, sample_order):
        """Test update order with database error."""
        mock_adapter.execute_query.side_effect = Exception("Update failed")

        with pytest.raises(RepositoryError, match="Failed to update order"):
            await repository.update_order(sample_order)

    async def test_delete_order_error(self, repository, mock_adapter):
        """Test delete order with database error."""
        mock_adapter.execute_query.side_effect = Exception("Delete failed")

        with pytest.raises(RepositoryError, match="Failed to delete order"):
            await repository.delete_order(uuid4())


@pytest.mark.unit
class TestOrderLifecycle:
    """Test order lifecycle operations."""

    async def test_order_submission_flow(self, repository, mock_adapter):
        """Test order from creation to submission."""
        # Create pending order
        order = Order(
            id=uuid4(),
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.PENDING,
            quantity=Decimal("100"),
            limit_price=Decimal("150.00"),
            stop_price=None,
            time_in_force=TimeInForce.DAY,
            broker_order_id="BROKER123",
            filled_quantity=Decimal("0"),
            average_fill_price=None,
            created_at=datetime.now(UTC),
            submitted_at=None,
            filled_at=None,
            cancelled_at=None,
            reason=None,
            tags={"strategy": "momentum"},
        )

        # Save initial order
        mock_adapter.fetch_one.return_value = None
        await repository.save_order(order)

        # Update to submitted
        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.now(UTC)
        sample_record = {
            "id": order.id,
            "symbol": "AAPL",
            "side": "buy",
            "order_type": "limit",
            "status": "pending",
            "quantity": Decimal("100"),
            "limit_price": Decimal("150.00"),
            "stop_price": None,
            "time_in_force": "day",
            "broker_order_id": "BROKER123",
            "filled_quantity": Decimal("0"),
            "average_fill_price": None,
            "created_at": datetime.now(UTC),
            "submitted_at": None,
            "filled_at": None,
            "cancelled_at": None,
            "reason": None,
            "tags": {"strategy": "momentum"},
        }
        mock_adapter.fetch_one.return_value = sample_record
        mock_adapter.execute_query.return_value = "UPDATE 1"

        result = await repository.save_order(order)
        assert result.status == OrderStatus.SUBMITTED

    async def test_order_fill_flow(self, repository, mock_adapter):
        """Test order filling process."""
        order = Order(
            id=uuid4(),
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.SUBMITTED,
            quantity=Decimal("100"),
            limit_price=Decimal("150.00"),
            stop_price=None,
            time_in_force=TimeInForce.DAY,
            broker_order_id="BROKER123",
            filled_quantity=Decimal("0"),
            average_fill_price=None,
            created_at=datetime.now(UTC),
            submitted_at=None,
            filled_at=None,
            cancelled_at=None,
            reason=None,
            tags={"strategy": "momentum"},
        )

        # Partial fill
        order.status = OrderStatus.PARTIALLY_FILLED
        order.filled_quantity = Decimal("50")
        order.average_fill_price = Decimal("149.50")

        mock_adapter.execute_query.return_value = "UPDATE 1"
        result = await repository.update_order(order)
        assert result.status == OrderStatus.PARTIALLY_FILLED
        assert result.filled_quantity == Decimal("50")

        # Complete fill
        order.status = OrderStatus.FILLED
        order.filled_quantity = Decimal("100")
        order.filled_at = datetime.now(UTC)

        result = await repository.update_order(order)
        assert result.status == OrderStatus.FILLED
        assert result.filled_quantity == Decimal("100")

    async def test_order_cancellation_flow(self, repository, mock_adapter):
        """Test order cancellation process."""
        order = Order(
            id=uuid4(),
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.SUBMITTED,
            quantity=Decimal("100"),
            limit_price=Decimal("150.00"),
            stop_price=None,
            time_in_force=TimeInForce.DAY,
            broker_order_id="BROKER123",
            filled_quantity=Decimal("0"),
            average_fill_price=None,
            created_at=datetime.now(UTC),
            submitted_at=None,
            filled_at=None,
            cancelled_at=None,
            reason=None,
            tags={"strategy": "momentum"},
        )

        # Cancel order
        order.status = OrderStatus.CANCELLED
        order.cancelled_at = datetime.now(UTC)
        order.reason = "User requested cancellation"

        mock_adapter.execute_query.return_value = "UPDATE 1"
        result = await repository.update_order(order)

        assert result.status == OrderStatus.CANCELLED
        assert result.cancelled_at is not None
        assert result.reason == "User requested cancellation"


@pytest.mark.unit
class TestOrderBulkOperations:
    """Test bulk order operations."""

    async def test_get_multiple_active_orders(self, repository, mock_adapter):
        """Test retrieving multiple active orders."""
        sample_record = {
            "id": uuid4(),
            "symbol": "AAPL",
            "side": "buy",
            "order_type": "limit",
            "status": "pending",
            "quantity": Decimal("100"),
            "limit_price": Decimal("150.00"),
            "stop_price": None,
            "time_in_force": "day",
            "broker_order_id": "BROKER123",
            "filled_quantity": Decimal("0"),
            "average_fill_price": None,
            "created_at": datetime.now(UTC),
            "submitted_at": None,
            "filled_at": None,
            "cancelled_at": None,
            "reason": None,
            "tags": {"strategy": "momentum"},
        }
        active_records = [
            {**sample_record, "id": uuid4(), "symbol": "AAPL", "status": "pending"},
            {**sample_record, "id": uuid4(), "symbol": "GOOGL", "status": "submitted"},
            {**sample_record, "id": uuid4(), "symbol": "MSFT", "status": "partially_filled"},
        ]
        mock_adapter.fetch_all.return_value = active_records

        result = await repository.get_active_orders()

        assert len(result) == 3
        symbols = {order.symbol for order in result}
        assert symbols == {"AAPL", "GOOGL", "MSFT"}

    async def test_get_orders_multiple_statuses(self, repository, mock_adapter):
        """Test getting orders with different statuses."""
        sample_record = {
            "id": uuid4(),
            "symbol": "AAPL",
            "side": "buy",
            "order_type": "limit",
            "status": "pending",
            "quantity": Decimal("100"),
            "limit_price": Decimal("150.00"),
            "stop_price": None,
            "time_in_force": "day",
            "broker_order_id": "BROKER123",
            "filled_quantity": Decimal("0"),
            "average_fill_price": None,
            "created_at": datetime.now(UTC),
            "submitted_at": None,
            "filled_at": None,
            "cancelled_at": None,
            "reason": None,
            "tags": {"strategy": "momentum"},
        }
        for status in OrderStatus:
            mock_adapter.fetch_all.return_value = [{**sample_record, "status": status}]

            result = await repository.get_orders_by_status(status)
            assert len(result) == 1
            assert result[0].status == status


@pytest.mark.unit
class TestOrderQueryOptimizations:
    """Test query optimization and edge cases."""

    async def test_get_orders_with_empty_results(self, repository, mock_adapter):
        """Test methods that return empty lists."""
        mock_adapter.fetch_all.return_value = []

        # Test empty symbol search
        result = await repository.get_orders_by_symbol("NONEXISTENT")
        assert result == []

        # Test empty status search
        result = await repository.get_orders_by_status(OrderStatus.FILLED)
        assert result == []

        # Test empty active orders
        result = await repository.get_active_orders()
        assert result == []

        # Test empty date range
        start_date = datetime.now(UTC) - timedelta(days=1)
        end_date = datetime.now(UTC)
        result = await repository.get_orders_by_date_range(start_date, end_date)
        assert result == []

        # Test empty broker ID search
        result = await repository.get_orders_by_broker_id("NONEXISTENT")
        assert result == []

    async def test_save_order_exception_during_get(self, repository, mock_adapter, sample_order):
        """Test save order when get_order_by_id throws exception."""
        # First call to fetch_one throws exception, should be caught and wrapped
        mock_adapter.fetch_one.side_effect = Exception("Database error")

        with pytest.raises(RepositoryError, match="Failed to save order"):
            await repository.save_order(sample_order)

    async def test_save_order_exception_during_update(
        self, repository, mock_adapter, sample_order, sample_order_record
    ):
        """Test save order when update throws exception."""
        # First call returns record (order exists), second call in update fails
        mock_adapter.fetch_one.return_value = sample_order_record
        mock_adapter.execute_query.side_effect = Exception("Update failed")

        with pytest.raises(RepositoryError, match="Failed to update order"):
            await repository.save_order(sample_order)

    async def test_save_order_exception_during_insert(self, repository, mock_adapter, sample_order):
        """Test save order when insert throws exception."""
        # First call returns None (order doesn't exist), insert fails
        mock_adapter.fetch_one.return_value = None
        mock_adapter.execute_query.side_effect = Exception("Insert failed")

        with pytest.raises(RepositoryError, match="Failed to save order"):
            await repository.save_order(sample_order)

    async def test_update_order_not_found_exception_propagation(
        self, repository, mock_adapter, sample_order
    ):
        """Test that OrderNotFoundError is properly propagated in update_order."""
        mock_adapter.execute_query.return_value = "UPDATE 0"

        with pytest.raises(OrderNotFoundError):
            await repository.update_order(sample_order)

    async def test_repository_initialization(self):
        """Test repository initialization."""
        mock_adapter = AsyncMock()
        repo = PostgreSQLOrderRepository(mock_adapter)

        assert repo.adapter is mock_adapter

    async def test_private_insert_order_method(self, repository, mock_adapter, sample_order):
        """Test the private _insert_order method directly."""
        result = await repository._insert_order(sample_order)

        assert result == sample_order
        mock_adapter.execute_query.assert_called_once()

        # Verify query structure
        call_args = mock_adapter.execute_query.call_args
        assert "INSERT INTO orders" in call_args[0][0]
        # Verify all 18 parameters are passed
        assert len(call_args[0]) == 19  # query + 18 params
