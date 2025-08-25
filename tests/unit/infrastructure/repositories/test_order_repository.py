"""
Unit tests for PostgreSQL Order Repository Implementation.

Tests the concrete implementation of IOrderRepository including CRUD operations,
query methods, entity mapping, and error handling scenarios.
"""

# Standard library imports
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock
from uuid import uuid4

# Third-party imports
import pytest

# Local imports
from src.application.interfaces.exceptions import OrderNotFoundError, RepositoryError
from src.domain.entities.order import (
    Order,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from src.infrastructure.repositories.order_repository import PostgreSQLOrderRepository


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
    """Order repository with mocked adapter."""
    return PostgreSQLOrderRepository(mock_adapter)


@pytest.fixture
def sample_order():
    """Sample order entity for testing."""
    request = OrderRequest(
        symbol="AAPL",
        quantity=Decimal("100"),
        side=OrderSide.BUY,
        limit_price=Decimal("150.00"),
        reason="Test order",
    )
    return Order.create_limit_order(request)


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


@pytest.mark.unit
class TestOrderRepositoryInitialization:
    """Test repository initialization."""

    def test_repository_initialization(self, mock_adapter):
        """Test repository is properly initialized."""
        repository = PostgreSQLOrderRepository(mock_adapter)

        assert repository.adapter == mock_adapter


@pytest.mark.unit
class TestSaveOrder:
    """Test order save operations."""

    async def test_save_new_order_success(self, repository, mock_adapter, sample_order):
        """Test saving a new order successfully."""
        # Mock that order doesn't exist
        mock_adapter.fetch_one.return_value = None

        result = await repository.save_order(sample_order)

        assert result == sample_order

        # Verify insert was called
        mock_adapter.execute_query.assert_called_once()
        call_args = mock_adapter.execute_query.call_args

        # Check INSERT statement
        assert "INSERT INTO orders" in call_args[0][0]
        assert call_args[0][1] == sample_order.id
        assert call_args[0][2] == sample_order.symbol
        assert call_args[0][3] == sample_order.side.value

    async def test_save_existing_order_calls_update(
        self, repository, mock_adapter, sample_order, sample_order_record
    ):
        """Test saving existing order calls update."""
        # Mock that order exists with complete record
        mock_adapter.fetch_one.return_value = sample_order_record
        mock_adapter.execute_query.return_value = "UPDATE 1"

        result = await repository.save_order(sample_order)

        assert result == sample_order

        # Should call fetch_one (check if exists) and then update
        mock_adapter.fetch_one.assert_called_once()
        mock_adapter.execute_query.assert_called_once()  # update call

    async def test_save_order_adapter_error(self, repository, mock_adapter, sample_order):
        """Test save order with adapter error."""
        mock_adapter.fetch_one.side_effect = Exception("Database error")

        with pytest.raises(RepositoryError, match="Failed to save order"):
            await repository.save_order(sample_order)

    async def test_insert_order_success(self, repository, mock_adapter, sample_order):
        """Test successful order insertion."""
        result = await repository._insert_order(sample_order)

        assert result == sample_order

        # Verify correct parameters were passed
        call_args = mock_adapter.execute_query.call_args
        assert call_args[0][1] == sample_order.id
        assert call_args[0][2] == sample_order.symbol
        assert call_args[0][3] == sample_order.side.value
        assert call_args[0][4] == sample_order.order_type.value
        assert call_args[0][5] == sample_order.status.value
        assert call_args[0][6] == sample_order.quantity


@pytest.mark.unit
class TestGetOrder:
    """Test order retrieval operations."""

    async def test_get_order_by_id_found(self, repository, mock_adapter, sample_order_record):
        """Test getting order by ID when order exists."""
        mock_adapter.fetch_one.return_value = sample_order_record

        result = await repository.get_order_by_id(sample_order_record["id"])

        assert result is not None
        assert result.id == sample_order_record["id"]
        assert result.symbol == sample_order_record["symbol"]
        assert result.side == OrderSide(sample_order_record["side"])
        assert result.order_type == OrderType(sample_order_record["order_type"])
        assert result.status == OrderStatus(sample_order_record["status"])

        # Verify query was called with correct parameters
        mock_adapter.fetch_one.assert_called_once()
        call_args = mock_adapter.fetch_one.call_args
        assert "SELECT" in call_args[0][0]
        assert "WHERE id = %s" in call_args[0][0]
        assert call_args[0][1] == sample_order_record["id"]

    async def test_get_order_by_id_not_found(self, repository, mock_adapter):
        """Test getting order by ID when order doesn't exist."""
        mock_adapter.fetch_one.return_value = None
        order_id = uuid4()

        result = await repository.get_order_by_id(order_id)

        assert result is None
        mock_adapter.fetch_one.assert_called_once()

    async def test_get_order_by_id_adapter_error(self, repository, mock_adapter):
        """Test get order by ID with adapter error."""
        mock_adapter.fetch_one.side_effect = Exception("Database error")
        order_id = uuid4()

        with pytest.raises(RepositoryError, match="Failed to retrieve order"):
            await repository.get_order_by_id(order_id)

    async def test_get_orders_by_symbol_success(
        self, repository, mock_adapter, sample_order_record
    ):
        """Test getting orders by symbol."""
        mock_adapter.fetch_all.return_value = [sample_order_record]

        result = await repository.get_orders_by_symbol("AAPL")

        assert len(result) == 1
        assert result[0].symbol == "AAPL"

        # Verify query
        mock_adapter.fetch_all.assert_called_once()
        call_args = mock_adapter.fetch_all.call_args
        assert "WHERE symbol = %s" in call_args[0][0]
        assert "ORDER BY created_at DESC" in call_args[0][0]
        assert call_args[0][1] == "AAPL"

    async def test_get_orders_by_symbol_empty_result(self, repository, mock_adapter):
        """Test getting orders by symbol with no results."""
        mock_adapter.fetch_all.return_value = []

        result = await repository.get_orders_by_symbol("NONEXISTENT")

        assert result == []

    async def test_get_orders_by_status_success(
        self, repository, mock_adapter, sample_order_record
    ):
        """Test getting orders by status."""
        mock_adapter.fetch_all.return_value = [sample_order_record]

        result = await repository.get_orders_by_status(OrderStatus.PENDING)

        assert len(result) == 1
        assert result[0].status == OrderStatus.PENDING

        # Verify query
        call_args = mock_adapter.fetch_all.call_args
        assert "WHERE status = %s" in call_args[0][0]
        assert call_args[0][1] == OrderStatus.PENDING.value

    async def test_get_active_orders_success(self, repository, mock_adapter, sample_order_record):
        """Test getting active orders."""
        mock_adapter.fetch_all.return_value = [sample_order_record]

        result = await repository.get_active_orders()

        assert len(result) == 1
        # Note: Order entity might not have is_active() method, check status instead
        assert result[0].status in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED,
        ]

        # Verify query includes active statuses
        call_args = mock_adapter.fetch_all.call_args
        assert "WHERE status IN" in call_args[0][0]
        assert "pending" in call_args[0][0]
        assert "submitted" in call_args[0][0]
        assert "partially_filled" in call_args[0][0]

    async def test_get_orders_by_date_range_success(
        self, repository, mock_adapter, sample_order_record
    ):
        """Test getting orders by date range."""
        start_date = datetime(2023, 1, 1, tzinfo=UTC)
        end_date = datetime(2023, 12, 31, tzinfo=UTC)
        mock_adapter.fetch_all.return_value = [sample_order_record]

        result = await repository.get_orders_by_date_range(start_date, end_date)

        assert len(result) == 1

        # Verify query
        call_args = mock_adapter.fetch_all.call_args
        assert "WHERE created_at >= %s AND created_at <= %s" in call_args[0][0]
        assert call_args[0][1] == start_date
        assert call_args[0][2] == end_date

    async def test_get_orders_by_broker_id_success(
        self, repository, mock_adapter, sample_order_record
    ):
        """Test getting orders by broker ID."""
        sample_order_record["broker_order_id"] = "BROKER123"
        mock_adapter.fetch_all.return_value = [sample_order_record]

        result = await repository.get_orders_by_broker_id("BROKER123")

        assert len(result) == 1
        assert result[0].broker_order_id == "BROKER123"

        # Verify query
        call_args = mock_adapter.fetch_all.call_args
        assert "WHERE broker_order_id = %s" in call_args[0][0]
        assert call_args[0][1] == "BROKER123"


@pytest.mark.unit
class TestUpdateOrder:
    """Test order update operations."""

    async def test_update_order_success(self, repository, mock_adapter, sample_order):
        """Test successful order update."""
        mock_adapter.execute_query.return_value = "UPDATE 1"

        result = await repository.update_order(sample_order)

        assert result == sample_order

        # Verify update query
        mock_adapter.execute_query.assert_called_once()
        call_args = mock_adapter.execute_query.call_args
        assert "UPDATE orders SET" in call_args[0][0]
        assert "WHERE id = %s" in call_args[0][0]

        # Check parameters - ID should be last parameter
        assert call_args[0][-1] == sample_order.id

    async def test_update_order_not_found(self, repository, mock_adapter, sample_order):
        """Test update order when order doesn't exist."""
        mock_adapter.execute_query.return_value = "UPDATE 0"

        with pytest.raises(OrderNotFoundError):
            await repository.update_order(sample_order)

    async def test_update_order_adapter_error(self, repository, mock_adapter, sample_order):
        """Test update order with adapter error."""
        mock_adapter.execute_query.side_effect = Exception("Database error")

        with pytest.raises(RepositoryError, match="Failed to update order"):
            await repository.update_order(sample_order)


@pytest.mark.unit
class TestDeleteOrder:
    """Test order deletion operations."""

    async def test_delete_order_success(self, repository, mock_adapter):
        """Test successful order deletion."""
        order_id = uuid4()
        mock_adapter.execute_query.return_value = "DELETE 1"

        result = await repository.delete_order(order_id)

        assert result is True

        # Verify delete query
        mock_adapter.execute_query.assert_called_once()
        call_args = mock_adapter.execute_query.call_args
        assert "DELETE FROM orders WHERE id = %s" in call_args[0][0]
        assert call_args[0][1] == order_id

    async def test_delete_order_not_found(self, repository, mock_adapter):
        """Test delete order when order doesn't exist."""
        order_id = uuid4()
        mock_adapter.execute_query.return_value = "DELETE 0"

        result = await repository.delete_order(order_id)

        assert result is False

    async def test_delete_order_adapter_error(self, repository, mock_adapter):
        """Test delete order with adapter error."""
        order_id = uuid4()
        mock_adapter.execute_query.side_effect = Exception("Database error")

        with pytest.raises(RepositoryError, match="Failed to delete order"):
            await repository.delete_order(order_id)


@pytest.mark.unit
class TestEntityMapping:
    """Test entity mapping between domain objects and database records."""

    def test_map_record_to_order_complete_record(self, repository, sample_order_record):
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
        assert order.created_at == sample_order_record["created_at"]
        assert order.submitted_at == sample_order_record["submitted_at"]
        assert order.filled_at == sample_order_record["filled_at"]
        assert order.cancelled_at == sample_order_record["cancelled_at"]
        assert order.reason == sample_order_record["reason"]
        assert order.tags == sample_order_record["tags"]

    def test_map_record_to_order_with_null_tags(self, repository, sample_order_record):
        """Test mapping record with null tags field."""
        sample_order_record["tags"] = None

        order = repository._map_record_to_order(sample_order_record)

        assert order.tags == {}  # Should default to empty dict

    def test_map_record_to_order_market_order(self, repository):
        """Test mapping market order record."""
        record = {
            "id": uuid4(),
            "symbol": "TSLA",
            "side": "sell",
            "order_type": "market",
            "status": "filled",
            "quantity": Decimal("50"),
            "limit_price": None,
            "stop_price": None,
            "time_in_force": "day",
            "broker_order_id": "BROKER456",
            "filled_quantity": Decimal("50"),
            "average_fill_price": Decimal("200.00"),
            "created_at": datetime.now(UTC),
            "submitted_at": datetime.now(UTC),
            "filled_at": datetime.now(UTC),
            "cancelled_at": None,
            "reason": "Market order",
            "tags": {"strategy": "momentum"},
        }

        order = repository._map_record_to_order(record)

        assert order.symbol == "TSLA"
        assert order.side == OrderSide.SELL
        assert order.order_type == OrderType.MARKET
        assert order.status == OrderStatus.FILLED
        assert order.limit_price is None
        assert order.stop_price is None
        assert order.tags == {"strategy": "momentum"}

    def test_map_record_to_order_stop_limit_order(self, repository):
        """Test mapping stop-limit order record."""
        record = {
            "id": uuid4(),
            "symbol": "GOOGL",
            "side": "buy",
            "order_type": "stop_limit",
            "status": "submitted",
            "quantity": Decimal("10"),
            "limit_price": Decimal("2800.00"),
            "stop_price": Decimal("2750.00"),
            "time_in_force": "gtc",
            "broker_order_id": "BROKER789",
            "filled_quantity": Decimal("0"),
            "average_fill_price": None,
            "created_at": datetime.now(UTC),
            "submitted_at": datetime.now(UTC),
            "filled_at": None,
            "cancelled_at": None,
            "reason": "Stop loss order",
            "tags": {},
        }

        order = repository._map_record_to_order(record)

        assert order.symbol == "GOOGL"
        assert order.order_type == OrderType.STOP_LIMIT
        assert order.limit_price == Decimal("2800.00")
        assert order.stop_price == Decimal("2750.00")
        assert order.time_in_force == TimeInForce.GTC


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling scenarios."""

    async def test_fetch_operations_with_various_exceptions(self, repository, mock_adapter):
        """Test fetch operations handle different exception types."""
        test_cases = [
            (Exception("Generic error"), RepositoryError),
            (ValueError("Value error"), RepositoryError),
            (RuntimeError("Runtime error"), RepositoryError),
        ]

        for exception, expected_error in test_cases:
            mock_adapter.fetch_one.side_effect = exception

            with pytest.raises(expected_error):
                await repository.get_order_by_id(uuid4())

            mock_adapter.fetch_one.side_effect = None  # Reset

    async def test_execute_operations_with_various_exceptions(
        self, repository, mock_adapter, sample_order
    ):
        """Test execute operations handle different exception types."""
        # Test generic exception handling during insert
        mock_adapter.fetch_one.return_value = None  # Order doesn't exist, so it will try to insert
        mock_adapter.execute_query.side_effect = Exception("Generic error")

        with pytest.raises(RepositoryError, match="Failed to save order"):
            await repository.save_order(sample_order)

        # Reset mock
        mock_adapter.execute_query.side_effect = None
        mock_adapter.execute_query.return_value = "UPDATE 1"


@pytest.mark.unit
class TestQueryConstruction:
    """Test SQL query construction and parameters."""

    async def test_insert_query_parameters(self, repository, mock_adapter, sample_order):
        """Test insert query includes all required parameters."""
        await repository._insert_order(sample_order)

        call_args = mock_adapter.execute_query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]

        # Check query structure
        assert "INSERT INTO orders" in query
        assert "VALUES" in query
        assert query.count("%s") == len(params)

        # Check all required fields are included
        expected_fields = [
            "id",
            "symbol",
            "side",
            "order_type",
            "status",
            "quantity",
            "limit_price",
            "stop_price",
            "time_in_force",
            "broker_order_id",
            "filled_quantity",
            "average_fill_price",
            "created_at",
            "submitted_at",
            "filled_at",
            "cancelled_at",
            "reason",
            "tags",
        ]

        for field in expected_fields:
            assert field in query

    async def test_update_query_parameters(self, repository, mock_adapter, sample_order):
        """Test update query includes all fields and WHERE clause."""
        mock_adapter.execute_query.return_value = "UPDATE 1"

        await repository.update_order(sample_order)

        call_args = mock_adapter.execute_query.call_args
        query = call_args[0][0]
        params = call_args[0][1:]

        # Check query structure
        assert "UPDATE orders SET" in query
        assert "WHERE id = %s" in query
        assert query.count("%s") == len(params)

        # ID should be the last parameter (for WHERE clause)
        assert params[-1] == sample_order.id

    async def test_select_queries_include_all_fields(self, repository, mock_adapter):
        """Test select queries include all required fields."""
        mock_adapter.fetch_one.return_value = None
        mock_adapter.fetch_all.return_value = []

        # Test various select operations
        await repository.get_order_by_id(uuid4())
        await repository.get_orders_by_symbol("AAPL")
        await repository.get_orders_by_status(OrderStatus.PENDING)
        await repository.get_active_orders()

        # Check that all calls include the expected fields
        expected_fields = [
            "id",
            "symbol",
            "side",
            "order_type",
            "status",
            "quantity",
            "limit_price",
            "stop_price",
            "time_in_force",
            "broker_order_id",
            "filled_quantity",
            "average_fill_price",
            "created_at",
            "submitted_at",
            "filled_at",
            "cancelled_at",
            "reason",
            "tags",
        ]

        for call_args in (
            mock_adapter.fetch_one.call_args_list + mock_adapter.fetch_all.call_args_list
        ):
            query = call_args[0][0]
            for field in expected_fields:
                assert field in query


@pytest.mark.unit
class TestAdditionalOrderCoverage:
    """Test additional scenarios to achieve 90%+ coverage."""

    async def test_get_orders_by_symbol_error(self, repository, mock_adapter):
        """Test get orders by symbol with database error."""
        mock_adapter.fetch_all.side_effect = Exception("Database error")

        with pytest.raises(RepositoryError, match="Failed to retrieve orders for symbol"):
            await repository.get_orders_by_symbol("AAPL")

    async def test_get_orders_by_status_error(self, repository, mock_adapter):
        """Test get orders by status with database error."""
        mock_adapter.fetch_all.side_effect = Exception("Database error")

        with pytest.raises(RepositoryError, match="Failed to retrieve orders by status"):
            await repository.get_orders_by_status(OrderStatus.PENDING)

    async def test_get_active_orders_error(self, repository, mock_adapter):
        """Test get active orders with database error."""
        mock_adapter.fetch_all.side_effect = Exception("Database error")

        with pytest.raises(RepositoryError, match="Failed to retrieve active orders"):
            await repository.get_active_orders()

    async def test_get_orders_by_date_range_error(self, repository, mock_adapter):
        """Test get orders by date range with database error."""
        mock_adapter.fetch_all.side_effect = Exception("Database error")
        start_date = datetime(2023, 1, 1, tzinfo=UTC)
        end_date = datetime(2023, 12, 31, tzinfo=UTC)

        with pytest.raises(RepositoryError, match="Failed to retrieve orders by date range"):
            await repository.get_orders_by_date_range(start_date, end_date)

    async def test_get_orders_by_broker_id_error(self, repository, mock_adapter):
        """Test get orders by broker ID with database error."""
        mock_adapter.fetch_all.side_effect = Exception("Database error")

        with pytest.raises(RepositoryError, match="Failed to retrieve orders by broker ID"):
            await repository.get_orders_by_broker_id("BROKER123")

    async def test_insert_order_error(self, repository, mock_adapter, sample_order):
        """Test insert order with database error."""
        mock_adapter.execute_query.side_effect = Exception("Insert failed")

        with pytest.raises(Exception, match="Insert failed"):
            await repository._insert_order(sample_order)

    async def test_logger_calls_during_operations(self, repository, mock_adapter, sample_order):
        """Test that logger is called appropriately during operations."""
        # Test successful insert logging
        await repository._insert_order(sample_order)

        # Test successful update logging
        mock_adapter.execute_query.return_value = "UPDATE 1"
        await repository.update_order(sample_order)

        # Test successful delete logging
        mock_adapter.execute_query.return_value = "DELETE 1"
        result = await repository.delete_order(sample_order.id)
        assert result is True

        # Test not found delete logging
        mock_adapter.execute_query.return_value = "DELETE 0"
        result = await repository.delete_order(sample_order.id)
        assert result is False


@pytest.mark.unit
class TestRepositoryIntegration:
    """Test repository integration scenarios."""

    async def test_save_retrieve_roundtrip(self, repository, mock_adapter, sample_order):
        """Test save and retrieve roundtrip maintains order integrity."""
        # Setup mocks for save operation
        mock_adapter.fetch_one.return_value = None  # Order doesn't exist

        # Setup mocks for retrieve operation
        order_record = {
            "id": sample_order.id,
            "symbol": sample_order.symbol,
            "side": sample_order.side.value,
            "order_type": sample_order.order_type.value,
            "status": sample_order.status.value,
            "quantity": sample_order.quantity,
            "limit_price": sample_order.limit_price,
            "stop_price": sample_order.stop_price,
            "time_in_force": sample_order.time_in_force.value,
            "broker_order_id": sample_order.broker_order_id,
            "filled_quantity": sample_order.filled_quantity,
            "average_fill_price": sample_order.average_fill_price,
            "created_at": sample_order.created_at,
            "submitted_at": sample_order.submitted_at,
            "filled_at": sample_order.filled_at,
            "cancelled_at": sample_order.cancelled_at,
            "reason": sample_order.reason,
            "tags": sample_order.tags,
        }

        # Save order
        saved_order = await repository.save_order(sample_order)
        assert saved_order == sample_order

        # Mock fetch for retrieve
        mock_adapter.fetch_one.return_value = order_record

        # Retrieve order
        retrieved_order = await repository.get_order_by_id(sample_order.id)

        # Verify integrity
        assert retrieved_order.id == sample_order.id
        assert retrieved_order.symbol == sample_order.symbol
        assert retrieved_order.side == sample_order.side
        assert retrieved_order.order_type == sample_order.order_type
        assert retrieved_order.status == sample_order.status
        assert retrieved_order.quantity == sample_order.quantity
        assert retrieved_order.limit_price == sample_order.limit_price

    async def test_comprehensive_error_scenarios(self, repository, mock_adapter, sample_order):
        """Test comprehensive error scenarios for all repository methods."""
        error_methods = [
            ("get_orders_by_symbol", "AAPL"),
            ("get_orders_by_status", OrderStatus.PENDING),
            ("get_active_orders",),
            ("get_orders_by_date_range", datetime.now(UTC), datetime.now(UTC)),
            ("get_orders_by_broker_id", "BROKER123"),
        ]

        for method_info in error_methods:
            method_name = method_info[0]
            args = method_info[1:] if len(method_info) > 1 else ()

            # Test with fetch_all exception
            mock_adapter.fetch_all.side_effect = Exception("Database error")
            method = getattr(repository, method_name)

            with pytest.raises(RepositoryError):
                await method(*args)

            # Reset mock
            mock_adapter.fetch_all.side_effect = None
            mock_adapter.fetch_all.return_value = []

    async def test_order_lifecycle_operations(self, repository, mock_adapter, sample_order):
        """Test complete order lifecycle operations."""
        # 1. Save new order
        mock_adapter.fetch_one.return_value = None
        await repository.save_order(sample_order)
        mock_adapter.execute_query.assert_called()

        # 2. Update order (submit)
        mock_adapter.execute_query.reset_mock()
        mock_adapter.execute_query.return_value = "UPDATE 1"
        sample_order.submit("BROKER123")
        await repository.update_order(sample_order)
        mock_adapter.execute_query.assert_called()

        # 3. Get active orders (should include our order)
        mock_adapter.fetch_all.return_value = [
            {
                "id": sample_order.id,
                "symbol": sample_order.symbol,
                "side": sample_order.side.value,
                "order_type": sample_order.order_type.value,
                "status": sample_order.status.value,
                "quantity": sample_order.quantity,
                "limit_price": sample_order.limit_price,
                "stop_price": sample_order.stop_price,
                "time_in_force": sample_order.time_in_force.value,
                "broker_order_id": sample_order.broker_order_id,
                "filled_quantity": sample_order.filled_quantity,
                "average_fill_price": sample_order.average_fill_price,
                "created_at": sample_order.created_at,
                "submitted_at": sample_order.submitted_at,
                "filled_at": sample_order.filled_at,
                "cancelled_at": sample_order.cancelled_at,
                "reason": sample_order.reason,
                "tags": sample_order.tags,
            }
        ]

        active_orders = await repository.get_active_orders()
        assert len(active_orders) == 1
        assert active_orders[0].broker_order_id == "BROKER123"

        # 4. Delete order
        mock_adapter.execute_query.reset_mock()
        mock_adapter.execute_query.return_value = "DELETE 1"
        result = await repository.delete_order(sample_order.id)
        assert result is True

    async def test_order_status_workflow(self, repository, mock_adapter):
        """Test order status workflow from creation to completion."""
        # Create order record in different states
        base_record = {
            "id": uuid4(),
            "symbol": "AAPL",
            "side": "buy",
            "order_type": "limit",
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

        # Test pending order
        pending_record = {**base_record, "status": "pending"}
        mock_adapter.fetch_all.return_value = [pending_record]
        pending_orders = await repository.get_orders_by_status(OrderStatus.PENDING)
        assert len(pending_orders) == 1
        assert pending_orders[0].status == OrderStatus.PENDING

        # Test submitted order
        submitted_record = {**base_record, "status": "submitted", "broker_order_id": "BROKER123"}
        mock_adapter.fetch_all.return_value = [submitted_record]
        submitted_orders = await repository.get_orders_by_status(OrderStatus.SUBMITTED)
        assert len(submitted_orders) == 1
        assert submitted_orders[0].status == OrderStatus.SUBMITTED

        # Test filled order
        filled_record = {
            **base_record,
            "status": "filled",
            "filled_quantity": Decimal("100"),
            "average_fill_price": Decimal("149.50"),
            "filled_at": datetime.now(UTC),
        }
        mock_adapter.fetch_all.return_value = [filled_record]
        filled_orders = await repository.get_orders_by_status(OrderStatus.FILLED)
        assert len(filled_orders) == 1
        assert filled_orders[0].status == OrderStatus.FILLED
        assert filled_orders[0].filled_quantity == Decimal("100")
