"""
Unit tests for PostgreSQL Database Adapter.

Tests the database adapter functionality including connection management,
query execution, transaction handling, and error scenarios.
"""

# Standard library imports
import builtins
from unittest.mock import AsyncMock, Mock

# Third-party imports
import psycopg
import pytest
from psycopg_pool import AsyncConnectionPool

# Local imports
from src.application.interfaces.exceptions import (
    ConnectionError,
    IntegrityError,
    RepositoryError,
    TimeoutError,
    TransactionError,
)
from src.infrastructure.database.adapter import PostgreSQLAdapter


@pytest.fixture
def mock_pool():
    """Mock psycopg3 connection pool."""
    pool = AsyncMock(spec=AsyncConnectionPool)
    pool.max_size = 10
    pool.min_size = 1
    pool.closed = False
    return pool


@pytest.fixture
def mock_connection():
    """Mock psycopg3 connection."""
    connection = AsyncMock(spec=psycopg.AsyncConnection)
    return connection


@pytest.fixture
def mock_cursor():
    """Mock psycopg3 cursor."""
    cursor = AsyncMock()
    cursor.rowcount = 1
    return cursor


@pytest.fixture
def mock_transaction():
    """Mock psycopg3 transaction."""
    transaction = AsyncMock()
    return transaction


@pytest.fixture
def adapter(mock_pool):
    """PostgreSQL adapter with mocked pool."""
    return PostgreSQLAdapter(mock_pool)


@pytest.mark.unit
class TestPostgreSQLAdapterInitialization:
    """Test adapter initialization and properties."""

    def test_adapter_initialization(self, mock_pool):
        """Test adapter is properly initialized."""
        adapter = PostgreSQLAdapter(mock_pool)

        assert adapter.pool == mock_pool
        assert adapter._connection is None
        assert adapter._transaction is None
        assert not adapter.has_active_transaction

    def test_pool_property(self, adapter, mock_pool):
        """Test pool property access."""
        assert adapter.pool == mock_pool

    def test_has_active_transaction_property(self, adapter):
        """Test has_active_transaction property."""
        assert not adapter.has_active_transaction

        # Simulate active transaction
        adapter._transaction = Mock()
        assert adapter.has_active_transaction

        adapter._transaction = None
        assert not adapter.has_active_transaction


@pytest.mark.unit
class TestConnectionManagement:
    """Test connection acquisition and management."""

    async def test_acquire_connection_from_pool(self, adapter, mock_pool, mock_connection):
        """Test connection acquisition from pool."""
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None

        async with adapter.acquire_connection() as conn:
            assert conn == mock_connection

        mock_pool.connection.assert_called_once()

    async def test_acquire_connection_uses_existing_in_transaction(self, adapter, mock_connection):
        """Test that existing connection is used when in transaction."""
        adapter._connection = mock_connection

        async with adapter.acquire_connection() as conn:
            assert conn == mock_connection

        # Pool should not be called
        adapter._pool.connection.assert_not_called()

    async def test_acquire_connection_operational_error(self, adapter, mock_pool):
        """Test connection acquisition failure."""
        mock_pool.connection.side_effect = psycopg.OperationalError("Connection failed")

        with pytest.raises(ConnectionError, match="Failed to acquire database connection"):
            async with adapter.acquire_connection():
                pass

    async def test_acquire_connection_timeout_error(self, adapter, mock_pool):
        """Test connection acquisition timeout."""
        mock_pool.connection.side_effect = builtins.TimeoutError()

        with pytest.raises(TimeoutError, match="acquire_connection"):
            async with adapter.acquire_connection():
                pass


@pytest.mark.unit
class TestQueryExecution:
    """Test query execution methods."""

    async def test_execute_query_success(self, adapter, mock_pool, mock_connection, mock_cursor):
        """Test successful query execution."""
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None
        mock_cursor.rowcount = 5

        result = await adapter.execute_query("INSERT INTO test VALUES (%s)", "value")

        assert result == "EXECUTE 5"
        mock_cursor.execute.assert_called_once_with("INSERT INTO test VALUES (%s)", ("value",))

    async def test_execute_query_integrity_error(
        self, adapter, mock_pool, mock_connection, mock_cursor
    ):
        """Test query execution with integrity constraint violation."""
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None
        mock_cursor.execute.side_effect = psycopg.IntegrityError("Constraint violation")

        with pytest.raises(IntegrityError, match="Constraint violation"):
            await adapter.execute_query("INSERT INTO test VALUES (%s)", "value")

    async def test_execute_query_operational_error(
        self, adapter, mock_pool, mock_connection, mock_cursor
    ):
        """Test query execution with operational error."""
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None
        mock_cursor.execute.side_effect = psycopg.OperationalError("Query failed")

        with pytest.raises(RepositoryError, match="Query execution failed"):
            await adapter.execute_query("SELECT * FROM test")

    async def test_execute_query_timeout(self, adapter, mock_pool, mock_connection, mock_cursor):
        """Test query execution timeout."""
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None
        mock_cursor.execute.side_effect = builtins.TimeoutError()

        with pytest.raises(TimeoutError, match="execute_query"):
            await adapter.execute_query("SELECT * FROM test")


@pytest.mark.unit
class TestFetchOperations:
    """Test data fetching operations."""

    async def test_fetch_one_success(self, adapter, mock_pool, mock_connection, mock_cursor):
        """Test successful fetch one operation."""
        mock_row = {"id": 1, "name": "test"}

        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None
        mock_cursor.fetchone.return_value = mock_row

        result = await adapter.fetch_one("SELECT * FROM test WHERE id = %s", 1)

        assert result == mock_row
        mock_cursor.execute.assert_called_once_with("SELECT * FROM test WHERE id = %s", (1,))
        mock_cursor.fetchone.assert_called_once()

    async def test_fetch_one_not_found(self, adapter, mock_pool, mock_connection, mock_cursor):
        """Test fetch one when no record found."""
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None
        mock_cursor.fetchone.return_value = None

        result = await adapter.fetch_one("SELECT * FROM test WHERE id = %s", 999)

        assert result is None

    async def test_fetch_all_success(self, adapter, mock_pool, mock_connection, mock_cursor):
        """Test successful fetch all operation."""
        mock_rows = [{"id": 1, "name": "test1"}, {"id": 2, "name": "test2"}]

        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None
        mock_cursor.fetchall.return_value = mock_rows

        result = await adapter.fetch_all("SELECT * FROM test")

        assert result == mock_rows
        mock_cursor.fetchall.assert_called_once()

    async def test_fetch_all_empty_result(self, adapter, mock_pool, mock_connection, mock_cursor):
        """Test fetch all with empty result."""
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None
        mock_cursor.fetchall.return_value = []

        result = await adapter.fetch_all("SELECT * FROM test WHERE 1=0")

        assert result == []

    async def test_fetch_values_success(self, adapter, mock_pool, mock_connection, mock_cursor):
        """Test successful fetch values operation."""
        mock_rows = [("value1",), ("value2",), ("value3",)]

        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None
        mock_cursor.fetchall.return_value = mock_rows

        result = await adapter.fetch_values("SELECT name FROM test")

        assert result == ["value1", "value2", "value3"]

    async def test_fetch_values_index_error(self, adapter, mock_pool, mock_connection, mock_cursor):
        """Test fetch values with invalid column access."""
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None
        mock_cursor.fetchall.side_effect = IndexError("Invalid column access")

        with pytest.raises(RepositoryError, match="Fetch values query failed"):
            await adapter.fetch_values("SELECT name FROM test")


@pytest.mark.unit
class TestBatchOperations:
    """Test batch execution operations."""

    async def test_execute_batch_success(self, adapter, mock_pool, mock_connection, mock_cursor):
        """Test successful batch execution."""
        args_list = [("value1",), ("value2",), ("value3",)]

        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None

        await adapter.execute_batch("INSERT INTO test VALUES (%s)", args_list)

        mock_cursor.executemany.assert_called_once_with("INSERT INTO test VALUES (%s)", args_list)

    async def test_execute_batch_integrity_error(
        self, adapter, mock_pool, mock_connection, mock_cursor
    ):
        """Test batch execution with integrity error."""
        args_list = [("value1",), ("value2",)]

        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None
        mock_cursor.executemany.side_effect = psycopg.IntegrityError("Batch constraint violation")

        with pytest.raises(IntegrityError, match="Batch constraint violation"):
            await adapter.execute_batch("INSERT INTO test VALUES (%s)", args_list)


@pytest.mark.unit
class TestTransactionManagement:
    """Test transaction management operations."""

    async def test_begin_transaction_success(
        self, adapter, mock_pool, mock_connection, mock_transaction
    ):
        """Test successful transaction begin."""
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_connection.transaction.return_value = mock_transaction

        await adapter.begin_transaction()

        assert adapter.has_active_transaction
        assert adapter._connection == mock_connection
        assert adapter._transaction == mock_transaction
        mock_transaction.__aenter__.assert_called_once()

    async def test_begin_transaction_already_active_error(self, adapter):
        """Test beginning transaction when one is already active."""
        adapter._transaction = Mock()

        with pytest.raises(TransactionError, match="Transaction is already active"):
            await adapter.begin_transaction()

    async def test_begin_transaction_operational_error(self, adapter, mock_pool):
        """Test transaction begin with operational error."""
        mock_pool.connection.return_value.__aenter__.side_effect = psycopg.OperationalError(
            "Connection failed"
        )

        with pytest.raises(TransactionError, match="Failed to start transaction"):
            await adapter.begin_transaction()

    async def test_commit_transaction_success(self, adapter, mock_transaction, mock_connection):
        """Test successful transaction commit."""
        adapter._transaction = mock_transaction
        adapter._connection = mock_connection

        await adapter.commit_transaction()

        assert not adapter.has_active_transaction
        assert adapter._connection is None
        assert adapter._transaction is None
        mock_transaction.__aexit__.assert_called_once_with(None, None, None)

    async def test_commit_transaction_no_active_error(self, adapter):
        """Test commit when no active transaction."""
        with pytest.raises(TransactionError, match="No active transaction to commit"):
            await adapter.commit_transaction()

    async def test_commit_transaction_operational_error(
        self, adapter, mock_transaction, mock_connection
    ):
        """Test commit with operational error."""
        adapter._transaction = mock_transaction
        adapter._connection = mock_connection
        mock_transaction.__aexit__.side_effect = psycopg.OperationalError("Commit failed")

        with pytest.raises(TransactionError, match="Failed to commit transaction"):
            await adapter.commit_transaction()

    async def test_rollback_transaction_success(self, adapter, mock_transaction, mock_connection):
        """Test successful transaction rollback."""
        adapter._transaction = mock_transaction
        adapter._connection = mock_connection

        await adapter.rollback_transaction()

        assert not adapter.has_active_transaction
        assert adapter._connection is None
        assert adapter._transaction is None
        mock_transaction.__aexit__.assert_called_once_with(Exception, Exception(), None)

    async def test_rollback_transaction_no_active_warning(self, adapter):
        """Test rollback when no active transaction (should not raise error)."""
        # Should not raise error, just log warning
        await adapter.rollback_transaction()

        assert not adapter.has_active_transaction

    async def test_rollback_transaction_operational_error(
        self, adapter, mock_transaction, mock_connection
    ):
        """Test rollback with operational error."""
        adapter._transaction = mock_transaction
        adapter._connection = mock_connection
        mock_transaction.__aexit__.side_effect = psycopg.OperationalError("Rollback failed")

        with pytest.raises(TransactionError, match="Failed to rollback transaction"):
            await adapter.rollback_transaction()


@pytest.mark.unit
class TestHealthCheck:
    """Test health check functionality."""

    async def test_health_check_success(self, adapter, mock_pool, mock_connection, mock_cursor):
        """Test successful health check."""
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None
        mock_cursor.fetchone.return_value = (1,)

        result = await adapter.health_check()

        assert result is True
        mock_cursor.execute.assert_called_once_with("SELECT 1")
        mock_cursor.fetchone.assert_called_once()

    async def test_health_check_failure(self, adapter, mock_pool):
        """Test health check failure."""
        mock_pool.connection.side_effect = psycopg.OperationalError("Connection failed")

        result = await adapter.health_check()

        assert result is False


@pytest.mark.unit
class TestConnectionInfo:
    """Test connection information retrieval."""

    async def test_get_connection_info(self, adapter, mock_pool):
        """Test connection info retrieval."""
        mock_pool.max_size = 20
        mock_pool.min_size = 5
        mock_pool.closed = False

        info = await adapter.get_connection_info()

        expected = {"max_size": 20, "min_size": 5, "pool_status": "active"}
        assert info == expected

    async def test_get_connection_info_closed_pool(self, adapter, mock_pool):
        """Test connection info for closed pool."""
        mock_pool.max_size = 10
        mock_pool.min_size = 1
        mock_pool.closed = True

        info = await adapter.get_connection_info()

        expected = {"max_size": 10, "min_size": 1, "pool_status": "closed"}
        assert info == expected


@pytest.mark.unit
class TestStringRepresentation:
    """Test adapter string representation."""

    def test_str_without_transaction(self, adapter, mock_pool):
        """Test string representation without active transaction."""
        mock_pool.max_size = 10

        result = str(adapter)

        assert "PostgreSQLAdapter" in result
        assert "Pool(max_size=10)" in result
        assert "no transaction" in result

    def test_str_with_transaction(self, adapter, mock_pool):
        """Test string representation with active transaction."""
        mock_pool.max_size = 10
        adapter._transaction = Mock()

        result = str(adapter)

        assert "PostgreSQLAdapter" in result
        assert "Pool(max_size=10)" in result
        assert "with active transaction" in result


@pytest.mark.unit
class TestCleanupTransaction:
    """Test transaction cleanup functionality."""

    async def test_cleanup_transaction_with_connection(self, adapter, mock_connection):
        """Test cleanup when connection exists."""
        adapter._connection = mock_connection
        adapter._transaction = Mock()

        await adapter._cleanup_transaction()

        assert adapter._connection is None
        assert adapter._transaction is None
        mock_connection.__aexit__.assert_called_once_with(None, None, None)

    async def test_cleanup_transaction_connection_error(self, adapter, mock_connection):
        """Test cleanup when connection release fails."""
        adapter._connection = mock_connection
        adapter._transaction = Mock()
        mock_connection.__aexit__.side_effect = Exception("Release failed")

        # Should not raise error, just log warning
        await adapter._cleanup_transaction()

        assert adapter._connection is None
        assert adapter._transaction is None

    async def test_cleanup_transaction_no_connection(self, adapter):
        """Test cleanup when no connection exists."""
        adapter._connection = None
        adapter._transaction = Mock()

        await adapter._cleanup_transaction()

        assert adapter._connection is None
        assert adapter._transaction is None
