"""
Comprehensive unit tests for PostgreSQL Database Adapter.

Tests the database adapter including connection management, query execution,
transaction handling, error handling, and all database operations with full coverage.
"""

# Standard library imports
import builtins
from unittest.mock import AsyncMock, MagicMock, patch

# Third-party imports
import psycopg
import pytest
from psycopg import AsyncConnection, AsyncTransaction
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
    """Mock async connection pool."""
    pool = AsyncMock(spec=AsyncConnectionPool)
    pool.max_size = 10
    pool.min_size = 2
    pool.closed = False
    return pool


@pytest.fixture
def mock_connection():
    """Mock async database connection."""
    connection = AsyncMock(spec=AsyncConnection)
    connection.closed = False
    return connection


@pytest.fixture
def mock_cursor():
    """Mock database cursor."""
    cursor = AsyncMock()
    cursor.execute = AsyncMock()
    cursor.executemany = AsyncMock()
    cursor.fetchone = AsyncMock(return_value=None)
    cursor.fetchall = AsyncMock(return_value=[])
    cursor.rowcount = 1
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=None)
    return cursor


@pytest.fixture
def adapter(mock_pool):
    """PostgreSQL adapter with mocked pool."""
    return PostgreSQLAdapter(mock_pool)


@pytest.mark.unit
class TestAdapterInitialization:
    """Test adapter initialization and properties."""

    def test_adapter_initialization(self, mock_pool):
        """Test adapter initialization with pool."""
        adapter = PostgreSQLAdapter(mock_pool)

        assert adapter._pool is mock_pool
        assert adapter._connection is None
        assert adapter._transaction is None

    def test_pool_property(self, adapter, mock_pool):
        """Test pool property getter."""
        assert adapter.pool is mock_pool

    def test_has_active_transaction_false(self, adapter):
        """Test has_active_transaction when no transaction."""
        assert adapter.has_active_transaction is False

    def test_has_active_transaction_true(self, adapter):
        """Test has_active_transaction with active transaction."""
        adapter._transaction = MagicMock()
        assert adapter.has_active_transaction is True

    def test_string_representation(self, adapter):
        """Test string representation of adapter."""
        result = str(adapter)
        assert "PostgreSQLAdapter" in result
        assert "Pool(max_size=10)" in result
        assert "no transaction" in result

        adapter._transaction = MagicMock()
        result = str(adapter)
        assert "with active transaction" in result


@pytest.mark.unit
class TestConnectionManagement:
    """Test connection acquisition and management."""

    @pytest.mark.asyncio
    async def test_acquire_connection_from_pool(self, adapter, mock_pool, mock_connection):
        """Test acquiring connection from pool."""
        mock_pool.connection.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.connection.return_value.__aexit__ = AsyncMock(return_value=None)

        async with adapter.acquire_connection() as conn:
            assert conn is mock_connection

        mock_pool.connection.assert_called_once()

    @pytest.mark.asyncio
    async def test_acquire_connection_with_existing_connection(self, adapter, mock_connection):
        """Test acquiring connection when one already exists (transaction)."""
        adapter._connection = mock_connection

        async with adapter.acquire_connection() as conn:
            assert conn is mock_connection

        # Should not acquire new connection from pool

    @pytest.mark.asyncio
    async def test_acquire_connection_operational_error(self, adapter, mock_pool):
        """Test connection acquisition with operational error."""
        mock_pool.connection.side_effect = psycopg.OperationalError("Connection failed")

        with pytest.raises(ConnectionError, match="Failed to acquire database connection"):
            async with adapter.acquire_connection():
                pass

    @pytest.mark.asyncio
    async def test_acquire_connection_timeout(self, adapter, mock_pool):
        """Test connection acquisition timeout."""
        mock_pool.connection.side_effect = builtins.TimeoutError("Timeout")

        with pytest.raises(TimeoutError) as exc_info:
            async with adapter.acquire_connection():
                pass

        assert exc_info.operation == "acquire_connection"


@pytest.mark.unit
class TestQueryExecution:
    """Test query execution methods."""

    @pytest.mark.asyncio
    async def test_execute_query_success(self, adapter, mock_pool, mock_connection, mock_cursor):
        """Test successful query execution."""
        mock_pool.connection.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.connection.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.rowcount = 5

        result = await adapter.execute_query("UPDATE table SET x = %s", 1)

        assert result == "EXECUTE 5"
        mock_cursor.execute.assert_called_once_with("UPDATE table SET x = %s", (1,))

    @pytest.mark.asyncio
    async def test_execute_query_integrity_error(
        self, adapter, mock_pool, mock_connection, mock_cursor
    ):
        """Test query execution with integrity constraint violation."""
        mock_pool.connection.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.connection.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.execute.side_effect = psycopg.IntegrityError("Unique constraint violated")

        with pytest.raises(IntegrityError, match="Unique constraint violated"):
            await adapter.execute_query("INSERT INTO table VALUES (%s)", 1)

    @pytest.mark.asyncio
    async def test_execute_query_operational_error(
        self, adapter, mock_pool, mock_connection, mock_cursor
    ):
        """Test query execution with operational error."""
        mock_pool.connection.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.connection.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.execute.side_effect = psycopg.OperationalError("Query failed")

        with pytest.raises(RepositoryError, match="Query execution failed"):
            await adapter.execute_query("SELECT * FROM table")

    @pytest.mark.asyncio
    async def test_execute_query_timeout(self, adapter, mock_pool, mock_connection, mock_cursor):
        """Test query execution timeout."""
        mock_pool.connection.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.connection.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.execute.side_effect = builtins.TimeoutError("Query timeout")

        with pytest.raises(TimeoutError) as exc_info:
            await adapter.execute_query("SELECT * FROM large_table", command_timeout=5.0)

        assert exc_info.operation == "execute_query"
        assert exc_info.timeout == 5.0

    @pytest.mark.asyncio
    async def test_fetch_one_success(self, adapter, mock_pool, mock_connection, mock_cursor):
        """Test successful fetch one operation."""
        mock_pool.connection.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.connection.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = {"id": 1, "name": "Test"}

        result = await adapter.fetch_one("SELECT * FROM table WHERE id = %s", 1)

        assert result == {"id": 1, "name": "Test"}
        mock_cursor.execute.assert_called_once_with("SELECT * FROM table WHERE id = %s", (1,))
        mock_cursor.fetchone.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_one_not_found(self, adapter, mock_pool, mock_connection, mock_cursor):
        """Test fetch one when no record found."""
        mock_pool.connection.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.connection.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None

        result = await adapter.fetch_one("SELECT * FROM table WHERE id = %s", 999)

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_one_error(self, adapter, mock_pool, mock_connection, mock_cursor):
        """Test fetch one with error."""
        mock_pool.connection.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.connection.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.execute.side_effect = psycopg.OperationalError("Fetch failed")

        with pytest.raises(RepositoryError, match="Fetch one query failed"):
            await adapter.fetch_one("SELECT * FROM table")

    @pytest.mark.asyncio
    async def test_fetch_all_success(self, adapter, mock_pool, mock_connection, mock_cursor):
        """Test successful fetch all operation."""
        mock_pool.connection.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.connection.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            {"id": 1, "name": "Test1"},
            {"id": 2, "name": "Test2"},
        ]

        result = await adapter.fetch_all("SELECT * FROM table")

        assert len(result) == 2
        assert result[0]["name"] == "Test1"
        assert result[1]["name"] == "Test2"
        mock_cursor.fetchall.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_all_empty(self, adapter, mock_pool, mock_connection, mock_cursor):
        """Test fetch all with no results."""
        mock_pool.connection.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.connection.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []

        result = await adapter.fetch_all("SELECT * FROM empty_table")

        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_all_error(self, adapter, mock_pool, mock_connection, mock_cursor):
        """Test fetch all with error."""
        mock_pool.connection.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.connection.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.execute.side_effect = psycopg.OperationalError("Fetch all failed")

        with pytest.raises(RepositoryError, match="Fetch all query failed"):
            await adapter.fetch_all("SELECT * FROM table")

    @pytest.mark.asyncio
    async def test_fetch_values_success(self, adapter, mock_pool, mock_connection, mock_cursor):
        """Test successful fetch values operation."""
        mock_pool.connection.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.connection.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            {"value": 1},
            {"value": 2},
            {"value": 3},
        ]

        result = await adapter.fetch_values("SELECT value FROM table")

        assert result == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_fetch_values_empty(self, adapter, mock_pool, mock_connection, mock_cursor):
        """Test fetch values with no results."""
        mock_pool.connection.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.connection.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []

        result = await adapter.fetch_values("SELECT value FROM empty_table")

        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_values_error(self, adapter, mock_pool, mock_connection, mock_cursor):
        """Test fetch values with error."""
        mock_pool.connection.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.connection.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.execute.side_effect = psycopg.OperationalError("Fetch values failed")

        with pytest.raises(RepositoryError, match="Fetch values query failed"):
            await adapter.fetch_values("SELECT value FROM table")

    @pytest.mark.asyncio
    async def test_execute_batch_success(self, adapter, mock_pool, mock_connection, mock_cursor):
        """Test successful batch execution."""
        mock_pool.connection.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.connection.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_connection.cursor.return_value = mock_cursor

        args_list = [(1, "A"), (2, "B"), (3, "C")]
        await adapter.execute_batch("INSERT INTO table VALUES (%s, %s)", args_list)

        mock_cursor.executemany.assert_called_once_with(
            "INSERT INTO table VALUES (%s, %s)", args_list
        )

    @pytest.mark.asyncio
    async def test_execute_batch_integrity_error(
        self, adapter, mock_pool, mock_connection, mock_cursor
    ):
        """Test batch execution with integrity error."""
        mock_pool.connection.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.connection.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.executemany.side_effect = psycopg.IntegrityError("Duplicate key")

        with pytest.raises(IntegrityError, match="Duplicate key"):
            await adapter.execute_batch("INSERT INTO table VALUES (%s)", [(1,), (1,)])

    @pytest.mark.asyncio
    async def test_execute_batch_error(self, adapter, mock_pool, mock_connection, mock_cursor):
        """Test batch execution with error."""
        mock_pool.connection.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.connection.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.executemany.side_effect = psycopg.OperationalError("Batch failed")

        with pytest.raises(RepositoryError, match="Batch execution failed"):
            await adapter.execute_batch("INSERT INTO table VALUES (%s)", [(1,)])


@pytest.mark.unit
class TestTransactionManagement:
    """Test transaction management."""

    @pytest.mark.asyncio
    async def test_begin_transaction_success(self, adapter, mock_pool):
        """Test successful transaction start."""
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_transaction = AsyncMock(spec=AsyncTransaction)
        mock_pool.connection.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_connection.transaction.return_value.__aenter__ = AsyncMock(
            return_value=mock_transaction
        )

        await adapter.begin_transaction()

        assert adapter._connection is mock_connection
        assert adapter._transaction is mock_transaction

    @pytest.mark.asyncio
    async def test_begin_transaction_already_active(self, adapter):
        """Test starting transaction when one is already active."""
        adapter._transaction = MagicMock()

        with pytest.raises(TransactionError, match="Transaction is already active"):
            await adapter.begin_transaction()

    @pytest.mark.asyncio
    async def test_begin_transaction_error(self, adapter, mock_pool):
        """Test transaction start with error."""
        mock_pool.connection.side_effect = psycopg.OperationalError("Connection failed")

        with pytest.raises(TransactionError, match="Failed to start transaction"):
            await adapter.begin_transaction()

        # Should cleanup on error
        assert adapter._connection is None
        assert adapter._transaction is None

    @pytest.mark.asyncio
    async def test_commit_transaction_success(self, adapter):
        """Test successful transaction commit."""
        mock_transaction = AsyncMock()
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        adapter._transaction = mock_transaction
        adapter._connection = AsyncMock()
        adapter._connection.__aexit__ = AsyncMock(return_value=None)

        await adapter.commit_transaction()

        mock_transaction.__aexit__.assert_called_once_with(None, None, None)
        assert adapter._transaction is None
        assert adapter._connection is None

    @pytest.mark.asyncio
    async def test_commit_transaction_no_active(self, adapter):
        """Test committing when no transaction is active."""
        with pytest.raises(TransactionError, match="No active transaction to commit"):
            await adapter.commit_transaction()

    @pytest.mark.asyncio
    async def test_commit_transaction_error(self, adapter):
        """Test transaction commit with error."""
        mock_transaction = AsyncMock()
        mock_transaction.__aexit__.side_effect = psycopg.OperationalError("Commit failed")
        adapter._transaction = mock_transaction
        adapter._connection = AsyncMock()
        adapter._connection.__aexit__ = AsyncMock(return_value=None)

        with pytest.raises(TransactionError, match="Failed to commit transaction"):
            await adapter.commit_transaction()

        # Should cleanup even on error
        assert adapter._transaction is None
        assert adapter._connection is None

    @pytest.mark.asyncio
    async def test_rollback_transaction_success(self, adapter):
        """Test successful transaction rollback."""
        mock_transaction = AsyncMock()
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        adapter._transaction = mock_transaction
        adapter._connection = AsyncMock()
        adapter._connection.__aexit__ = AsyncMock(return_value=None)

        await adapter.rollback_transaction()

        # Rollback passes Exception to __aexit__
        call_args = mock_transaction.__aexit__.call_args
        assert call_args[0][0] == Exception
        assert isinstance(call_args[0][1], Exception)
        assert adapter._transaction is None
        assert adapter._connection is None

    @pytest.mark.asyncio
    async def test_rollback_transaction_no_active(self, adapter):
        """Test rollback when no transaction is active."""
        # Should not raise error, just log warning
        await adapter.rollback_transaction()

    @pytest.mark.asyncio
    async def test_rollback_transaction_error(self, adapter):
        """Test transaction rollback with error."""
        mock_transaction = AsyncMock()
        mock_transaction.__aexit__.side_effect = psycopg.OperationalError("Rollback failed")
        adapter._transaction = mock_transaction
        adapter._connection = AsyncMock()
        adapter._connection.__aexit__ = AsyncMock(return_value=None)

        with pytest.raises(TransactionError, match="Failed to rollback transaction"):
            await adapter.rollback_transaction()

        # Should cleanup even on error
        assert adapter._transaction is None
        assert adapter._connection is None

    @pytest.mark.asyncio
    async def test_cleanup_transaction(self, adapter):
        """Test transaction cleanup."""
        mock_connection = AsyncMock()
        mock_connection.__aexit__ = AsyncMock(return_value=None)
        adapter._connection = mock_connection
        adapter._transaction = MagicMock()

        await adapter._cleanup_transaction()

        mock_connection.__aexit__.assert_called_once()
        assert adapter._connection is None
        assert adapter._transaction is None

    @pytest.mark.asyncio
    async def test_cleanup_transaction_connection_error(self, adapter):
        """Test transaction cleanup with connection release error."""
        mock_connection = AsyncMock()
        mock_connection.__aexit__.side_effect = Exception("Release failed")
        adapter._connection = mock_connection
        adapter._transaction = MagicMock()

        # Should not raise error, just log warning
        await adapter._cleanup_transaction()

        assert adapter._connection is None
        assert adapter._transaction is None


@pytest.mark.unit
class TestHealthCheck:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_success(self, adapter, mock_pool, mock_connection, mock_cursor):
        """Test successful health check."""
        mock_pool.connection.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.connection.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = {"?column?": 1}

        result = await adapter.health_check()

        assert result is True
        mock_cursor.execute.assert_called_once_with("SELECT 1")

    @pytest.mark.asyncio
    async def test_health_check_failure(self, adapter, mock_pool):
        """Test health check failure."""
        mock_pool.connection.side_effect = Exception("Database unavailable")

        result = await adapter.health_check()

        assert result is False


@pytest.mark.unit
class TestConnectionInfo:
    """Test connection info retrieval."""

    @pytest.mark.asyncio
    async def test_get_connection_info_active(self, adapter, mock_pool):
        """Test getting connection info when pool is active."""
        mock_pool.closed = False

        info = await adapter.get_connection_info()

        assert info["max_size"] == 10
        assert info["min_size"] == 2
        assert info["pool_status"] == "active"

    @pytest.mark.asyncio
    async def test_get_connection_info_closed(self, adapter, mock_pool):
        """Test getting connection info when pool is closed."""
        mock_pool.closed = True

        info = await adapter.get_connection_info()

        assert info["pool_status"] == "closed"


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_multiple_parameters_in_query(
        self, adapter, mock_pool, mock_connection, mock_cursor
    ):
        """Test query with multiple parameters."""
        mock_pool.connection.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.connection.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_connection.cursor.return_value = mock_cursor

        await adapter.execute_query(
            "INSERT INTO table (a, b, c, d) VALUES (%s, %s, %s, %s)", 1, "test", 3.14, True
        )

        mock_cursor.execute.assert_called_once_with(
            "INSERT INTO table (a, b, c, d) VALUES (%s, %s, %s, %s)", (1, "test", 3.14, True)
        )

    @pytest.mark.asyncio
    async def test_query_with_no_parameters(self, adapter, mock_pool, mock_connection, mock_cursor):
        """Test query with no parameters."""
        mock_pool.connection.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.connection.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_connection.cursor.return_value = mock_cursor

        await adapter.execute_query("TRUNCATE TABLE test_table")

        mock_cursor.execute.assert_called_once_with("TRUNCATE TABLE test_table", ())

    @pytest.mark.asyncio
    async def test_long_query_truncation_in_logs(
        self, adapter, mock_pool, mock_connection, mock_cursor
    ):
        """Test that long queries are truncated in logs."""
        mock_pool.connection.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.connection.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_connection.cursor.return_value = mock_cursor

        long_query = "SELECT " + ", ".join([f"column_{i}" for i in range(100)]) + " FROM table"

        with patch("src.infrastructure.database.adapter.logger") as mock_logger:
            await adapter.execute_query(long_query)

            # Verify query is truncated in log message
            log_call = mock_logger.debug.call_args[0][0]
            assert "..." in log_call
            assert len(log_call) < len(long_query) + 50  # Some buffer for the message
