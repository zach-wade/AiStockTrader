"""
Comprehensive unit tests for PostgreSQL Database Adapter.

Tests cover:
- Connection pool management
- Query execution
- Transaction management
- Error handling
- Health checks
- Batch operations
"""

import builtins
from unittest.mock import AsyncMock, MagicMock

import psycopg
import pytest
from psycopg import AsyncConnection, AsyncTransaction
from psycopg_pool import AsyncConnectionPool

from src.application.interfaces.exceptions import (
    ConnectionError,
    IntegrityError,
    TimeoutError,
    TransactionError,
)
from src.infrastructure.database.adapter import PostgreSQLAdapter


class TestPostgreSQLAdapterInitialization:
    """Test PostgreSQLAdapter initialization."""

    def test_initialization(self):
        """Test adapter initialization with pool."""
        mock_pool = MagicMock(spec=AsyncConnectionPool)
        adapter = PostgreSQLAdapter(mock_pool)

        assert adapter._pool == mock_pool
        assert adapter._connection is None
        assert adapter._transaction is None

    def test_pool_property(self):
        """Test pool property getter."""
        mock_pool = MagicMock(spec=AsyncConnectionPool)
        adapter = PostgreSQLAdapter(mock_pool)

        assert adapter.pool == mock_pool

    def test_has_active_transaction_false(self):
        """Test has_active_transaction when no transaction."""
        mock_pool = MagicMock(spec=AsyncConnectionPool)
        adapter = PostgreSQLAdapter(mock_pool)

        assert adapter.has_active_transaction is False

    def test_has_active_transaction_true(self):
        """Test has_active_transaction with active transaction."""
        mock_pool = MagicMock(spec=AsyncConnectionPool)
        adapter = PostgreSQLAdapter(mock_pool)
        adapter._transaction = MagicMock()

        assert adapter.has_active_transaction is True


class TestPostgreSQLAdapterConnectionManagement:
    """Test connection management."""

    @pytest.mark.asyncio
    async def test_acquire_connection_from_pool(self):
        """Test acquiring connection from pool."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)

        # Mock the connection context manager
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None

        adapter = PostgreSQLAdapter(mock_pool)

        async with adapter.acquire_connection() as conn:
            assert conn == mock_connection

        mock_pool.connection.assert_called_once()

    @pytest.mark.asyncio
    async def test_acquire_connection_uses_existing_when_in_transaction(self):
        """Test using existing connection when in transaction."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_existing_connection = AsyncMock(spec=AsyncConnection)

        adapter = PostgreSQLAdapter(mock_pool)
        adapter._connection = mock_existing_connection

        async with adapter.acquire_connection() as conn:
            assert conn == mock_existing_connection

        # Pool should not be called
        mock_pool.connection.assert_not_called()

    @pytest.mark.asyncio
    async def test_acquire_connection_operational_error(self):
        """Test connection acquisition with operational error."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_pool.connection.side_effect = psycopg.OperationalError("Connection failed")

        adapter = PostgreSQLAdapter(mock_pool)

        with pytest.raises(ConnectionError, match="Failed to acquire database connection"):
            async with adapter.acquire_connection():
                pass

    @pytest.mark.asyncio
    async def test_acquire_connection_timeout(self):
        """Test connection acquisition timeout."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_pool.connection.side_effect = builtins.TimeoutError("Timeout")

        adapter = PostgreSQLAdapter(mock_pool)

        with pytest.raises(TimeoutError) as exc_info:
            async with adapter.acquire_connection():
                pass

        # exc_info.value contains the actual exception
        assert str(exc_info.value) == "Timeout"


class TestPostgreSQLAdapterQueryExecution:
    """Test query execution methods."""

    @pytest.mark.asyncio
    async def test_execute_query_success(self):
        """Test successful query execution."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_cursor = AsyncMock()
        mock_cursor.rowcount = 5

        # Setup connection context
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None

        # Setup cursor context
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None

        adapter = PostgreSQLAdapter(mock_pool)

        result = await adapter.execute_query(
            "UPDATE users SET active = %s WHERE id = %s", True, 123
        )

        assert result == "EXECUTE 5"
        mock_cursor.execute.assert_called_once_with(
            "UPDATE users SET active = %s WHERE id = %s", (True, 123)
        )

    @pytest.mark.asyncio
    async def test_execute_query_integrity_error(self):
        """Test query execution with integrity error."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_cursor = AsyncMock()
        mock_cursor.execute.side_effect = psycopg.IntegrityError("Duplicate key")

        # Setup contexts
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None

        adapter = PostgreSQLAdapter(mock_pool)

        with pytest.raises(IntegrityError, match="Duplicate key"):
            await adapter.execute_query("INSERT INTO users VALUES (%s)", "test")

    @pytest.mark.asyncio
    async def test_execute_query_operational_error(self):
        """Test query execution with operational error."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_cursor = AsyncMock()
        mock_cursor.execute.side_effect = psycopg.OperationalError("Query failed")

        # Setup contexts
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None

        adapter = PostgreSQLAdapter(mock_pool)

        with pytest.raises(ConnectionError, match="Failed to acquire database connection"):
            await adapter.execute_query("SELECT * FROM invalid_table")

    @pytest.mark.asyncio
    async def test_execute_query_timeout(self):
        """Test query execution timeout."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_cursor = AsyncMock()
        mock_cursor.execute.side_effect = builtins.TimeoutError("Query timeout")

        # Setup contexts
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None

        adapter = PostgreSQLAdapter(mock_pool)

        with pytest.raises(TimeoutError) as exc_info:
            await adapter.execute_query("SELECT * FROM large_table", command_timeout=10.0)

        assert exc_info.operation == "acquire_connection"
        assert exc_info.timeout_seconds == 30.0  # Default timeout in acquire_connection


class TestPostgreSQLAdapterFetchMethods:
    """Test data fetching methods."""

    @pytest.mark.asyncio
    async def test_fetch_one_found(self):
        """Test fetching single record that exists."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_cursor = AsyncMock()
        mock_cursor.fetchone.return_value = {"id": 1, "name": "Test"}

        # Setup contexts
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None

        adapter = PostgreSQLAdapter(mock_pool)

        result = await adapter.fetch_one("SELECT * FROM users WHERE id = %s", 1)

        assert result == {"id": 1, "name": "Test"}
        mock_cursor.execute.assert_called_once_with("SELECT * FROM users WHERE id = %s", (1,))

    @pytest.mark.asyncio
    async def test_fetch_one_not_found(self):
        """Test fetching single record that doesn't exist."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_cursor = AsyncMock()
        mock_cursor.fetchone.return_value = None

        # Setup contexts
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None

        adapter = PostgreSQLAdapter(mock_pool)

        result = await adapter.fetch_one("SELECT * FROM users WHERE id = %s", 999)

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_all_success(self):
        """Test fetching multiple records."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_cursor = AsyncMock()
        mock_cursor.fetchall.return_value = [
            {"id": 1, "name": "User1"},
            {"id": 2, "name": "User2"},
            {"id": 3, "name": "User3"},
        ]

        # Setup contexts
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None

        adapter = PostgreSQLAdapter(mock_pool)

        result = await adapter.fetch_all("SELECT * FROM users")

        assert len(result) == 3
        assert result[0]["name"] == "User1"
        assert result[2]["id"] == 3

    @pytest.mark.asyncio
    async def test_fetch_all_empty(self):
        """Test fetching when no records exist."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_cursor = AsyncMock()
        mock_cursor.fetchall.return_value = []

        # Setup contexts
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None

        adapter = PostgreSQLAdapter(mock_pool)

        result = await adapter.fetch_all("SELECT * FROM empty_table")

        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_values_success(self):
        """Test fetching values from single column."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_cursor = AsyncMock()
        mock_cursor.fetchall.return_value = [{"user_id": 1}, {"user_id": 2}, {"user_id": 3}]

        # Setup contexts
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None

        adapter = PostgreSQLAdapter(mock_pool)

        result = await adapter.fetch_values("SELECT user_id FROM users")

        assert result == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_fetch_values_empty(self):
        """Test fetching values when no records."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_cursor = AsyncMock()
        mock_cursor.fetchall.return_value = []

        # Setup contexts
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None

        adapter = PostgreSQLAdapter(mock_pool)

        result = await adapter.fetch_values("SELECT id FROM empty_table")

        assert result == []


class TestPostgreSQLAdapterBatchOperations:
    """Test batch execution."""

    @pytest.mark.asyncio
    async def test_execute_batch_success(self):
        """Test successful batch execution."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_cursor = AsyncMock()

        # Setup contexts
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None

        adapter = PostgreSQLAdapter(mock_pool)

        args_list = [("User1", 1), ("User2", 2), ("User3", 3)]

        await adapter.execute_batch("UPDATE users SET name = %s WHERE id = %s", args_list)

        mock_cursor.executemany.assert_called_once_with(
            "UPDATE users SET name = %s WHERE id = %s", args_list
        )

    @pytest.mark.asyncio
    async def test_execute_batch_integrity_error(self):
        """Test batch execution with integrity error."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_cursor = AsyncMock()
        mock_cursor.executemany.side_effect = psycopg.IntegrityError("Constraint violation")

        # Setup contexts
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None

        adapter = PostgreSQLAdapter(mock_pool)

        with pytest.raises(IntegrityError, match="Constraint violation"):
            await adapter.execute_batch(
                "INSERT INTO users VALUES (%s, %s)",
                [(1, "User1"), (1, "User2")],  # Duplicate IDs
            )


class TestPostgreSQLAdapterTransactionManagement:
    """Test transaction management."""

    @pytest.mark.asyncio
    async def test_begin_transaction_success(self):
        """Test beginning a transaction."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_transaction = AsyncMock(spec=AsyncTransaction)

        # Setup connection acquisition
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection

        # Setup transaction context
        mock_connection.transaction.return_value.__aenter__.return_value = mock_transaction

        adapter = PostgreSQLAdapter(mock_pool)

        await adapter.begin_transaction()

        assert adapter._connection == mock_connection
        assert adapter._transaction == mock_transaction
        assert adapter.has_active_transaction is True

    @pytest.mark.asyncio
    async def test_begin_transaction_already_active(self):
        """Test beginning transaction when one is already active."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        adapter = PostgreSQLAdapter(mock_pool)
        adapter._transaction = MagicMock()  # Simulate active transaction

        with pytest.raises(TransactionError, match="Transaction is already active"):
            await adapter.begin_transaction()

    @pytest.mark.asyncio
    async def test_begin_transaction_operational_error(self):
        """Test transaction begin with operational error."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_pool.connection.return_value.__aenter__.side_effect = psycopg.OperationalError(
            "Failed"
        )

        adapter = PostgreSQLAdapter(mock_pool)

        with pytest.raises(TransactionError, match="Failed to start transaction"):
            await adapter.begin_transaction()

    @pytest.mark.asyncio
    async def test_commit_transaction_success(self):
        """Test committing a transaction."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_transaction = AsyncMock(spec=AsyncTransaction)

        adapter = PostgreSQLAdapter(mock_pool)
        adapter._connection = mock_connection
        adapter._transaction = mock_transaction

        await adapter.commit_transaction()

        mock_transaction.__aexit__.assert_called_once_with(None, None, None)
        mock_connection.__aexit__.assert_called_once()
        assert adapter._connection is None
        assert adapter._transaction is None

    @pytest.mark.asyncio
    async def test_commit_transaction_no_active(self):
        """Test committing when no transaction is active."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        adapter = PostgreSQLAdapter(mock_pool)

        with pytest.raises(TransactionError, match="No active transaction to commit"):
            await adapter.commit_transaction()

    @pytest.mark.asyncio
    async def test_rollback_transaction_success(self):
        """Test rolling back a transaction."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_transaction = AsyncMock(spec=AsyncTransaction)

        adapter = PostgreSQLAdapter(mock_pool)
        adapter._connection = mock_connection
        adapter._transaction = mock_transaction

        await adapter.rollback_transaction()

        mock_transaction.__aexit__.assert_called_once()
        # Check that Exception was passed to trigger rollback
        call_args = mock_transaction.__aexit__.call_args
        assert call_args[0][0] == Exception
        assert isinstance(call_args[0][1], Exception)

        assert adapter._connection is None
        assert adapter._transaction is None

    @pytest.mark.asyncio
    async def test_rollback_transaction_no_active(self):
        """Test rollback when no transaction is active."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        adapter = PostgreSQLAdapter(mock_pool)

        # Should not raise, just log warning
        await adapter.rollback_transaction()

        assert adapter._connection is None
        assert adapter._transaction is None


class TestPostgreSQLAdapterHealthCheck:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_cursor = AsyncMock()
        mock_cursor.fetchone.return_value = {"?column?": 1}

        # Setup contexts
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None

        adapter = PostgreSQLAdapter(mock_pool)

        result = await adapter.health_check()

        assert result is True
        mock_cursor.execute.assert_called_once_with("SELECT 1")

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test failed health check."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_pool.connection.side_effect = psycopg.OperationalError("Connection failed")

        adapter = PostgreSQLAdapter(mock_pool)

        result = await adapter.health_check()

        assert result is False


class TestPostgreSQLAdapterConnectionInfo:
    """Test connection info retrieval."""

    @pytest.mark.asyncio
    async def test_get_connection_info_active(self):
        """Test getting connection info for active pool."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_pool.max_size = 10
        mock_pool.min_size = 2
        mock_pool.closed = False

        adapter = PostgreSQLAdapter(mock_pool)

        info = await adapter.get_connection_info()

        assert info["max_size"] == 10
        assert info["min_size"] == 2
        assert info["pool_status"] == "active"

    @pytest.mark.asyncio
    async def test_get_connection_info_closed(self):
        """Test getting connection info for closed pool."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_pool.max_size = 5
        mock_pool.min_size = 1
        mock_pool.closed = True

        adapter = PostgreSQLAdapter(mock_pool)

        info = await adapter.get_connection_info()

        assert info["max_size"] == 5
        assert info["min_size"] == 1
        assert info["pool_status"] == "closed"


class TestPostgreSQLAdapterStringRepresentation:
    """Test string representation."""

    def test_str_without_transaction(self):
        """Test string representation without active transaction."""
        mock_pool = MagicMock(spec=AsyncConnectionPool)
        mock_pool.max_size = 10

        adapter = PostgreSQLAdapter(mock_pool)

        result = str(adapter)

        assert "PostgreSQLAdapter" in result
        assert "Pool(max_size=10)" in result
        assert "no transaction" in result

    def test_str_with_transaction(self):
        """Test string representation with active transaction."""
        mock_pool = MagicMock(spec=AsyncConnectionPool)
        mock_pool.max_size = 5

        adapter = PostgreSQLAdapter(mock_pool)
        adapter._transaction = MagicMock()

        result = str(adapter)

        assert "PostgreSQLAdapter" in result
        assert "Pool(max_size=5)" in result
        assert "with active transaction" in result
