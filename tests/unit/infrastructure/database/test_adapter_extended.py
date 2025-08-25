"""
Extended unit tests for PostgreSQL Database Adapter.

Additional test coverage for edge cases, error handling, and concurrent operations.
"""

import asyncio
import builtins
from unittest.mock import AsyncMock, Mock, patch

import psycopg
import pytest
from psycopg import AsyncConnection, AsyncTransaction
from psycopg_pool import AsyncConnectionPool

from src.application.interfaces.exceptions import (
    IntegrityError,
    RepositoryError,
    TimeoutError,
    TransactionError,
)
from src.infrastructure.database.adapter import PostgreSQLAdapter


class TestPostgreSQLAdapterEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.mark.asyncio
    async def test_fetch_one_operational_error(self):
        """Test fetch_one with operational error."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_cursor = AsyncMock()
        mock_cursor.execute.side_effect = psycopg.OperationalError("Database unavailable")

        # Setup contexts
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None

        adapter = PostgreSQLAdapter(mock_pool)

        with pytest.raises(RepositoryError, match="Fetch one query failed"):
            await adapter.fetch_one("SELECT * FROM users WHERE id = %s", 1)

    @pytest.mark.asyncio
    async def test_fetch_one_timeout_error(self):
        """Test fetch_one with timeout."""
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
            await adapter.fetch_one("SELECT * FROM large_table", command_timeout=5.0)

        assert exc_info.operation == "fetch_one"
        assert exc_info.timeout_seconds == 5.0

    @pytest.mark.asyncio
    async def test_fetch_all_operational_error(self):
        """Test fetch_all with operational error."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_cursor = AsyncMock()
        mock_cursor.execute.side_effect = psycopg.OperationalError("Connection lost")

        # Setup contexts
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None

        adapter = PostgreSQLAdapter(mock_pool)

        with pytest.raises(RepositoryError, match="Fetch all query failed"):
            await adapter.fetch_all("SELECT * FROM users")

    @pytest.mark.asyncio
    async def test_fetch_all_timeout_with_custom_timeout(self):
        """Test fetch_all with custom timeout value."""
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
            await adapter.fetch_all("SELECT * FROM huge_table", command_timeout=120.0)

        assert exc_info.operation == "fetch_all"
        assert exc_info.timeout_seconds == 120.0

    @pytest.mark.asyncio
    async def test_fetch_values_with_multiple_columns(self):
        """Test fetch_values extracts first column correctly."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_cursor = AsyncMock()
        mock_cursor.fetchall.return_value = [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25},
            {"id": 3, "name": "Charlie", "age": 35},
        ]

        # Setup contexts
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None

        adapter = PostgreSQLAdapter(mock_pool)

        result = await adapter.fetch_values("SELECT id, name, age FROM users")

        # Should extract first column (id)
        assert result == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_fetch_values_index_error(self):
        """Test fetch_values with malformed results."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_cursor = AsyncMock()
        # Return invalid structure
        mock_cursor.fetchall.return_value = [{}]  # Empty dict, no keys

        # Setup contexts
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None

        adapter = PostgreSQLAdapter(mock_pool)

        with pytest.raises(RepositoryError, match="Fetch values query failed"):
            await adapter.fetch_values("SELECT * FROM invalid")

    @pytest.mark.asyncio
    async def test_fetch_values_operational_error(self):
        """Test fetch_values with operational error."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_cursor = AsyncMock()
        mock_cursor.execute.side_effect = psycopg.OperationalError("Query error")

        # Setup contexts
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None

        adapter = PostgreSQLAdapter(mock_pool)

        with pytest.raises(RepositoryError, match="Fetch values query failed"):
            await adapter.fetch_values("SELECT id FROM users")

    @pytest.mark.asyncio
    async def test_fetch_values_timeout(self):
        """Test fetch_values with timeout."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_cursor = AsyncMock()
        mock_cursor.execute.side_effect = builtins.TimeoutError("Timeout")

        # Setup contexts
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None

        adapter = PostgreSQLAdapter(mock_pool)

        with pytest.raises(TimeoutError) as exc_info:
            await adapter.fetch_values("SELECT id FROM users", command_timeout=10.0)

        assert exc_info.operation == "fetch_values"
        assert exc_info.timeout_seconds == 10.0

    @pytest.mark.asyncio
    async def test_execute_batch_empty_args_list(self):
        """Test execute_batch with empty arguments list."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_cursor = AsyncMock()

        # Setup contexts
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None

        adapter = PostgreSQLAdapter(mock_pool)

        await adapter.execute_batch("UPDATE users SET active = %s", [])

        mock_cursor.executemany.assert_called_once_with("UPDATE users SET active = %s", [])

    @pytest.mark.asyncio
    async def test_execute_batch_operational_error(self):
        """Test execute_batch with operational error."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_cursor = AsyncMock()
        mock_cursor.executemany.side_effect = psycopg.OperationalError("Batch failed")

        # Setup contexts
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None

        adapter = PostgreSQLAdapter(mock_pool)

        with pytest.raises(RepositoryError, match="Batch execution failed"):
            await adapter.execute_batch(
                "INSERT INTO users VALUES (%s, %s)", [(1, "User1"), (2, "User2")]
            )

    @pytest.mark.asyncio
    async def test_execute_batch_timeout(self):
        """Test execute_batch with timeout."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_cursor = AsyncMock()
        mock_cursor.executemany.side_effect = builtins.TimeoutError("Batch timeout")

        # Setup contexts
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None

        adapter = PostgreSQLAdapter(mock_pool)

        with pytest.raises(TimeoutError) as exc_info:
            await adapter.execute_batch(
                "UPDATE users SET status = %s", [("active",)] * 1000, command_timeout=60.0
            )

        assert exc_info.operation == "execute_batch"
        assert exc_info.timeout_seconds == 60.0


class TestPostgreSQLAdapterTransactionEdgeCases:
    """Test transaction edge cases and complex scenarios."""

    @pytest.mark.asyncio
    async def test_commit_transaction_operational_error(self):
        """Test commit with operational error."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_transaction = AsyncMock(spec=AsyncTransaction)
        mock_transaction.__aexit__.side_effect = psycopg.OperationalError("Commit failed")

        adapter = PostgreSQLAdapter(mock_pool)
        adapter._connection = mock_connection
        adapter._transaction = mock_transaction

        with pytest.raises(TransactionError, match="Failed to commit transaction"):
            await adapter.commit_transaction()

        # Cleanup should still happen
        mock_connection.__aexit__.assert_called_once()
        assert adapter._connection is None
        assert adapter._transaction is None

    @pytest.mark.asyncio
    async def test_rollback_transaction_operational_error(self):
        """Test rollback with operational error."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_transaction = AsyncMock(spec=AsyncTransaction)
        mock_transaction.__aexit__.side_effect = psycopg.OperationalError("Rollback failed")

        adapter = PostgreSQLAdapter(mock_pool)
        adapter._connection = mock_connection
        adapter._transaction = mock_transaction

        with pytest.raises(TransactionError, match="Failed to rollback transaction"):
            await adapter.rollback_transaction()

        # Cleanup should still happen
        mock_connection.__aexit__.assert_called_once()
        assert adapter._connection is None
        assert adapter._transaction is None

    @pytest.mark.asyncio
    async def test_cleanup_transaction_with_connection_error(self):
        """Test transaction cleanup when connection release fails."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_connection.__aexit__.side_effect = Exception("Connection release failed")

        adapter = PostgreSQLAdapter(mock_pool)
        adapter._connection = mock_connection
        adapter._transaction = Mock()

        # Should not raise, just log warning
        await adapter._cleanup_transaction()

        assert adapter._connection is None
        assert adapter._transaction is None

    @pytest.mark.asyncio
    async def test_begin_transaction_with_existing_connection(self):
        """Test beginning transaction with reused connection."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_transaction = AsyncMock(spec=AsyncTransaction)

        # Setup transaction context
        mock_connection.transaction.return_value.__aenter__.return_value = mock_transaction

        adapter = PostgreSQLAdapter(mock_pool)
        # Simulate existing connection from previous operation
        adapter._connection = mock_connection

        await adapter.begin_transaction()

        # Should reuse existing connection
        assert adapter._connection == mock_connection
        assert adapter._transaction == mock_transaction
        # Pool should not be called
        mock_pool.connection.assert_not_called()

    @pytest.mark.asyncio
    async def test_nested_transaction_prevention(self):
        """Test that nested transactions are prevented."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_transaction = AsyncMock(spec=AsyncTransaction)

        adapter = PostgreSQLAdapter(mock_pool)
        adapter._connection = mock_connection
        adapter._transaction = mock_transaction

        # Should not allow nested transaction
        with pytest.raises(TransactionError, match="Transaction is already active"):
            await adapter.begin_transaction()

        # Original transaction should remain active
        assert adapter._transaction == mock_transaction


class TestPostgreSQLAdapterConcurrency:
    """Test concurrent operations and thread safety."""

    @pytest.mark.asyncio
    async def test_concurrent_queries_without_transaction(self):
        """Test multiple concurrent queries without transaction."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)

        # Create multiple mock connections for concurrent operations
        connections = []
        cursors = []
        for i in range(3):
            mock_conn = AsyncMock(spec=AsyncConnection)
            mock_cursor = AsyncMock()
            mock_cursor.fetchone.return_value = {"id": i}

            mock_conn.cursor.return_value.__aenter__.return_value = mock_cursor
            mock_conn.cursor.return_value.__aexit__.return_value = None

            connections.append(mock_conn)
            cursors.append(mock_cursor)

        # Pool returns different connections
        mock_pool.connection.side_effect = [
            AsyncMock(__aenter__=AsyncMock(return_value=conn), __aexit__=AsyncMock())
            for conn in connections
        ]

        adapter = PostgreSQLAdapter(mock_pool)

        # Execute concurrent queries
        tasks = [adapter.fetch_one("SELECT * FROM users WHERE id = %s", i) for i in range(3)]

        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["id"] == i

        # Each query should use separate connection
        assert mock_pool.connection.call_count == 3

    @pytest.mark.asyncio
    async def test_concurrent_queries_with_transaction(self):
        """Test that queries in transaction use same connection."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_transaction = AsyncMock(spec=AsyncTransaction)
        mock_cursor = AsyncMock()

        # Setup pool to return connection
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection

        # Setup transaction
        mock_connection.transaction.return_value.__aenter__.return_value = mock_transaction

        # Setup cursor for queries
        mock_cursor.fetchone.return_value = {"id": 1}
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None

        adapter = PostgreSQLAdapter(mock_pool)

        # Begin transaction
        await adapter.begin_transaction()

        # Execute multiple queries in transaction
        tasks = [adapter.fetch_one("SELECT * FROM users WHERE id = %s", i) for i in range(3)]

        results = await asyncio.gather(*tasks)

        # All queries should use the same transaction connection
        assert adapter._connection == mock_connection
        assert all(r["id"] == 1 for r in results)

        # Pool should only be called once (for transaction)
        mock_pool.connection.assert_called_once()

        # Commit transaction
        await adapter.commit_transaction()


class TestPostgreSQLAdapterQueryParameters:
    """Test query parameter handling and SQL injection prevention."""

    @pytest.mark.asyncio
    async def test_query_with_no_parameters(self):
        """Test query execution without parameters."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_cursor = AsyncMock()
        mock_cursor.rowcount = 0

        # Setup contexts
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None

        adapter = PostgreSQLAdapter(mock_pool)

        result = await adapter.execute_query("TRUNCATE TABLE temp_data")

        assert result == "EXECUTE 0"
        mock_cursor.execute.assert_called_once_with("TRUNCATE TABLE temp_data", ())

    @pytest.mark.asyncio
    async def test_query_with_multiple_parameters(self):
        """Test query with multiple parameters."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_cursor = AsyncMock()
        mock_cursor.fetchall.return_value = [{"id": 1}, {"id": 2}]

        # Setup contexts
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None

        adapter = PostgreSQLAdapter(mock_pool)

        result = await adapter.fetch_all(
            "SELECT * FROM users WHERE age > %s AND status = %s AND city = %s",
            18,
            "active",
            "New York",
        )

        assert len(result) == 2
        mock_cursor.execute.assert_called_once_with(
            "SELECT * FROM users WHERE age > %s AND status = %s AND city = %s",
            (18, "active", "New York"),
        )

    @pytest.mark.asyncio
    async def test_query_with_none_parameters(self):
        """Test query with None as parameter value."""
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

        result = await adapter.fetch_one("SELECT * FROM users WHERE email = %s", None)

        assert result is None
        mock_cursor.execute.assert_called_once_with("SELECT * FROM users WHERE email = %s", (None,))

    @pytest.mark.asyncio
    async def test_batch_with_varied_parameter_types(self):
        """Test batch execution with various parameter types."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_cursor = AsyncMock()

        # Setup contexts
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None

        adapter = PostgreSQLAdapter(mock_pool)

        args_list = [
            (1, "string", 3.14, True, None),
            (2, "another", 2.71, False, "not null"),
            (3, "third", 1.41, True, None),
        ]

        await adapter.execute_batch("INSERT INTO test_table VALUES (%s, %s, %s, %s, %s)", args_list)

        mock_cursor.executemany.assert_called_once_with(
            "INSERT INTO test_table VALUES (%s, %s, %s, %s, %s)", args_list
        )


class TestPostgreSQLAdapterLogging:
    """Test logging behavior."""

    @pytest.mark.asyncio
    @patch("src.infrastructure.database.adapter.logger")
    async def test_query_logging_truncation(self, mock_logger):
        """Test that long queries are truncated in logs."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_cursor = AsyncMock()
        mock_cursor.rowcount = 1

        # Setup contexts
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None

        adapter = PostgreSQLAdapter(mock_pool)

        # Very long query
        long_query = "SELECT " + ", ".join([f"column_{i}" for i in range(100)]) + " FROM huge_table"

        await adapter.execute_query(long_query)

        # Check that debug log truncates query
        debug_calls = [call for call in mock_logger.debug.call_args_list]
        assert len(debug_calls) > 0
        logged_message = debug_calls[0][0][0]
        assert "..." in logged_message  # Query should be truncated
        assert len(logged_message) < len(long_query)

    @pytest.mark.asyncio
    @patch("src.infrastructure.database.adapter.logger")
    async def test_error_logging_with_query_context(self, mock_logger):
        """Test that errors include query context in logs."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_connection = AsyncMock(spec=AsyncConnection)
        mock_cursor = AsyncMock()
        mock_cursor.execute.side_effect = psycopg.IntegrityError("Unique constraint violated")

        # Setup contexts
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_connection.cursor.return_value.__aexit__.return_value = None

        adapter = PostgreSQLAdapter(mock_pool)

        query = "INSERT INTO users (email) VALUES (%s)"

        with pytest.raises(IntegrityError):
            await adapter.execute_query(query, "duplicate@example.com")

        # Check error logging includes query context
        error_calls = mock_logger.error.call_args_list
        assert len(error_calls) > 0
        logged_message = error_calls[0][0][0]
        assert "Integrity constraint violated" in logged_message
        assert query[:100] in logged_message or "INSERT INTO users" in logged_message
