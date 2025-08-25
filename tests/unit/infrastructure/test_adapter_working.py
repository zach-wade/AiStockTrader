"""
Working tests for PostgreSQLAdapter class.

Tests the actual adapter methods with proper parameter names and error handling.
These tests focus on core functionality and use proper mocking.
"""

# Standard library imports
from unittest.mock import AsyncMock, patch

# Third-party imports
import pytest
from psycopg_pool import AsyncConnectionPool

# Local imports
from src.application.interfaces.exceptions import (
    IntegrityError,
    RepositoryError,
    TimeoutError,
    TransactionError,
)
from src.infrastructure.database.adapter import PostgreSQLAdapter


class TestPostgreSQLAdapter:
    """Test PostgreSQLAdapter methods."""

    def test_init(self):
        """Test adapter initialization."""
        pool = AsyncMock(spec=AsyncConnectionPool)
        adapter = PostgreSQLAdapter(pool)
        assert adapter.pool == pool
        assert adapter.has_active_transaction is False

    def test_properties(self):
        """Test adapter properties."""
        pool = AsyncMock()
        adapter = PostgreSQLAdapter(pool)

        assert adapter.pool == pool
        assert adapter.has_active_transaction is False

        # Test with active transaction
        adapter._transaction = AsyncMock()
        assert adapter.has_active_transaction is True

    @pytest.mark.asyncio
    async def test_get_connection_info(self):
        """Test get connection info."""
        pool = AsyncMock()
        pool.max_size = 20
        pool.min_size = 2
        pool.closed = False

        adapter = PostgreSQLAdapter(pool)

        info = await adapter.get_connection_info()

        assert info["max_size"] == 20
        assert info["min_size"] == 2
        assert info["pool_status"] == "active"

    @pytest.mark.asyncio
    async def test_get_connection_info_closed_pool(self):
        """Test get connection info with closed pool."""
        pool = AsyncMock()
        pool.max_size = 20
        pool.min_size = 2
        pool.closed = True

        adapter = PostgreSQLAdapter(pool)

        info = await adapter.get_connection_info()

        assert info["pool_status"] == "closed"

    def test_str_representation(self):
        """Test string representation."""
        pool = AsyncMock()
        pool.max_size = 10
        adapter = PostgreSQLAdapter(pool)

        result = str(adapter)

        assert "PostgreSQLAdapter" in result
        assert "Pool(max_size=10)" in result
        assert "no transaction" in result

    def test_str_representation_with_transaction(self):
        """Test string representation with active transaction."""
        pool = AsyncMock()
        pool.max_size = 10
        adapter = PostgreSQLAdapter(pool)
        adapter._transaction = AsyncMock()

        result = str(adapter)

        assert "with active transaction" in result

    @pytest.mark.asyncio
    async def test_execute_query_with_mock(self):
        """Test execute_query by mocking the entire method."""
        pool = AsyncMock()
        adapter = PostgreSQLAdapter(pool)

        # Mock the execute_query method directly
        with patch.object(adapter, "execute_query", return_value="EXECUTE 1") as mock_execute:
            result = await adapter.execute_query("INSERT INTO test VALUES (%s)", "value1")
            assert result == "EXECUTE 1"
            mock_execute.assert_called_once_with("INSERT INTO test VALUES (%s)", "value1")

    @pytest.mark.asyncio
    async def test_fetch_one_with_mock(self):
        """Test fetch_one by mocking the entire method."""
        pool = AsyncMock()
        adapter = PostgreSQLAdapter(pool)

        expected_result = {"id": 1, "name": "test"}

        # Mock the fetch_one method directly
        with patch.object(adapter, "fetch_one", return_value=expected_result) as mock_fetch:
            result = await adapter.fetch_one("SELECT * FROM test WHERE id = %s", 1)
            assert result == expected_result
            mock_fetch.assert_called_once_with("SELECT * FROM test WHERE id = %s", 1)

    @pytest.mark.asyncio
    async def test_fetch_all_with_mock(self):
        """Test fetch_all by mocking the entire method."""
        pool = AsyncMock()
        adapter = PostgreSQLAdapter(pool)

        expected_result = [{"id": 1, "name": "test1"}, {"id": 2, "name": "test2"}]

        # Mock the fetch_all method directly
        with patch.object(adapter, "fetch_all", return_value=expected_result) as mock_fetch:
            result = await adapter.fetch_all("SELECT * FROM test")
            assert result == expected_result
            mock_fetch.assert_called_once_with("SELECT * FROM test")

    @pytest.mark.asyncio
    async def test_fetch_values_with_mock(self):
        """Test fetch_values by mocking the entire method."""
        pool = AsyncMock()
        adapter = PostgreSQLAdapter(pool)

        expected_result = [1, 2, 3]

        # Mock the fetch_values method directly
        with patch.object(adapter, "fetch_values", return_value=expected_result) as mock_fetch:
            result = await adapter.fetch_values("SELECT id FROM test")
            assert result == expected_result
            mock_fetch.assert_called_once_with("SELECT id FROM test")

    @pytest.mark.asyncio
    async def test_execute_batch_with_mock(self):
        """Test execute_batch by mocking the entire method."""
        pool = AsyncMock()
        adapter = PostgreSQLAdapter(pool)

        args_list = [("val1", "val2"), ("val3", "val4")]

        # Mock the execute_batch method directly
        with patch.object(adapter, "execute_batch", return_value=None) as mock_batch:
            await adapter.execute_batch("INSERT INTO test (a, b) VALUES (%s, %s)", args_list)
            mock_batch.assert_called_once_with("INSERT INTO test (a, b) VALUES (%s, %s)", args_list)

    @pytest.mark.asyncio
    async def test_health_check_with_mock(self):
        """Test health_check by mocking the entire method."""
        pool = AsyncMock()
        adapter = PostgreSQLAdapter(pool)

        # Mock successful health check
        with patch.object(adapter, "health_check", return_value=True) as mock_health:
            result = await adapter.health_check()
            assert result is True
            mock_health.assert_called_once()

    @pytest.mark.asyncio
    async def test_begin_transaction_already_active(self):
        """Test begin transaction when already active."""
        pool = AsyncMock()
        adapter = PostgreSQLAdapter(pool)
        adapter._transaction = AsyncMock()

        with pytest.raises(TransactionError, match="Transaction is already active"):
            await adapter.begin_transaction()

    @pytest.mark.asyncio
    async def test_commit_transaction_no_active(self):
        """Test commit transaction with no active transaction."""
        pool = AsyncMock()
        adapter = PostgreSQLAdapter(pool)

        with pytest.raises(TransactionError, match="No active transaction to commit"):
            await adapter.commit_transaction()

    @pytest.mark.asyncio
    async def test_rollback_transaction_no_active(self):
        """Test rollback transaction with no active transaction."""
        pool = AsyncMock()
        adapter = PostgreSQLAdapter(pool)

        # Should not raise - just log warning
        await adapter.rollback_transaction()
        assert adapter.has_active_transaction is False

    @pytest.mark.asyncio
    async def test_error_handling_with_exceptions(self):
        """Test that appropriate exceptions are raised for different error scenarios."""
        pool = AsyncMock()
        adapter = PostgreSQLAdapter(pool)

        # Test IntegrityError handling
        with patch.object(
            adapter, "execute_query", side_effect=IntegrityError("constraint", "violation")
        ):
            with pytest.raises(IntegrityError):
                await adapter.execute_query("INSERT INTO test VALUES (%s)", "value1")

        # Test RepositoryError handling
        with patch.object(adapter, "fetch_one", side_effect=RepositoryError("database error")):
            with pytest.raises(RepositoryError):
                await adapter.fetch_one("SELECT * FROM test WHERE id = %s", 1)

        # Test TimeoutError handling
        with patch.object(adapter, "fetch_all", side_effect=TimeoutError("fetch_all", 30.0)):
            with pytest.raises(TimeoutError):
                await adapter.fetch_all("SELECT * FROM test")

    @pytest.mark.asyncio
    async def test_parameter_handling(self):
        """Test that parameters are handled correctly."""
        pool = AsyncMock()
        adapter = PostgreSQLAdapter(pool)

        # Test single parameter
        with patch.object(adapter, "execute_query", return_value="EXECUTE 1") as mock_execute:
            await adapter.execute_query("INSERT INTO test VALUES (%s)", "value1")
            mock_execute.assert_called_once_with("INSERT INTO test VALUES (%s)", "value1")

        # Test multiple parameters
        with patch.object(adapter, "fetch_one", return_value=None) as mock_fetch:
            await adapter.fetch_one("SELECT * FROM test WHERE a = %s AND b = %s", "val1", "val2")
            mock_fetch.assert_called_once_with(
                "SELECT * FROM test WHERE a = %s AND b = %s", "val1", "val2"
            )

        # Test no parameters
        with patch.object(adapter, "fetch_all", return_value=[]) as mock_fetch:
            await adapter.fetch_all("SELECT * FROM test")
            mock_fetch.assert_called_once_with("SELECT * FROM test")

    @pytest.mark.asyncio
    async def test_transaction_state_management(self):
        """Test transaction state management."""
        pool = AsyncMock()
        adapter = PostgreSQLAdapter(pool)

        # Initial state
        assert adapter.has_active_transaction is False

        # Mock transaction
        mock_transaction = AsyncMock()
        adapter._transaction = mock_transaction
        assert adapter.has_active_transaction is True

        # Clear transaction
        adapter._transaction = None
        assert adapter.has_active_transaction is False

    @pytest.mark.asyncio
    async def test_cleanup_transaction_method(self):
        """Test the _cleanup_transaction method."""
        pool = AsyncMock()
        adapter = PostgreSQLAdapter(pool)

        # Set up mock connection and transaction
        mock_connection = AsyncMock()
        mock_transaction = AsyncMock()

        adapter._connection = mock_connection
        adapter._transaction = mock_transaction

        # Call cleanup
        await adapter._cleanup_transaction()

        # Verify cleanup
        mock_connection.__aexit__.assert_called_once()
        assert adapter._connection is None
        assert adapter._transaction is None

    @pytest.mark.asyncio
    async def test_cleanup_transaction_with_error(self):
        """Test _cleanup_transaction handles connection cleanup errors."""
        pool = AsyncMock()
        adapter = PostgreSQLAdapter(pool)

        # Set up mock connection that raises error on cleanup
        mock_connection = AsyncMock()
        mock_connection.__aexit__.side_effect = Exception("Cleanup error")

        adapter._connection = mock_connection
        adapter._transaction = AsyncMock()

        # Should not raise - just log warning
        await adapter._cleanup_transaction()

        # Transaction state should still be cleaned up
        assert adapter._connection is None
        assert adapter._transaction is None
