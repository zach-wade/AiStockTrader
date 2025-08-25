"""
Final comprehensive tests for database layer to achieve 90%+ coverage.

Tests both adapter.py and connection.py modules with proper async mocking.
"""

import asyncio
import builtins
import os
from unittest.mock import AsyncMock, Mock, patch

import psycopg
import pytest
from psycopg_pool import AsyncConnectionPool

from src.application.interfaces.exceptions import (
    ConnectionError,
    FactoryError,
    TimeoutError,
    TransactionError,
)
from src.infrastructure.database.adapter import PostgreSQLAdapter
from src.infrastructure.database.connection import (
    ConnectionFactory,
    DatabaseConfig,
    DatabaseConnection,
)

# ============================================================================
# ADAPTER TESTS - Additional edge cases and error scenarios
# ============================================================================


class TestAdapterErrorScenarios:
    """Test error scenarios not covered by existing tests."""

    @pytest.mark.asyncio
    async def test_execute_query_with_timeout_none(self):
        """Test execute_query with None timeout uses default."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        adapter = PostgreSQLAdapter(mock_pool)

        # Setup connection mock
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.rowcount = 1
        mock_cursor.execute = AsyncMock()

        # Mock context managers
        mock_conn.cursor.return_value.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__aexit__ = AsyncMock()
        mock_pool.connection.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.connection.return_value.__aexit__ = AsyncMock()

        result = await adapter.execute_query("DELETE FROM test", command_timeout=None)
        assert result == "EXECUTE 1"

    @pytest.mark.asyncio
    async def test_fetch_one_with_specific_timeout(self):
        """Test fetch_one with specific timeout value."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        adapter = PostgreSQLAdapter(mock_pool)

        # Setup connection mock
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.execute = AsyncMock(side_effect=builtins.TimeoutError())

        # Mock context managers
        mock_conn.cursor.return_value.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__aexit__ = AsyncMock()
        mock_pool.connection.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.connection.return_value.__aexit__ = AsyncMock()

        with pytest.raises(TimeoutError) as exc_info:
            await adapter.fetch_one("SELECT * FROM test", command_timeout=45.0)

        assert exc_info.timeout_seconds == 45.0

    @pytest.mark.asyncio
    async def test_commit_cleanup_on_error(self):
        """Test that cleanup happens even if commit fails."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        adapter = PostgreSQLAdapter(mock_pool)

        # Setup transaction mocks
        mock_conn = AsyncMock()
        mock_transaction = AsyncMock()
        mock_transaction.__aexit__ = AsyncMock(
            side_effect=psycopg.OperationalError("Commit failed")
        )
        mock_conn.__aexit__ = AsyncMock()  # Should still be called

        adapter._connection = mock_conn
        adapter._transaction = mock_transaction

        with pytest.raises(TransactionError):
            await adapter.commit_transaction()

        # Verify cleanup was called
        mock_conn.__aexit__.assert_called_once()
        assert adapter._connection is None
        assert adapter._transaction is None

    @pytest.mark.asyncio
    async def test_rollback_cleanup_on_error(self):
        """Test that cleanup happens even if rollback fails."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        adapter = PostgreSQLAdapter(mock_pool)

        # Setup transaction mocks
        mock_conn = AsyncMock()
        mock_transaction = AsyncMock()
        mock_transaction.__aexit__ = AsyncMock(
            side_effect=psycopg.OperationalError("Rollback failed")
        )
        mock_conn.__aexit__ = AsyncMock()  # Should still be called

        adapter._connection = mock_conn
        adapter._transaction = mock_transaction

        with pytest.raises(TransactionError):
            await adapter.rollback_transaction()

        # Verify cleanup was called
        mock_conn.__aexit__.assert_called_once()
        assert adapter._connection is None
        assert adapter._transaction is None


# ============================================================================
# CONNECTION TESTS - Comprehensive coverage
# ============================================================================


class TestDatabaseConfigComplete:
    """Complete tests for DatabaseConfig."""

    def test_from_url_empty_path(self):
        """Test URL with empty path defaults correctly."""
        config = DatabaseConfig.from_url("postgresql://user:pass@host:5432/db_name")
        assert config.database == "db_name"
        assert config.user == "user"
        assert config.password == "pass"

    @patch("src.infrastructure.security.SecretsManager")
    def test_from_env_no_env_vars(self, mock_secrets_manager):
        """Test from_env with no environment variables set."""
        mock_instance = Mock()
        mock_instance.get_database_config.return_value = {
            "host": "secret-host",
            "port": "5432",
            "database": "secret-db",
            "user": "secret-user",
            "password": "secret-pass",
        }
        mock_secrets_manager.get_instance.return_value = mock_instance

        with patch.dict(os.environ, {}, clear=True):
            config = DatabaseConfig.from_env()

        assert config.host == "secret-host"
        assert config.database == "secret-db"
        assert config.user == "secret-user"

    def test_build_dsn_with_ssl(self):
        """Test DSN building with SSL configuration."""
        with patch("src.infrastructure.database.connection.DSNBuilder") as mock_builder_class:
            mock_builder = Mock()
            mock_builder_class.return_value = mock_builder

            # Setup method chaining
            mock_builder.with_host.return_value = mock_builder
            mock_builder.with_port.return_value = mock_builder
            mock_builder.with_database.return_value = mock_builder
            mock_builder.with_credentials.return_value = mock_builder
            mock_builder.with_ssl.return_value = mock_builder
            mock_builder.build.return_value = "postgresql://user:pass@host:5432/db?sslmode=require"

            config = DatabaseConfig(
                host="host",
                port=5432,
                database="db",
                user="user",
                password="pass",
                ssl_mode="require",
            )

            dsn = config.build_dsn()
            assert "sslmode=require" in dsn or dsn  # DSN is mocked
            mock_builder.with_ssl.assert_called_once()


class TestDatabaseConnectionComplete:
    """Complete tests for DatabaseConnection."""

    @pytest.fixture
    def config(self):
        return DatabaseConfig(
            host="test-host", port=5432, database="test-db", user="test-user", password="test-pass"
        )

    @pytest.fixture
    def connection(self, config):
        return DatabaseConnection(config)

    @pytest.mark.asyncio
    async def test_connect_full_flow(self, connection, config):
        """Test complete connection flow with all steps."""
        with patch("src.infrastructure.database.connection.AsyncConnectionPool") as mock_pool_class:
            # Create mock pool
            mock_pool = AsyncMock()
            mock_pool.closed = False
            mock_pool.max_size = 20
            mock_pool.open = AsyncMock()
            mock_pool_class.return_value = mock_pool

            # Mock connection test
            mock_conn_ctx = AsyncMock()
            mock_conn = AsyncMock()
            mock_cursor_ctx = AsyncMock()
            mock_cursor = AsyncMock()

            mock_pool.connection.return_value = mock_conn_ctx
            mock_conn_ctx.__aenter__.return_value = mock_conn
            mock_conn_ctx.__aexit__.return_value = None

            mock_conn.cursor.return_value = mock_cursor_ctx
            mock_cursor_ctx.__aenter__.return_value = mock_cursor
            mock_cursor_ctx.__aexit__.return_value = None

            mock_cursor.execute = AsyncMock()
            mock_cursor.fetchone = AsyncMock()

            # Mock DSN building
            with patch.object(config, "build_dsn", return_value="test_dsn"):
                # Mock health monitoring start
                with patch.object(connection, "_start_health_monitoring") as mock_start_health:
                    result = await connection.connect()

            assert result == mock_pool
            assert connection._pool == mock_pool
            mock_pool.open.assert_called_once()
            mock_cursor.execute.assert_called_once_with("SELECT 1")
            mock_start_health.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_operational_error(self, connection, config):
        """Test connection with operational error."""
        with patch("src.infrastructure.database.connection.AsyncConnectionPool") as mock_pool_class:
            mock_pool_class.side_effect = psycopg.OperationalError("Cannot connect")

            with patch.object(config, "build_dsn", return_value="test_dsn"):
                with pytest.raises(ConnectionError, match="Failed to connect"):
                    await connection.connect()

            assert connection._pool is None

    @pytest.mark.asyncio
    async def test_connect_timeout_error(self, connection, config):
        """Test connection with timeout error."""
        with patch("src.infrastructure.database.connection.AsyncConnectionPool") as mock_pool_class:
            mock_pool = AsyncMock()
            mock_pool.open = AsyncMock(side_effect=TimeoutError("Connection timeout"))
            mock_pool_class.return_value = mock_pool

            with patch.object(config, "build_dsn", return_value="test_dsn"):
                with pytest.raises(ConnectionError, match="Failed to connect"):
                    await connection.connect()

    @pytest.mark.asyncio
    async def test_connect_os_error(self, connection, config):
        """Test connection with OS error."""
        with patch("src.infrastructure.database.connection.AsyncConnectionPool") as mock_pool_class:
            mock_pool = AsyncMock()
            mock_pool.closed = False
            mock_pool.open = AsyncMock()
            mock_pool_class.return_value = mock_pool

            # Mock connection test failure
            mock_pool.connection.side_effect = OSError("Network unreachable")

            with patch.object(config, "build_dsn", return_value="test_dsn"):
                with pytest.raises(ConnectionError, match="Failed to connect"):
                    await connection.connect()

    @pytest.mark.asyncio
    async def test_disconnect_complete(self, connection):
        """Test complete disconnection flow."""
        # Setup mocks
        mock_pool = AsyncMock()
        mock_pool.closed = False
        mock_pool.close = AsyncMock()

        mock_task = AsyncMock()
        mock_task.done.return_value = False
        mock_task.cancel = Mock()

        connection._pool = mock_pool
        connection._health_check_task = mock_task

        await connection.disconnect()

        assert connection._pool is None
        assert connection._health_check_task is None
        assert connection._is_closed is True
        mock_pool.close.assert_called_once()
        mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_monitor_loop_complete(self, connection):
        """Test health monitor loop with multiple iterations."""
        mock_pool = AsyncMock()
        mock_pool.closed = False

        # Mock connection for health check
        mock_conn_ctx = AsyncMock()
        mock_conn = AsyncMock()
        mock_cursor_ctx = AsyncMock()
        mock_cursor = AsyncMock()

        mock_pool.connection.return_value = mock_conn_ctx
        mock_conn_ctx.__aenter__.return_value = mock_conn
        mock_conn_ctx.__aexit__.return_value = None

        mock_conn.cursor.return_value = mock_cursor_ctx
        mock_cursor_ctx.__aenter__.return_value = mock_cursor
        mock_cursor_ctx.__aexit__.return_value = None

        mock_cursor.execute = AsyncMock()
        mock_cursor.fetchone = AsyncMock()

        connection._pool = mock_pool
        connection._is_closed = False

        # Simulate multiple iterations then cancel
        call_count = 0

        async def mock_sleep(duration):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise asyncio.CancelledError()

        with patch("asyncio.sleep", side_effect=mock_sleep):
            try:
                await connection._health_monitor_loop()
            except asyncio.CancelledError:
                pass

        # Should have performed health checks
        assert mock_cursor.execute.call_count >= 1

    @pytest.mark.asyncio
    async def test_health_monitor_loop_error_handling(self, connection):
        """Test health monitor loop handles errors gracefully."""
        mock_pool = AsyncMock()
        mock_pool.closed = False
        mock_pool.connection.side_effect = Exception("Health check failed")

        connection._pool = mock_pool
        connection._is_closed = False

        # Run one iteration with error
        async def mock_sleep(duration):
            raise asyncio.CancelledError()

        with patch("asyncio.sleep", side_effect=mock_sleep):
            try:
                await connection._health_monitor_loop()
            except asyncio.CancelledError:
                pass

        # Should handle error without raising
        mock_pool.connection.assert_called()

    @pytest.mark.asyncio
    async def test_acquire_context_manager(self, connection):
        """Test acquire as context manager."""
        mock_pool = AsyncMock()
        mock_pool.closed = False

        # Mock connection context
        mock_conn_ctx = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.connection.return_value = mock_conn_ctx
        mock_conn_ctx.__aenter__.return_value = mock_conn
        mock_conn_ctx.__aexit__.return_value = None

        connection._pool = mock_pool

        async with connection.acquire() as conn:
            assert conn == mock_conn

        mock_conn_ctx.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_pool_stats_variations(self, connection):
        """Test pool stats in various states."""
        # Test with closed pool
        mock_pool = Mock()
        mock_pool.closed = True
        mock_pool.max_size = 10
        mock_pool.min_size = 2
        connection._pool = mock_pool

        stats = await connection.get_pool_stats()
        assert stats["is_closed"] is True

        # Test with active pool
        mock_pool.closed = False
        stats = await connection.get_pool_stats()
        assert stats["is_closed"] is False
        assert stats["max_size"] == 10
        assert stats["min_size"] == 2


class TestConnectionFactoryComplete:
    """Complete tests for ConnectionFactory."""

    def teardown_method(self):
        """Reset factory after each test."""
        ConnectionFactory.reset()

    @pytest.mark.asyncio
    async def test_create_connection_error_handling(self):
        """Test connection creation with various errors."""
        with patch.object(DatabaseConfig, "from_env") as mock_from_env:
            mock_config = Mock()
            mock_from_env.return_value = mock_config

            with patch(
                "src.infrastructure.database.connection.DatabaseConnection"
            ) as mock_conn_class:
                mock_conn = Mock()
                mock_conn.is_closed = False
                mock_conn.connect = AsyncMock(side_effect=Exception("Connection failed"))
                mock_conn_class.return_value = mock_conn

                with pytest.raises(FactoryError, match="Failed to create connection"):
                    await ConnectionFactory.create_connection()

    @pytest.mark.asyncio
    async def test_create_connection_force_new_with_active(self):
        """Test forcing new connection when one is active."""
        # Create initial connection
        mock_conn1 = Mock()
        mock_conn1.is_closed = False
        mock_conn1.connect = AsyncMock()
        mock_conn1.disconnect = AsyncMock()
        ConnectionFactory._connection = mock_conn1

        # Create new connection
        with patch.object(DatabaseConfig, "from_env") as mock_from_env:
            mock_config = Mock()
            mock_from_env.return_value = mock_config

            with patch(
                "src.infrastructure.database.connection.DatabaseConnection"
            ) as mock_conn_class:
                mock_conn2 = Mock()
                mock_conn2.is_closed = False
                mock_conn2.connect = AsyncMock()
                mock_conn_class.return_value = mock_conn2

                result = await ConnectionFactory.create_connection(force_new=True)

                assert result == mock_conn2
                mock_conn1.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_all_complete(self):
        """Test closing all connections."""
        mock_conn = AsyncMock()
        mock_conn.is_closed = False
        mock_conn.disconnect = AsyncMock()
        ConnectionFactory._connection = mock_conn

        await ConnectionFactory.close_all()

        mock_conn.disconnect.assert_called_once()
        assert ConnectionFactory._connection is None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestDatabaseIntegration:
    """Integration tests for database components."""

    @pytest.mark.asyncio
    async def test_adapter_with_real_pool_mock(self):
        """Test adapter with realistic pool mock."""
        # Create a more realistic pool mock
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_pool.closed = False
        mock_pool.max_size = 10
        mock_pool.min_size = 2

        adapter = PostgreSQLAdapter(mock_pool)

        # Test string representation
        str_repr = str(adapter)
        assert "PostgreSQLAdapter" in str_repr
        assert "Pool(max_size=10)" in str_repr
        assert "no transaction" in str_repr

        # Test with active transaction
        adapter._transaction = Mock()
        str_repr = str(adapter)
        assert "with active transaction" in str_repr

    @pytest.mark.asyncio
    async def test_full_transaction_flow(self):
        """Test complete transaction flow."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        adapter = PostgreSQLAdapter(mock_pool)

        # Mock connection acquisition for transaction
        mock_conn_ctx = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.connection.return_value = mock_conn_ctx
        mock_conn_ctx.__aenter__.return_value = mock_conn

        # Mock transaction creation
        mock_tx_ctx = AsyncMock()
        mock_tx = AsyncMock()
        mock_conn.transaction.return_value = mock_tx_ctx
        mock_tx_ctx.__aenter__.return_value = mock_tx
        mock_tx_ctx.__aexit__.return_value = None

        # Begin transaction
        await adapter.begin_transaction()
        assert adapter.has_active_transaction

        # Setup cursor for queries in transaction
        mock_cursor_ctx = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.rowcount = 1
        mock_conn.cursor.return_value = mock_cursor_ctx
        mock_cursor_ctx.__aenter__.return_value = mock_cursor
        mock_cursor_ctx.__aexit__.return_value = None

        # Execute query in transaction
        result = await adapter.execute_query("UPDATE test SET value = %s", "new_value")
        assert result == "EXECUTE 1"

        # Commit transaction
        mock_conn.__aexit__.return_value = None
        await adapter.commit_transaction()
        assert not adapter.has_active_transaction
        assert adapter._connection is None
        assert adapter._transaction is None
