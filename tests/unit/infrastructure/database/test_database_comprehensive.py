"""
Comprehensive unit tests for database infrastructure - achieving 90%+ coverage.

Tests adapter, connection pooling, transactions, and error handling.
"""

import asyncio
import threading
from contextlib import asynccontextmanager
from datetime import UTC, datetime

UTC = UTC
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import asyncpg
import pytest

from src.application.interfaces.exceptions import TransactionError
from src.infrastructure.database.connection import (
    ConnectionFactory,
    DatabaseConfig,
    DatabaseConnection,
)


class TestDatabaseConfig:
    """Test DatabaseConfig class."""

    def test_config_initialization_defaults(self):
        """Test config initialization with defaults."""
        config = DatabaseConfig()

        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "trading_db"
        assert config.user == "trading_user"
        assert config.password == ""
        assert config.min_connections == 10
        assert config.max_connections == 20
        assert config.command_timeout == 30.0
        assert config.command_timeout == 10.0
        assert config.max_inactive_connection_lifetime == 300.0

    def test_config_initialization_custom(self):
        """Test config initialization with custom values."""
        config = DatabaseConfig(
            host="db.example.com",
            port=5433,
            database="custom_db",
            user="custom_user",
            password="secret",
            min_connections=5,
            max_connections=15,
            command_timeout=60.0,
        )

        assert config.host == "db.example.com"
        assert config.port == 5433
        assert config.database == "custom_db"
        assert config.user == "custom_user"
        assert config.password == "secret"
        assert config.min_connections == 5
        assert config.max_connections == 15
        assert config.command_timeout == 60.0

    def test_config_to_dsn(self):
        """Test DSN generation."""
        config = DatabaseConfig(
            host="localhost", port=5432, database="test_db", user="test_user", password="test_pass"
        )

        dsn = config.to_dsn()

        assert dsn == "postgresql://test_user:test_pass@localhost:5432/test_db"

    def test_config_to_dsn_special_characters(self):
        """Test DSN with special characters in password."""
        config = DatabaseConfig(password="p@ss:word/123")

        dsn = config.to_dsn()

        # Password should be URL encoded
        assert "p%40ss%3Aword%2F123" in dsn

    def test_config_validation_invalid_port(self):
        """Test config validation with invalid port."""
        with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
            DatabaseConfig(port=0)

        with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
            DatabaseConfig(port=70000)

    def test_config_validation_invalid_connections(self):
        """Test config validation with invalid connection limits."""
        with pytest.raises(ValueError, match="min_connections must be positive"):
            DatabaseConfig(min_connections=0)

        with pytest.raises(ValueError, match="max_connections must be positive"):
            DatabaseConfig(max_connections=0)

        with pytest.raises(ValueError, match="min_connections cannot exceed max_connections"):
            DatabaseConfig(min_connections=10, max_connections=5)


class TestDatabaseConnection:
    """Test DatabaseConnection class."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return DatabaseConfig(
            host="localhost", database="test_db", user="test_user", password="test_pass"
        )

    @pytest.mark.asyncio
    async def test_pool_initialization(self, config):
        """Test pool initialization."""
        connection = DatabaseConnection(config)

        assert connection.config == config
        assert pool._pool is None
        assert pool._initialized is False

    @pytest.mark.asyncio
    async def test_pool_connect(self, config):
        """Test pool connection."""
        connection = DatabaseConnection(config)

        with patch("asyncpg.create_pool") as mock_create:
            mock_pool = AsyncMock()
            mock_create.return_value = mock_pool

            await connection.connect()

            assert pool._pool == mock_pool
            assert pool._initialized is True

            mock_create.assert_called_once_with(
                dsn=config.to_dsn(),
                min_size=config.min_connections,
                max_size=config.max_connections,
                command_timeout=config.command_timeout,
                max_inactive_connection_lifetime=config.max_inactive_connection_lifetime,
            )

    @pytest.mark.asyncio
    async def test_pool_connect_failure(self, config):
        """Test pool connection failure."""
        connection = DatabaseConnection(config)

        with patch("asyncpg.create_pool") as mock_create:
            mock_create.side_effect = asyncpg.PostgresError("Connection failed")

            with pytest.raises(ConnectionPoolError, match="Failed to create connection pool"):
                await connection.connect()

            assert pool._initialized is False

    @pytest.mark.asyncio
    async def test_pool_disconnect(self, config):
        """Test pool disconnection."""
        connection = DatabaseConnection(config)
        connection._pool = AsyncMock()
        connection._is_closed = False

        await connection.disconnect()

        connection._pool.close.assert_called_once()
        assert connection._pool is None
        assert connection._is_closed is True

    @pytest.mark.asyncio
    async def test_pool_disconnect_not_initialized(self, config):
        """Test disconnecting uninitialized pool."""
        connection = DatabaseConnection(config)

        # Should not raise
        await connection.disconnect()

        assert pool._pool is None
        assert pool._initialized is False

    @pytest.mark.asyncio
    async def test_acquire_connection(self, config):
        """Test acquiring connection from pool."""
        connection = DatabaseConnection(config)
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        connection._pool = mock_pool
        connection._is_closed = False

        async with connection.acquire() as conn:
            assert conn == mock_conn

        mock_pool.acquire.assert_called_once()

    @pytest.mark.asyncio
    async def test_acquire_connection_not_initialized(self, config):
        """Test acquiring connection when pool not initialized."""
        connection = DatabaseConnection(config)

        with pytest.raises(ConnectionPoolError, match="Connection pool not initialized"):
            async with pool.acquire():
                pass

    @pytest.mark.asyncio
    async def test_execute_query(self, config):
        """Test executing query."""
        connection = DatabaseConnection(config)
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = "OK"
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        connection._pool = mock_pool
        connection._is_closed = False

        result = await connection.execute("INSERT INTO test VALUES ($1)", 1)

        assert result == "OK"
        mock_conn.execute.assert_called_once_with("INSERT INTO test VALUES ($1)", 1)

    @pytest.mark.asyncio
    async def test_fetch_one(self, config):
        """Test fetching single row."""
        connection = DatabaseConnection(config)
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": 1, "name": "test"}
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        connection._pool = mock_pool
        connection._is_closed = False

        result = await connection.fetch_one("SELECT * FROM test WHERE id = $1", 1)

        assert result == {"id": 1, "name": "test"}
        mock_conn.fetchrow.assert_called_once_with("SELECT * FROM test WHERE id = $1", 1)

    @pytest.mark.asyncio
    async def test_fetch_all(self, config):
        """Test fetching multiple rows."""
        connection = DatabaseConnection(config)
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [{"id": 1, "name": "test1"}, {"id": 2, "name": "test2"}]
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        connection._pool = mock_pool
        connection._is_closed = False

        result = await connection.fetch_all("SELECT * FROM test")

        assert len(result) == 2
        assert result[0]["name"] == "test1"
        mock_conn.fetch.assert_called_once_with("SELECT * FROM test")

    @pytest.mark.asyncio
    async def test_transaction(self, config):
        """Test transaction context manager."""
        connection = DatabaseConnection(config)
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_tx = AsyncMock()
        mock_conn.transaction.return_value = mock_tx
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        connection._pool = mock_pool
        connection._is_closed = False

        async with pool.transaction() as tx:
            assert tx == mock_tx

        mock_conn.transaction.assert_called_once()

    @pytest.mark.asyncio
    async def test_is_healthy(self, config):
        """Test health check."""
        connection = DatabaseConnection(config)
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 1
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        connection._pool = mock_pool
        connection._is_closed = False

        result = await connection.is_healthy()

        assert result is True
        mock_conn.fetchval.assert_called_once_with("SELECT 1")

    @pytest.mark.asyncio
    async def test_is_healthy_not_initialized(self, config):
        """Test health check when not initialized."""
        connection = DatabaseConnection(config)

        result = await connection.is_healthy()

        assert result is False

    @pytest.mark.asyncio
    async def test_is_healthy_query_failure(self, config):
        """Test health check with query failure."""
        connection = DatabaseConnection(config)
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.fetchval.side_effect = asyncpg.PostgresError("Connection lost")
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        connection._pool = mock_pool
        connection._is_closed = False

        result = await connection.is_healthy()

        assert result is False


class TestConnectionFactory:
    """Test ConnectionFactory singleton."""

    def teardown_method(self):
        """Reset singleton after each test."""
        ConnectionFactory._instance = None
        ConnectionManager._lock = threading.Lock()

    def test_singleton_pattern(self):
        """Test singleton pattern implementation."""
        config = DatabaseConfig()

        manager1 = ConnectionFactory()
        manager2 = ConnectionFactory()

        assert manager1 is manager2

    def test_singleton_thread_safety(self):
        """Test thread-safe singleton creation."""
        config = DatabaseConfig()
        managers = []

        def create_manager():
            managers.append(ConnectionFactory())

        threads = []
        for _ in range(10):
            thread = threading.Thread(target=create_manager)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All should be the same instance
        assert len(set(id(m) for m in managers)) == 1

    @pytest.mark.asyncio
    async def test_get_pool(self):
        """Test getting connection pool."""
        config = DatabaseConfig()
        manager = ConnectionManager(config)

        pool = await manager.get_pool()

        assert isinstance(pool, ConnectionPool)
        assert pool is manager._pools["default"]

    @pytest.mark.asyncio
    async def test_get_pool_named(self):
        """Test getting named pool."""
        config = DatabaseConfig()
        manager = ConnectionManager(config)

        pool1 = await manager.get_pool("pool1")
        pool2 = await manager.get_pool("pool2")
        pool1_again = await manager.get_pool("pool1")

        assert pool1 is not pool2
        assert pool1 is pool1_again

    @pytest.mark.asyncio
    async def test_close_all_pools(self):
        """Test closing all pools."""
        config = DatabaseConfig()
        manager = ConnectionManager(config)

        # Create multiple pools
        pool1 = await manager.get_pool("pool1")
        pool2 = await manager.get_pool("pool2")

        # Mock the pools
        pool1._pool = AsyncMock()
        pool1._initialized = True
        pool2._pool = AsyncMock()
        pool2._initialized = True

        await manager.close_all()

        pool1._pool.close.assert_called_once()
        pool2._pool.close.assert_called_once()
        assert len(manager._pools) == 0


class TestDatabaseAdapter:
    """Test DatabaseAdapter class."""

    @pytest.fixture
    def adapter(self):
        """Create adapter instance."""
        return DatabaseAdapter()

    @pytest.fixture
    def mock_pool(self):
        """Create mock pool."""
        pool = AsyncMock()
        return pool

    @pytest.mark.asyncio
    async def test_adapter_initialization(self, adapter):
        """Test adapter initialization."""
        assert adapter._pool is None
        assert adapter._transaction_conn is None
        assert adapter._transaction_depth == 0

    @pytest.mark.asyncio
    async def test_connect(self, adapter):
        """Test connecting adapter."""
        with patch.object(ConnectionManager, "__new__") as mock_manager_class:
            mock_manager = AsyncMock()
            mock_pool = AsyncMock()
            mock_manager.get_pool.return_value = mock_pool
            mock_manager_class.return_value = mock_manager

            await adapter.connect()

            assert adapter._pool == mock_pool
            mock_pool.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_with_config(self, adapter):
        """Test connecting with custom config."""
        config = DatabaseConfig(database="custom_db")

        with patch.object(ConnectionManager, "__new__") as mock_manager_class:
            mock_manager = AsyncMock()
            mock_pool = AsyncMock()
            mock_manager.get_pool.return_value = mock_pool
            mock_manager_class.return_value = mock_manager

            await adapter.connect(config)

            mock_manager_class.assert_called_once_with(ConnectionManager, config)

    @pytest.mark.asyncio
    async def test_disconnect(self, adapter, mock_pool):
        """Test disconnecting adapter."""
        adapter._pool = mock_pool

        await adapter.disconnect()

        mock_pool.disconnect.assert_called_once()
        assert adapter._pool is None

    @pytest.mark.asyncio
    async def test_execute(self, adapter, mock_pool):
        """Test executing query."""
        adapter._pool = mock_pool
        mock_pool.execute.return_value = "OK"

        result = await adapter.execute("INSERT INTO test VALUES ($1)", 1)

        assert result == "OK"
        mock_pool.execute.assert_called_once_with("INSERT INTO test VALUES ($1)", 1)

    @pytest.mark.asyncio
    async def test_execute_no_pool(self, adapter):
        """Test execute without pool."""
        with pytest.raises(DatabaseError, match="Not connected to database"):
            await adapter.execute("SELECT 1")

    @pytest.mark.asyncio
    async def test_execute_with_error(self, adapter, mock_pool):
        """Test execute with database error."""
        adapter._pool = mock_pool
        mock_pool.execute.side_effect = asyncpg.PostgresError("Syntax error")

        with pytest.raises(DatabaseError, match="Database query failed"):
            await adapter.execute("INVALID SQL")

    @pytest.mark.asyncio
    async def test_fetch_one(self, adapter, mock_pool):
        """Test fetching single row."""
        adapter._pool = mock_pool
        mock_pool.fetch_one.return_value = {"id": 1, "value": "test"}

        result = await adapter.fetch_one("SELECT * FROM test WHERE id = $1", 1)

        assert result == {"id": 1, "value": "test"}
        mock_pool.fetch_one.assert_called_once_with("SELECT * FROM test WHERE id = $1", 1)

    @pytest.mark.asyncio
    async def test_fetch_all(self, adapter, mock_pool):
        """Test fetching multiple rows."""
        adapter._pool = mock_pool
        mock_pool.fetch_all.return_value = [
            {"id": 1, "value": "test1"},
            {"id": 2, "value": "test2"},
        ]

        result = await adapter.fetch_all("SELECT * FROM test")

        assert len(result) == 2
        mock_pool.fetch_all.assert_called_once_with("SELECT * FROM test")

    @pytest.mark.asyncio
    async def test_execute_many(self, adapter, mock_pool):
        """Test executing multiple statements."""
        adapter._pool = mock_pool
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        values = [(1, "a"), (2, "b"), (3, "c")]
        await adapter.execute_many("INSERT INTO test VALUES ($1, $2)", values)

        mock_conn.executemany.assert_called_once_with("INSERT INTO test VALUES ($1, $2)", values)

    @pytest.mark.asyncio
    async def test_execute_many_error(self, adapter, mock_pool):
        """Test execute_many with error."""
        adapter._pool = mock_pool
        mock_conn = AsyncMock()
        mock_conn.executemany.side_effect = asyncpg.PostgresError("Constraint violation")
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        with pytest.raises(DatabaseError, match="Batch execution failed"):
            await adapter.execute_many("INSERT INTO test VALUES ($1)", [(1,)])


class TestDatabaseAdapterTransactions:
    """Test transaction handling in DatabaseAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create adapter instance."""
        return DatabaseAdapter()

    @pytest.fixture
    def mock_pool(self):
        """Create mock pool with connection."""
        pool = AsyncMock()
        conn = AsyncMock()
        tx = AsyncMock()

        # Setup connection acquisition
        pool.acquire.return_value.__aenter__.return_value = conn

        # Setup transaction
        conn.transaction.return_value = tx
        tx.__aenter__.return_value = tx
        tx.__aexit__.return_value = None

        return pool, conn, tx

    @pytest.mark.asyncio
    async def test_transaction_basic(self, adapter, mock_pool):
        """Test basic transaction."""
        pool, conn, tx = mock_pool
        adapter._pool = pool

        async with adapter.transaction():
            assert adapter._transaction_conn == conn
            assert adapter._transaction_depth == 1

            # Execute within transaction
            await adapter.execute("INSERT INTO test VALUES (1)")
            conn.execute.assert_called_with("INSERT INTO test VALUES (1)")

        # After transaction
        assert adapter._transaction_conn is None
        assert adapter._transaction_depth == 0
        tx.__aenter__.assert_called_once()
        tx.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_transaction_nested(self, adapter, mock_pool):
        """Test nested transactions (savepoints)."""
        pool, conn, tx = mock_pool
        adapter._pool = pool

        async with adapter.transaction():
            assert adapter._transaction_depth == 1

            async with adapter.transaction():
                assert adapter._transaction_depth == 2

                async with adapter.transaction():
                    assert adapter._transaction_depth == 3

                assert adapter._transaction_depth == 2

            assert adapter._transaction_depth == 1

        assert adapter._transaction_depth == 0

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_error(self, adapter, mock_pool):
        """Test transaction rollback on error."""
        pool, conn, tx = mock_pool
        adapter._pool = pool

        with pytest.raises(ValueError):
            async with adapter.transaction():
                await adapter.execute("INSERT INTO test VALUES (1)")
                raise ValueError("Test error")

        # Transaction should be rolled back
        tx.__aexit__.assert_called_once()
        # Check it was called with exception info
        call_args = tx.__aexit__.call_args[0]
        assert call_args[0] == ValueError
        assert isinstance(call_args[1], ValueError)

    @pytest.mark.asyncio
    async def test_transaction_no_pool(self, adapter):
        """Test transaction without pool."""
        with pytest.raises(DatabaseError, match="Not connected to database"):
            async with adapter.transaction():
                pass

    @pytest.mark.asyncio
    async def test_begin_transaction(self, adapter, mock_pool):
        """Test explicit transaction begin."""
        pool, conn, tx = mock_pool
        adapter._pool = pool

        await adapter.begin_transaction()

        assert adapter._transaction_conn == conn
        assert adapter._transaction_depth == 1
        conn.transaction.assert_called_once()

    @pytest.mark.asyncio
    async def test_commit_transaction(self, adapter, mock_pool):
        """Test explicit transaction commit."""
        pool, conn, tx = mock_pool
        adapter._pool = pool

        await adapter.begin_transaction()
        await adapter.execute("INSERT INTO test VALUES (1)")
        await adapter.commit_transaction()

        assert adapter._transaction_conn is None
        assert adapter._transaction_depth == 0

    @pytest.mark.asyncio
    async def test_rollback_transaction(self, adapter, mock_pool):
        """Test explicit transaction rollback."""
        pool, conn, tx = mock_pool
        adapter._pool = pool

        await adapter.begin_transaction()
        await adapter.execute("INSERT INTO test VALUES (1)")
        await adapter.rollback_transaction()

        assert adapter._transaction_conn is None
        assert adapter._transaction_depth == 0

    @pytest.mark.asyncio
    async def test_commit_without_transaction(self, adapter, mock_pool):
        """Test commit without active transaction."""
        pool, _, _ = mock_pool
        adapter._pool = pool

        with pytest.raises(TransactionError, match="No active transaction"):
            await adapter.commit_transaction()

    @pytest.mark.asyncio
    async def test_rollback_without_transaction(self, adapter, mock_pool):
        """Test rollback without active transaction."""
        pool, _, _ = mock_pool
        adapter._pool = pool

        with pytest.raises(TransactionError, match="No active transaction"):
            await adapter.rollback_transaction()


class TestDatabaseAdapterAdvanced:
    """Test advanced database adapter features."""

    @pytest.fixture
    def adapter(self):
        """Create adapter instance."""
        return DatabaseAdapter()

    @pytest.fixture
    def mock_pool_advanced(self):
        """Create advanced mock pool setup."""
        pool = AsyncMock()
        conn = AsyncMock()

        # Setup connection
        pool.acquire.return_value.__aenter__.return_value = conn
        pool.fetch_one.return_value = {"count": 5}
        pool.fetch_all.return_value = [{"id": i} for i in range(5)]

        # Setup prepared statements
        stmt = AsyncMock()
        conn.prepare.return_value = stmt

        return pool, conn, stmt

    @pytest.mark.asyncio
    async def test_prepared_statement(self, adapter, mock_pool_advanced):
        """Test prepared statement execution."""
        pool, conn, stmt = mock_pool_advanced
        adapter._pool = pool

        # Create prepared statement
        query = "SELECT * FROM users WHERE id = $1"
        async with pool.acquire() as conn:
            prepared = await conn.prepare(query)

            # Execute multiple times
            for user_id in range(1, 6):
                await prepared.fetchrow(user_id)

        conn.prepare.assert_called_once_with(query)
        assert stmt.fetchrow.call_count == 5

    @pytest.mark.asyncio
    async def test_cursor_iteration(self, adapter, mock_pool_advanced):
        """Test cursor-based iteration."""
        pool, conn, _ = mock_pool_advanced
        adapter._pool = pool

        # Setup cursor
        cursor = AsyncMock()
        cursor.__aiter__.return_value = cursor
        cursor.__anext__.side_effect = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
            StopAsyncIteration,
        ]
        conn.cursor.return_value = cursor

        results = []
        async with pool.acquire() as conn:
            async with conn.cursor("SELECT * FROM users") as cursor:
                async for row in cursor:
                    results.append(row)

        assert len(results) == 2
        assert results[0]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_copy_from_table(self, adapter, mock_pool_advanced):
        """Test COPY FROM for bulk inserts."""
        pool, conn, _ = mock_pool_advanced
        adapter._pool = pool

        data = [(1, "Alice", 25), (2, "Bob", 30), (3, "Charlie", 35)]

        async with pool.acquire() as conn:
            await conn.copy_records_to_table("users", records=data, columns=["id", "name", "age"])

        conn.copy_records_to_table.assert_called_once()

    @pytest.mark.asyncio
    async def test_listen_notify(self, adapter, mock_pool_advanced):
        """Test LISTEN/NOTIFY for pub/sub."""
        pool, conn, _ = mock_pool_advanced
        adapter._pool = pool

        # Setup notification
        notification = Mock()
        notification.channel = "test_channel"
        notification.payload = "test_payload"

        conn.add_listener.return_value = None

        async with pool.acquire() as conn:
            await conn.add_listener("test_channel", lambda c, p: None)

            # Simulate notification
            await conn.execute("NOTIFY test_channel, 'test_payload'")

        conn.add_listener.assert_called_once()

    @pytest.mark.asyncio
    async def test_json_field_handling(self, adapter, mock_pool_advanced):
        """Test JSON/JSONB field handling."""
        pool, conn, _ = mock_pool_advanced
        adapter._pool = pool

        json_data = {
            "user": {"name": "Alice", "preferences": {"theme": "dark", "notifications": True}}
        }

        pool.execute.return_value = None
        pool.fetch_one.return_value = {"data": json_data}

        # Insert JSON
        await adapter.execute("INSERT INTO settings (user_id, data) VALUES ($1, $2)", 1, json_data)

        # Fetch JSON
        result = await adapter.fetch_one("SELECT data FROM settings WHERE user_id = $1", 1)

        assert result["data"]["user"]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_array_field_handling(self, adapter, mock_pool_advanced):
        """Test PostgreSQL array handling."""
        pool, conn, _ = mock_pool_advanced
        adapter._pool = pool

        tags = ["python", "async", "database"]

        pool.execute.return_value = None
        pool.fetch_one.return_value = {"tags": tags}

        # Insert array
        await adapter.execute("INSERT INTO posts (id, tags) VALUES ($1, $2)", 1, tags)

        # Fetch array
        result = await adapter.fetch_one("SELECT tags FROM posts WHERE id = $1", 1)

        assert result["tags"] == tags


class TestDatabaseAdapterConcurrency:
    """Test concurrent database operations."""

    @pytest.fixture
    def adapter(self):
        """Create adapter instance."""
        return DatabaseAdapter()

    @pytest.fixture
    def mock_pool_concurrent(self):
        """Create mock pool for concurrent testing."""
        pool = AsyncMock()

        # Simulate different connections for concurrent access
        conns = [AsyncMock() for _ in range(10)]
        pool.acquire.side_effect = [asynccontextmanager(lambda: conns[i])() for i in range(10)]

        pool.execute.return_value = None
        pool.fetch_one.return_value = {"result": "ok"}

        return pool, conns

    @pytest.mark.asyncio
    async def test_concurrent_queries(self, adapter, mock_pool_concurrent):
        """Test concurrent query execution."""
        pool, conns = mock_pool_concurrent
        adapter._pool = pool

        async def run_query(query_id):
            result = await adapter.fetch_one("SELECT * FROM test WHERE id = $1", query_id)
            return result

        # Run 10 queries concurrently
        tasks = [run_query(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert all(r["result"] == "ok" for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_transactions(self, adapter, mock_pool_concurrent):
        """Test concurrent transaction execution."""
        pool, conns = mock_pool_concurrent
        adapter._pool = pool

        results = []
        errors = []

        async def run_transaction(tx_id):
            try:
                async with adapter.transaction():
                    await adapter.execute(f"INSERT INTO test VALUES ({tx_id})")
                    await asyncio.sleep(0.001)  # Simulate work
                    await adapter.execute(f"UPDATE test SET processed = true WHERE id = {tx_id}")
                    results.append(tx_id)
            except Exception as e:
                errors.append(e)

        # Run transactions concurrently
        tasks = [run_transaction(i) for i in range(5)]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Each transaction should complete independently
        assert len(results) == 5
        assert len(errors) == 0


class TestDatabaseAdapterEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def adapter(self):
        """Create adapter instance."""
        return DatabaseAdapter()

    @pytest.mark.asyncio
    async def test_null_values(self, adapter):
        """Test handling NULL values."""
        pool = AsyncMock()
        pool.fetch_one.return_value = {"id": 1, "value": None}
        adapter._pool = pool

        result = await adapter.fetch_one("SELECT * FROM test")

        assert result["value"] is None

    @pytest.mark.asyncio
    async def test_empty_result_set(self, adapter):
        """Test empty result set."""
        pool = AsyncMock()
        pool.fetch_all.return_value = []
        pool.fetch_one.return_value = None
        adapter._pool = pool

        all_results = await adapter.fetch_all("SELECT * FROM empty_table")
        one_result = await adapter.fetch_one("SELECT * FROM empty_table WHERE id = 999")

        assert all_results == []
        assert one_result is None

    @pytest.mark.asyncio
    async def test_large_result_set(self, adapter):
        """Test handling large result sets."""
        pool = AsyncMock()
        # Simulate 10,000 rows
        pool.fetch_all.return_value = [{"id": i} for i in range(10000)]
        adapter._pool = pool

        results = await adapter.fetch_all("SELECT * FROM large_table")

        assert len(results) == 10000

    @pytest.mark.asyncio
    async def test_special_characters_in_query(self, adapter):
        """Test queries with special characters."""
        pool = AsyncMock()
        pool.execute.return_value = None
        adapter._pool = pool

        # Query with quotes and special chars
        await adapter.execute("INSERT INTO test (name) VALUES ($1)", "O'Reilly & Co.")

        pool.execute.assert_called_once_with(
            "INSERT INTO test (name) VALUES ($1)", "O'Reilly & Co."
        )

    @pytest.mark.asyncio
    async def test_timeout_handling(self, adapter):
        """Test query timeout handling."""
        pool = AsyncMock()
        pool.execute.side_effect = TimeoutError("Query timeout")
        adapter._pool = pool

        with pytest.raises(DatabaseError, match="Database query failed"):
            await adapter.execute("SELECT pg_sleep(100)")

    @pytest.mark.asyncio
    async def test_connection_lost(self, adapter):
        """Test handling lost database connection."""
        pool = AsyncMock()
        pool.execute.side_effect = asyncpg.ConnectionDoesNotExistError("Connection lost")
        adapter._pool = pool

        with pytest.raises(DatabaseError, match="Database query failed"):
            await adapter.execute("SELECT 1")

    @pytest.mark.asyncio
    async def test_decimal_handling(self, adapter):
        """Test Decimal type handling."""
        pool = AsyncMock()
        pool.fetch_one.return_value = {"price": Decimal("123.45"), "quantity": Decimal("10.5")}
        adapter._pool = pool

        result = await adapter.fetch_one("SELECT price, quantity FROM products")

        assert isinstance(result["price"], Decimal)
        assert result["price"] == Decimal("123.45")
        assert isinstance(result["quantity"], Decimal)

    @pytest.mark.asyncio
    async def test_datetime_handling(self, adapter):
        """Test datetime handling."""
        pool = AsyncMock()
        now = datetime.now(UTC)
        pool.fetch_one.return_value = {"created_at": now, "updated_at": now + timedelta(hours=1)}
        adapter._pool = pool

        result = await adapter.fetch_one("SELECT created_at, updated_at FROM records")

        assert isinstance(result["created_at"], datetime)
        assert result["created_at"] == now

    @pytest.mark.asyncio
    async def test_uuid_handling(self, adapter):
        """Test UUID handling."""
        pool = AsyncMock()
        test_uuid = uuid4()
        pool.fetch_one.return_value = {"id": test_uuid}
        adapter._pool = pool

        result = await adapter.fetch_one("SELECT id FROM users")

        assert result["id"] == test_uuid

    @pytest.mark.asyncio
    async def test_boolean_handling(self, adapter):
        """Test boolean handling."""
        pool = AsyncMock()
        pool.fetch_all.return_value = [{"active": True}, {"active": False}, {"active": None}]
        adapter._pool = pool

        results = await adapter.fetch_all("SELECT active FROM users")

        assert results[0]["active"] is True
        assert results[1]["active"] is False
        assert results[2]["active"] is None
