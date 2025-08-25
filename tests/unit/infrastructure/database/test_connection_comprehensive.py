"""
Comprehensive unit tests for Database Connection Management.

Tests the database connection module including configuration, connection pooling,
health monitoring, and factory pattern with full coverage.
"""

# Standard library imports
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

# Third-party imports
import psycopg
import pytest
from psycopg_pool import AsyncConnectionPool

# Local imports
from src.application.interfaces.exceptions import ConnectionError, FactoryError
from src.infrastructure.database.connection import (
    ConnectionFactory,
    DatabaseConfig,
    DatabaseConnection,
)


@pytest.fixture
def database_config():
    """Sample database configuration."""
    return DatabaseConfig(
        host="localhost",
        port=5432,
        database="test_db",
        user="test_user",
        password="test_password",
        min_pool_size=2,
        max_pool_size=10,
        max_idle_time=300.0,
        max_lifetime=3600.0,
        command_timeout=60.0,
        server_connection_timeout=30.0,
        ssl_mode="prefer",
    )


@pytest.fixture
def mock_pool():
    """Mock async connection pool."""
    pool = AsyncMock(spec=AsyncConnectionPool)
    pool.closed = False
    pool.max_size = 10
    pool.min_size = 2
    pool.open = AsyncMock()
    pool.close = AsyncMock()
    pool.connection.return_value.__aenter__ = AsyncMock()
    pool.connection.return_value.__aexit__ = AsyncMock()
    return pool


@pytest.fixture
def mock_connection():
    """Mock database connection."""
    conn = AsyncMock()
    cursor = AsyncMock()
    cursor.execute = AsyncMock()
    cursor.fetchone = AsyncMock(return_value={"?column?": 1})
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock()
    conn.cursor.return_value = cursor
    return conn


@pytest.fixture
def database_connection(database_config):
    """Database connection instance."""
    return DatabaseConnection(database_config)


@pytest.fixture(autouse=True)
def reset_connection_factory():
    """Reset ConnectionFactory singleton before each test."""
    ConnectionFactory.reset()
    yield
    ConnectionFactory.reset()


@pytest.mark.unit
class TestDatabaseConfig:
    """Test database configuration."""

    def test_database_config_defaults(self):
        """Test default configuration values."""
        config = DatabaseConfig()

        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "ai_trader"
        assert config.user == ""
        assert config.password is None
        assert config.min_pool_size == 5
        assert config.max_pool_size == 20
        assert config.max_idle_time == 300.0
        assert config.max_lifetime == 3600.0
        assert config.command_timeout == 60.0
        assert config.server_connection_timeout == 30.0
        assert config.ssl_mode == "prefer"

    def test_database_config_from_env(self):
        """Test loading configuration from environment variables."""
        env_vars = {
            "DATABASE_HOST": "db.example.com",
            "DATABASE_PORT": "5433",
            "DATABASE_NAME": "prod_db",
            "DATABASE_USER": "prod_user",
            "DATABASE_PASSWORD": "prod_pass",
            "DATABASE_MIN_POOL_SIZE": "10",
            "DATABASE_MAX_POOL_SIZE": "50",
            "DATABASE_MAX_IDLE_TIME": "600.0",
            "DATABASE_MAX_LIFETIME": "7200.0",
            "DATABASE_COMMAND_TIMEOUT": "120.0",
            "DATABASE_SERVER_TIMEOUT": "60.0",
            "DATABASE_SSL_MODE": "require",
            "DATABASE_SSL_CERT_FILE": "/path/to/cert",
            "DATABASE_SSL_KEY_FILE": "/path/to/key",
            "DATABASE_SSL_CA_FILE": "/path/to/ca",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            with patch("src.infrastructure.database.connection.SecretsManager") as mock_secrets:
                mock_secrets.get_instance.return_value.get_database_config.return_value = {}

                config = DatabaseConfig.from_env()

                assert config.host == "db.example.com"
                assert config.port == 5433
                assert config.database == "prod_db"
                assert config.user == "prod_user"
                assert config.password == "prod_pass"
                assert config.min_pool_size == 10
                assert config.max_pool_size == 50
                assert config.max_idle_time == 600.0
                assert config.max_lifetime == 7200.0
                assert config.command_timeout == 120.0
                assert config.server_connection_timeout == 60.0
                assert config.ssl_mode == "require"
                assert config.ssl_cert_file == "/path/to/cert"
                assert config.ssl_key_file == "/path/to/key"
                assert config.ssl_ca_file == "/path/to/ca"

    def test_database_config_from_env_with_secrets(self):
        """Test loading configuration from environment with secrets fallback."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.infrastructure.database.connection.SecretsManager") as mock_secrets:
                mock_secrets.get_instance.return_value.get_database_config.return_value = {
                    "host": "secret_host",
                    "port": "5434",
                    "database": "secret_db",
                    "user": "secret_user",
                    "password": "secret_pass",
                }

                config = DatabaseConfig.from_env()

                assert config.host == "secret_host"
                assert config.port == 5434
                assert config.database == "secret_db"
                assert config.user == "secret_user"
                assert config.password == "secret_pass"

    def test_database_config_from_url_postgresql(self):
        """Test loading configuration from PostgreSQL URL."""
        url = "postgresql://myuser:mypass@dbhost:5433/mydb"
        config = DatabaseConfig.from_url(url)

        assert config.host == "dbhost"
        assert config.port == 5433
        assert config.database == "mydb"
        assert config.user == "myuser"
        assert config.password == "mypass"

    def test_database_config_from_url_postgres(self):
        """Test loading configuration from postgres URL."""
        url = "postgres://user:pass@host/database"
        config = DatabaseConfig.from_url(url)

        assert config.host == "host"
        assert config.port == 5432  # Default port
        assert config.database == "database"
        assert config.user == "user"
        assert config.password == "pass"

    def test_database_config_from_url_minimal(self):
        """Test loading configuration from minimal URL."""
        url = "postgresql://user@localhost"
        config = DatabaseConfig.from_url(url)

        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "ai_trader"  # Default database
        assert config.user == "user"
        assert config.password is None

    def test_database_config_from_url_invalid_scheme(self):
        """Test invalid URL scheme."""
        url = "mysql://user:pass@host/db"

        with pytest.raises(ValueError, match="Invalid database URL scheme"):
            DatabaseConfig.from_url(url)

    def test_database_config_build_dsn(self, database_config):
        """Test building DSN string."""
        with patch("src.infrastructure.database.connection.DSNBuilder") as mock_builder:
            mock_instance = MagicMock()
            mock_builder.return_value = mock_instance
            mock_instance.with_host.return_value = mock_instance
            mock_instance.with_port.return_value = mock_instance
            mock_instance.with_database.return_value = mock_instance
            mock_instance.with_credentials.return_value = mock_instance
            mock_instance.with_ssl.return_value = mock_instance
            mock_instance.build.return_value = "postgresql://test_dsn"

            dsn = database_config.build_dsn()

            assert dsn == "postgresql://test_dsn"
            mock_instance.with_host.assert_called_once_with("localhost")
            mock_instance.with_port.assert_called_once_with(5432)
            mock_instance.with_database.assert_called_once_with("test_db")
            mock_instance.with_credentials.assert_called_once_with("test_user", "test_password")


@pytest.mark.unit
class TestDatabaseConnection:
    """Test database connection management."""

    def test_initialization(self, database_connection, database_config):
        """Test database connection initialization."""
        assert database_connection.config == database_config
        assert database_connection._pool is None
        assert database_connection._health_check_task is None
        assert database_connection._is_closed is False

    def test_is_connected_no_pool(self, database_connection):
        """Test is_connected when no pool exists."""
        assert database_connection.is_connected is False

    def test_is_connected_with_pool(self, database_connection, mock_pool):
        """Test is_connected with active pool."""
        database_connection._pool = mock_pool
        mock_pool.closed = False
        assert database_connection.is_connected is True

    def test_is_connected_pool_closed(self, database_connection, mock_pool):
        """Test is_connected with closed pool."""
        database_connection._pool = mock_pool
        mock_pool.closed = True
        assert database_connection.is_connected is False

    def test_is_closed(self, database_connection):
        """Test is_closed property."""
        assert database_connection.is_closed is False
        database_connection._is_closed = True
        assert database_connection.is_closed is True

    async def test_connect_success(self, database_connection, mock_pool, mock_connection):
        """Test successful connection."""
        with patch(
            "src.infrastructure.database.connection.AsyncConnectionPool", return_value=mock_pool
        ):
            mock_pool.connection.return_value.__aenter__.return_value = mock_connection

            pool = await database_connection.connect()

            assert pool is mock_pool
            assert database_connection._pool is mock_pool
            mock_pool.open.assert_called_once()
            mock_connection.cursor().execute.assert_called_with("SELECT 1")

    async def test_connect_already_connected(self, database_connection, mock_pool):
        """Test connecting when already connected."""
        database_connection._pool = mock_pool
        mock_pool.closed = False

        pool = await database_connection.connect()

        assert pool is mock_pool
        mock_pool.open.assert_not_called()

    async def test_connect_after_closed(self, database_connection):
        """Test connecting after connection was closed."""
        database_connection._is_closed = True

        with pytest.raises(ConnectionError, match="Connection manager has been closed"):
            await database_connection.connect()

    async def test_connect_timeout_error(self, database_connection):
        """Test connection timeout."""
        with patch("src.infrastructure.database.connection.AsyncConnectionPool") as mock_pool_class:
            mock_pool_class.side_effect = TimeoutError("Connection timeout")

            with pytest.raises(ConnectionError, match="Failed to connect to database"):
                await database_connection.connect()

    async def test_connect_operational_error(self, database_connection):
        """Test connection with operational error."""
        with patch("src.infrastructure.database.connection.AsyncConnectionPool") as mock_pool_class:
            mock_pool_class.side_effect = psycopg.OperationalError("Cannot connect")

            with pytest.raises(ConnectionError, match="Failed to connect to database"):
                await database_connection.connect()

    async def test_disconnect(self, database_connection, mock_pool):
        """Test disconnecting."""
        database_connection._pool = mock_pool
        mock_task = AsyncMock()
        database_connection._health_check_task = mock_task

        await database_connection.disconnect()

        assert database_connection._is_closed is True
        assert database_connection._pool is None
        mock_pool.close.assert_called_once()

    async def test_disconnect_already_closed(self, database_connection):
        """Test disconnecting when already closed."""
        database_connection._is_closed = True

        await database_connection.disconnect()  # Should not raise

    async def test_cleanup(self, database_connection, mock_pool):
        """Test resource cleanup."""
        database_connection._pool = mock_pool
        mock_task = MagicMock()
        mock_task.done.return_value = False
        mock_task.cancel = MagicMock()
        database_connection._health_check_task = mock_task

        await database_connection._cleanup()

        assert database_connection._pool is None
        assert database_connection._health_check_task is None
        mock_pool.close.assert_called_once()
        mock_task.cancel.assert_called_once()

    def test_start_health_monitoring(self, database_connection):
        """Test starting health monitoring."""
        with patch("asyncio.create_task") as mock_create_task:
            mock_task = MagicMock()
            mock_create_task.return_value = mock_task

            database_connection._start_health_monitoring()

            assert database_connection._health_check_task is mock_task
            mock_create_task.assert_called_once()

    def test_start_health_monitoring_already_running(self, database_connection):
        """Test starting health monitoring when already running."""
        mock_task = MagicMock()
        mock_task.done.return_value = False
        database_connection._health_check_task = mock_task

        with patch("asyncio.create_task") as mock_create_task:
            database_connection._start_health_monitoring()
            mock_create_task.assert_not_called()

    async def test_health_monitor_loop_success(
        self, database_connection, mock_pool, mock_connection
    ):
        """Test health monitor loop with successful checks."""
        database_connection._pool = mock_pool
        mock_pool.closed = False
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection

        # Run one iteration of health check
        with patch("asyncio.sleep", side_effect=[None, asyncio.CancelledError()]):
            try:
                await database_connection._health_monitor_loop()
            except asyncio.CancelledError:
                pass

        mock_connection.cursor().execute.assert_called_with("SELECT 1")

    async def test_health_monitor_loop_failure(self, database_connection, mock_pool):
        """Test health monitor loop with failed check."""
        database_connection._pool = mock_pool
        mock_pool.closed = False
        mock_pool.connection.side_effect = Exception("Health check failed")

        with patch("asyncio.sleep", side_effect=[None, asyncio.CancelledError()]):
            with patch("src.infrastructure.database.connection.logger") as mock_logger:
                try:
                    await database_connection._health_monitor_loop()
                except asyncio.CancelledError:
                    pass

                mock_logger.warning.assert_called()

    async def test_get_pool_stats_connected(self, database_connection, mock_pool):
        """Test getting pool statistics when connected."""
        database_connection._pool = mock_pool
        mock_pool.closed = False
        mock_pool.max_size = 20
        mock_pool.min_size = 5

        stats = await database_connection.get_pool_stats()

        assert stats["status"] == "connected"
        assert stats["max_size"] == 20
        assert stats["min_size"] == 5
        assert stats["is_closed"] is False

    async def test_get_pool_stats_disconnected(self, database_connection):
        """Test getting pool statistics when disconnected."""
        stats = await database_connection.get_pool_stats()
        assert stats == {"status": "disconnected"}

    async def test_acquire_connection_success(
        self, database_connection, mock_pool, mock_connection
    ):
        """Test acquiring connection from pool."""
        database_connection._pool = mock_pool
        mock_pool.closed = False
        mock_pool.connection.return_value.__aenter__.return_value = mock_connection

        async with database_connection.acquire() as conn:
            assert conn is mock_connection

    async def test_acquire_connection_not_connected(self, database_connection):
        """Test acquiring connection when not connected."""
        with pytest.raises(ConnectionError, match="Database is not connected"):
            async with database_connection.acquire():
                pass

    async def test_acquire_connection_no_pool(self, database_connection):
        """Test acquiring connection when pool is None."""
        database_connection._pool = None

        with pytest.raises(ConnectionError, match="Database pool is not initialized"):
            async with database_connection.acquire():
                pass

    def test_string_representation(self, database_connection):
        """Test string representation."""
        result = str(database_connection)
        assert "DatabaseConnection" in result
        assert "localhost:5432" in result
        assert "disconnected" in result

        database_connection._pool = MagicMock()
        database_connection._pool.closed = False
        result = str(database_connection)
        assert "connected" in result


@pytest.mark.unit
class TestConnectionFactory:
    """Test connection factory."""

    def test_singleton_pattern(self):
        """Test that factory implements singleton pattern."""
        factory1 = ConnectionFactory()
        factory2 = ConnectionFactory()
        assert factory1 is factory2

    async def test_create_connection_success(self, database_config):
        """Test successful connection creation."""
        with patch.object(DatabaseConnection, "connect", new_callable=AsyncMock) as mock_connect:
            connection = await ConnectionFactory.create_connection(config=database_config)

            assert isinstance(connection, DatabaseConnection)
            assert connection.config == database_config
            mock_connect.assert_called_once()

    async def test_create_connection_default_config(self):
        """Test connection creation with default config."""
        with patch.object(DatabaseConfig, "from_env") as mock_from_env:
            mock_from_env.return_value = DatabaseConfig()
            with patch.object(DatabaseConnection, "connect", new_callable=AsyncMock):
                connection = await ConnectionFactory.create_connection()

                assert isinstance(connection, DatabaseConnection)
                mock_from_env.assert_called_once()

    async def test_create_connection_force_new(self):
        """Test forcing creation of new connection."""
        # Create first connection
        with patch.object(DatabaseConnection, "connect", new_callable=AsyncMock):
            conn1 = await ConnectionFactory.create_connection()

            # Create second connection with force_new
            with patch.object(
                DatabaseConnection, "disconnect", new_callable=AsyncMock
            ) as mock_disconnect:
                conn2 = await ConnectionFactory.create_connection(force_new=True)

                assert conn1 is not conn2
                mock_disconnect.assert_called_once()

    async def test_create_connection_reuse_existing(self):
        """Test reusing existing connection."""
        with patch.object(DatabaseConnection, "connect", new_callable=AsyncMock) as mock_connect:
            conn1 = await ConnectionFactory.create_connection()
            conn2 = await ConnectionFactory.create_connection()

            assert conn1 is conn2
            mock_connect.assert_called_once()  # Only called for first connection

    async def test_create_connection_replace_closed(self):
        """Test replacing closed connection."""
        with patch.object(DatabaseConnection, "connect", new_callable=AsyncMock):
            conn1 = await ConnectionFactory.create_connection()
            conn1._is_closed = True

            conn2 = await ConnectionFactory.create_connection()

            assert conn1 is not conn2

    async def test_create_connection_error(self):
        """Test connection creation error."""
        with patch.object(DatabaseConnection, "connect", new_callable=AsyncMock) as mock_connect:
            mock_connect.side_effect = Exception("Connection failed")

            with pytest.raises(FactoryError, match="Failed to create connection"):
                await ConnectionFactory.create_connection()

    async def test_get_connection_exists(self):
        """Test getting existing connection."""
        with patch.object(DatabaseConnection, "connect", new_callable=AsyncMock):
            created_conn = await ConnectionFactory.create_connection()
            retrieved_conn = await ConnectionFactory.get_connection()

            assert created_conn is retrieved_conn

    async def test_get_connection_not_exists(self):
        """Test getting connection when none exists."""
        with pytest.raises(FactoryError, match="No active database connection"):
            await ConnectionFactory.get_connection()

    async def test_get_connection_closed(self):
        """Test getting connection when it's closed."""
        with patch.object(DatabaseConnection, "connect", new_callable=AsyncMock):
            conn = await ConnectionFactory.create_connection()
            conn._is_closed = True

            with pytest.raises(FactoryError, match="No active database connection"):
                await ConnectionFactory.get_connection()

    async def test_close_all(self):
        """Test closing all connections."""
        with patch.object(DatabaseConnection, "connect", new_callable=AsyncMock):
            conn = await ConnectionFactory.create_connection()

            with patch.object(conn, "disconnect", new_callable=AsyncMock) as mock_disconnect:
                await ConnectionFactory.close_all()

                mock_disconnect.assert_called_once()
                assert ConnectionFactory._connection is None

    async def test_close_all_no_connection(self):
        """Test closing when no connection exists."""
        await ConnectionFactory.close_all()  # Should not raise

    def test_reset(self):
        """Test factory reset."""
        factory = ConnectionFactory()
        ConnectionFactory._connection = MagicMock()

        ConnectionFactory.reset()

        assert ConnectionFactory._instance is None
        assert ConnectionFactory._connection is None
