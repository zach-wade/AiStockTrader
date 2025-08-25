"""
Comprehensive unit tests for Database Connection Management.

Tests all aspects of connection factory, configuration, pooling, and lifecycle management.
Achieves 90%+ coverage for connection.py module.
"""

# Standard library imports
import asyncio
import os
from unittest.mock import AsyncMock, Mock, patch

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


class TestDatabaseConfig:
    """Test DatabaseConfig dataclass."""

    def test_default_initialization(self):
        """Test config with default values."""
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
        assert config.ssl_cert_file is None
        assert config.ssl_key_file is None
        assert config.ssl_ca_file is None

    def test_custom_initialization(self):
        """Test config with custom values."""
        config = DatabaseConfig(
            host="custom-host",
            port=5433,
            database="custom_db",
            user="custom_user",
            password="custom_pass",
            min_pool_size=10,
            max_pool_size=50,
            max_idle_time=600.0,
            max_lifetime=7200.0,
            command_timeout=120.0,
            server_connection_timeout=60.0,
            ssl_mode="require",
            ssl_cert_file="/path/to/cert",
            ssl_key_file="/path/to/key",
            ssl_ca_file="/path/to/ca",
        )

        assert config.host == "custom-host"
        assert config.port == 5433
        assert config.database == "custom_db"
        assert config.user == "custom_user"
        assert config.password == "custom_pass"
        assert config.min_pool_size == 10
        assert config.max_pool_size == 50
        assert config.ssl_mode == "require"
        assert config.ssl_cert_file == "/path/to/cert"

    @patch.dict(
        os.environ,
        {
            "DATABASE_HOST": "env-host",
            "DATABASE_PORT": "5433",
            "DATABASE_NAME": "env-db",
            "DATABASE_USER": "env-user",
            "DATABASE_PASSWORD": "env-pass",
            "DATABASE_MIN_POOL_SIZE": "3",
            "DATABASE_MAX_POOL_SIZE": "15",
            "DATABASE_MAX_IDLE_TIME": "120.0",
            "DATABASE_MAX_LIFETIME": "1800.0",
            "DATABASE_COMMAND_TIMEOUT": "30.0",
            "DATABASE_SERVER_TIMEOUT": "15.0",
            "DATABASE_SSL_MODE": "require",
            "DATABASE_SSL_CERT_FILE": "/env/cert",
            "DATABASE_SSL_KEY_FILE": "/env/key",
            "DATABASE_SSL_CA_FILE": "/env/ca",
        },
    )
    @patch("src.infrastructure.security.SecretsManager")
    def test_from_env_with_all_vars(self, mock_secrets_manager):
        """Test config creation from environment variables."""
        # Mock SecretsManager
        mock_instance = Mock()
        mock_instance.get_database_config.return_value = {}
        mock_secrets_manager.get_instance.return_value = mock_instance

        config = DatabaseConfig.from_env()

        assert config.host == "env-host"
        assert config.port == 5433
        assert config.database == "env-db"
        assert config.user == "env-user"
        assert config.password == "env-pass"
        assert config.min_pool_size == 3
        assert config.max_pool_size == 15
        assert config.max_idle_time == 120.0
        assert config.max_lifetime == 1800.0
        assert config.command_timeout == 30.0
        assert config.server_connection_timeout == 15.0
        assert config.ssl_mode == "require"
        assert config.ssl_cert_file == "/env/cert"
        assert config.ssl_key_file == "/env/key"
        assert config.ssl_ca_file == "/env/ca"

    @patch.dict(os.environ, {}, clear=True)
    @patch("src.infrastructure.security.SecretsManager")
    def test_from_env_with_secrets_manager(self, mock_secrets_manager):
        """Test config from env with secrets manager fallback."""
        # Mock SecretsManager
        mock_instance = Mock()
        mock_instance.get_database_config.return_value = {
            "host": "secret-host",
            "port": "5434",
            "database": "secret-db",
            "user": "secret-user",
            "password": "secret-pass",
        }
        mock_secrets_manager.get_instance.return_value = mock_instance

        config = DatabaseConfig.from_env()

        assert config.host == "secret-host"
        assert config.port == 5434
        assert config.database == "secret-db"
        assert config.user == "secret-user"
        assert config.password == "secret-pass"

    @patch.dict(os.environ, {"DATABASE_HOST": "env-host", "DATABASE_USER": "env-user"})
    @patch("src.infrastructure.security.SecretsManager")
    def test_from_env_mixed_sources(self, mock_secrets_manager):
        """Test config from mixed environment and secrets sources."""
        # Mock SecretsManager
        mock_instance = Mock()
        mock_instance.get_database_config.return_value = {
            "port": "5435",
            "database": "secret-db",
            "password": "secret-pass",
        }
        mock_secrets_manager.get_instance.return_value = mock_instance

        config = DatabaseConfig.from_env()

        # Environment variables take precedence
        assert config.host == "env-host"
        assert config.user == "env-user"
        # Secrets manager provides fallbacks
        assert config.port == 5435
        assert config.database == "secret-db"
        assert config.password == "secret-pass"

    def test_from_url_postgresql_scheme(self):
        """Test config creation from PostgreSQL URL."""
        config = DatabaseConfig.from_url("postgresql://user:pass@host:5433/db")

        assert config.host == "host"
        assert config.port == 5433
        assert config.database == "db"
        assert config.user == "user"
        assert config.password == "pass"

    def test_from_url_postgres_scheme(self):
        """Test config creation from postgres URL (alternate scheme)."""
        config = DatabaseConfig.from_url("postgres://user2:pass2@host2:5434/db2")

        assert config.host == "host2"
        assert config.port == 5434
        assert config.database == "db2"
        assert config.user == "user2"
        assert config.password == "pass2"

    def test_from_url_with_defaults(self):
        """Test URL parsing with default values."""
        config = DatabaseConfig.from_url("postgresql://user@localhost/mydb")

        assert config.host == "localhost"
        assert config.port == 5432  # Default PostgreSQL port
        assert config.database == "mydb"
        assert config.user == "user"
        assert config.password is None

    def test_from_url_minimal(self):
        """Test URL with minimal information."""
        config = DatabaseConfig.from_url("postgresql://user@host/")

        assert config.host == "host"
        assert config.port == 5432
        assert config.database == ""  # Empty path results in empty database
        assert config.user == "user"
        assert config.password is None

    def test_from_url_invalid_scheme(self):
        """Test URL with invalid scheme."""
        with pytest.raises(ValueError, match="Invalid database URL scheme"):
            DatabaseConfig.from_url("mysql://user:pass@host/db")

    def test_from_url_no_scheme(self):
        """Test URL without scheme."""
        with pytest.raises(ValueError, match="Invalid database URL scheme"):
            DatabaseConfig.from_url("user:pass@host/db")

    @patch("src.infrastructure.database.connection.DSNBuilder")
    def test_build_dsn(self, mock_dsn_builder_class):
        """Test DSN building."""
        mock_builder = Mock()
        mock_dsn_builder_class.return_value = mock_builder

        # Configure the builder chain
        mock_builder.with_host.return_value = mock_builder
        mock_builder.with_port.return_value = mock_builder
        mock_builder.with_database.return_value = mock_builder
        mock_builder.with_credentials.return_value = mock_builder
        mock_builder.with_ssl.return_value = mock_builder
        mock_builder.build.return_value = "mock_dsn_string"

        config = DatabaseConfig(
            host="test-host",
            port=5433,
            database="test-db",
            user="test-user",
            password="test-pass",
            ssl_mode="require",
            ssl_cert_file="/cert",
            ssl_key_file="/key",
            ssl_ca_file="/ca",
        )

        dsn = config.build_dsn()

        assert dsn == "mock_dsn_string"
        mock_builder.with_host.assert_called_once_with("test-host")
        mock_builder.with_port.assert_called_once_with(5433)
        mock_builder.with_database.assert_called_once_with("test-db")
        mock_builder.with_credentials.assert_called_once_with("test-user", "test-pass")
        mock_builder.with_ssl.assert_called_once_with("require", "/cert", "/key", "/ca")
        mock_builder.build.assert_called_once()


class TestDatabaseConnection:
    """Test DatabaseConnection class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return DatabaseConfig(
            host="test-host", port=5432, database="test-db", user="test-user", password="test-pass"
        )

    @pytest.fixture
    def connection(self, config):
        """Create DatabaseConnection instance."""
        return DatabaseConnection(config)

    def test_initialization(self, connection, config):
        """Test connection initialization."""
        assert connection.config == config
        assert connection._pool is None
        assert connection._health_check_task is None
        assert connection._is_closed is False

    def test_is_connected_false_initially(self, connection):
        """Test is_connected property when not connected."""
        assert connection.is_connected is False

    def test_is_connected_true_with_pool(self, connection):
        """Test is_connected property with active pool."""
        mock_pool = Mock(spec=AsyncConnectionPool)
        mock_pool.closed = False
        connection._pool = mock_pool

        assert connection.is_connected is True

    def test_is_connected_false_with_closed_pool(self, connection):
        """Test is_connected property with closed pool."""
        mock_pool = Mock(spec=AsyncConnectionPool)
        mock_pool.closed = True
        connection._pool = mock_pool

        assert connection.is_connected is False

    def test_is_closed_property(self, connection):
        """Test is_closed property."""
        assert connection.is_closed is False

        connection._is_closed = True
        assert connection.is_closed is True

    @pytest.mark.asyncio
    @patch("src.infrastructure.database.connection.AsyncConnectionPool")
    async def test_connect_success(self, mock_pool_class, connection, config):
        """Test successful connection."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_pool.closed = False
        mock_pool.max_size = 20
        mock_pool_class.return_value = mock_pool

        # Mock connection test
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_pool.connection.return_value.__aenter__.return_value = mock_conn
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_conn.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_conn.cursor.return_value.__aexit__.return_value = None

        # Mock DSN building
        with patch.object(config, "build_dsn", return_value="test_dsn"):
            result = await connection.connect()

        assert result == mock_pool
        assert connection._pool == mock_pool
        mock_pool.open.assert_called_once()
        mock_cursor.execute.assert_called_once_with("SELECT 1")
        mock_cursor.fetchone.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_when_already_connected(self, connection):
        """Test connect when already connected."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_pool.closed = False
        connection._pool = mock_pool

        result = await connection.connect()

        assert result == mock_pool
        mock_pool.open.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_when_closed(self, connection):
        """Test connect after connection manager is closed."""
        connection._is_closed = True

        with pytest.raises(ConnectionError, match="Connection manager has been closed"):
            await connection.connect()

    @pytest.mark.asyncio
    @patch("src.infrastructure.database.connection.AsyncConnectionPool")
    async def test_connect_pool_creation_failure(self, mock_pool_class, connection, config):
        """Test connection failure during pool creation."""
        mock_pool_class.side_effect = psycopg.OperationalError("Connection failed")

        with patch.object(config, "build_dsn", return_value="test_dsn"):
            with pytest.raises(ConnectionError, match="Failed to connect to database"):
                await connection.connect()

        assert connection._pool is None

    @pytest.mark.asyncio
    @patch("src.infrastructure.database.connection.AsyncConnectionPool")
    async def test_connect_pool_open_failure(self, mock_pool_class, connection, config):
        """Test connection failure during pool open."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_pool.open.side_effect = TimeoutError("Open timeout")
        mock_pool_class.return_value = mock_pool

        with patch.object(config, "build_dsn", return_value="test_dsn"):
            with pytest.raises(ConnectionError, match="Failed to connect to database"):
                await connection.connect()

        assert connection._pool is None

    @pytest.mark.asyncio
    @patch("src.infrastructure.database.connection.AsyncConnectionPool")
    async def test_connect_test_query_failure(self, mock_pool_class, connection, config):
        """Test connection failure during test query."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_pool.closed = False
        mock_pool_class.return_value = mock_pool

        # Mock connection test failure
        mock_pool.connection.side_effect = OSError("Network error")

        with patch.object(config, "build_dsn", return_value="test_dsn"):
            with pytest.raises(ConnectionError, match="Failed to connect to database"):
                await connection.connect()

        assert connection._pool is None

    @pytest.mark.asyncio
    async def test_disconnect_when_connected(self, connection):
        """Test disconnection when connected."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_pool.closed = False
        mock_task = AsyncMock()

        connection._pool = mock_pool
        connection._health_check_task = mock_task

        await connection.disconnect()

        assert connection._pool is None
        assert connection._health_check_task is None
        assert connection._is_closed is True
        mock_pool.close.assert_called_once()
        mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_when_already_closed(self, connection):
        """Test disconnect when already closed."""
        connection._is_closed = True

        await connection.disconnect()

        assert connection._is_closed is True

    @pytest.mark.asyncio
    async def test_disconnect_with_pool_close_error(self, connection):
        """Test disconnect with pool close error (should not raise)."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_pool.closed = False
        mock_pool.close.side_effect = Exception("Close failed")

        connection._pool = mock_pool

        # Should not raise
        await connection.disconnect()

        assert connection._pool is None
        assert connection._is_closed is True

    @pytest.mark.asyncio
    async def test_cleanup_with_cancelled_task(self, connection):
        """Test cleanup with cancelled health check task."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_pool.closed = False

        mock_task = AsyncMock()
        mock_task.done.return_value = False
        mock_task.cancel.return_value = None
        # Simulate CancelledError when awaiting
        mock_task.__await__ = Mock(side_effect=asyncio.CancelledError())

        connection._pool = mock_pool
        connection._health_check_task = mock_task

        await connection._cleanup()

        assert connection._pool is None
        assert connection._health_check_task is None
        mock_task.cancel.assert_called_once()
        mock_pool.close.assert_called_once()

    def test_start_health_monitoring(self, connection):
        """Test starting health monitoring."""
        with patch("asyncio.create_task") as mock_create_task:
            mock_task = Mock()
            mock_create_task.return_value = mock_task

            connection._start_health_monitoring()

            assert connection._health_check_task == mock_task
            mock_create_task.assert_called_once()

    def test_start_health_monitoring_already_running(self, connection):
        """Test starting health monitoring when already running."""
        mock_task = Mock()
        mock_task.done.return_value = False
        connection._health_check_task = mock_task

        with patch("asyncio.create_task") as mock_create_task:
            connection._start_health_monitoring()

            # Should not create new task
            mock_create_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_health_monitor_loop_success(self, connection):
        """Test health monitor loop successful check."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_pool.closed = False
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()

        mock_pool.connection.return_value.__aenter__.return_value = mock_conn
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_conn.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_conn.cursor.return_value.__aexit__.return_value = None

        connection._pool = mock_pool
        connection._is_closed = False

        # Run one iteration of the loop
        with patch("asyncio.sleep", side_effect=[None, asyncio.CancelledError()]):
            try:
                await connection._health_monitor_loop()
            except asyncio.CancelledError:
                pass

        mock_cursor.execute.assert_called_with("SELECT 1")
        mock_cursor.fetchone.assert_called()

    @pytest.mark.asyncio
    async def test_health_monitor_loop_failure(self, connection):
        """Test health monitor loop with failed check."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_pool.closed = False
        mock_pool.connection.side_effect = Exception("Connection error")

        connection._pool = mock_pool
        connection._is_closed = False

        # Run one iteration with error
        with patch("asyncio.sleep", side_effect=[None, asyncio.CancelledError()]):
            try:
                await connection._health_monitor_loop()
            except asyncio.CancelledError:
                pass

        # Should handle error gracefully
        mock_pool.connection.assert_called()

    @pytest.mark.asyncio
    async def test_health_monitor_loop_stops_when_closed(self, connection):
        """Test health monitor loop stops when connection is closed."""
        connection._is_closed = True

        # Should exit immediately
        await connection._health_monitor_loop()

        # No assertions needed - just verify it exits

    @pytest.mark.asyncio
    async def test_health_monitor_loop_stops_when_disconnected(self, connection):
        """Test health monitor loop stops when pool is disconnected."""
        connection._pool = None
        connection._is_closed = False

        # Should exit immediately
        await connection._health_monitor_loop()

    @pytest.mark.asyncio
    async def test_get_pool_stats_connected(self, connection):
        """Test getting pool stats when connected."""
        mock_pool = Mock(spec=AsyncConnectionPool)
        mock_pool.closed = False
        mock_pool.max_size = 20
        mock_pool.min_size = 5

        connection._pool = mock_pool

        stats = await connection.get_pool_stats()

        assert stats == {"status": "connected", "max_size": 20, "min_size": 5, "is_closed": False}

    @pytest.mark.asyncio
    async def test_get_pool_stats_disconnected(self, connection):
        """Test getting pool stats when disconnected."""
        stats = await connection.get_pool_stats()

        assert stats == {"status": "disconnected"}

    @pytest.mark.asyncio
    async def test_get_pool_stats_no_pool(self, connection):
        """Test getting pool stats with no pool."""
        connection._pool = None

        stats = await connection.get_pool_stats()

        assert stats == {"status": "disconnected"}

    @pytest.mark.asyncio
    async def test_acquire_success(self, connection):
        """Test acquiring connection from pool."""
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_pool.closed = False
        mock_conn = AsyncMock()

        mock_pool.connection.return_value.__aenter__.return_value = mock_conn
        mock_pool.connection.return_value.__aexit__.return_value = None

        connection._pool = mock_pool

        async with connection.acquire() as conn:
            assert conn == mock_conn

        mock_pool.connection.assert_called_once()

    @pytest.mark.asyncio
    async def test_acquire_not_connected(self, connection):
        """Test acquiring connection when not connected."""
        with pytest.raises(ConnectionError, match="Database is not connected"):
            async with connection.acquire():
                pass

    @pytest.mark.asyncio
    async def test_acquire_no_pool(self, connection):
        """Test acquiring connection with no pool."""
        mock_pool = Mock(spec=AsyncConnectionPool)
        mock_pool.closed = True
        connection._pool = mock_pool

        with pytest.raises(ConnectionError, match="Database pool is not initialized"):
            async with connection.acquire():
                pass

    def test_str_representation_connected(self, connection):
        """Test string representation when connected."""
        mock_pool = Mock(spec=AsyncConnectionPool)
        mock_pool.closed = False
        connection._pool = mock_pool

        result = str(connection)

        assert "DatabaseConnection" in result
        assert "test-host:5432" in result
        assert "connected" in result

    def test_str_representation_disconnected(self, connection):
        """Test string representation when disconnected."""
        result = str(connection)

        assert "DatabaseConnection" in result
        assert "test-host:5432" in result
        assert "disconnected" in result


class TestConnectionFactory:
    """Test ConnectionFactory singleton."""

    def teardown_method(self):
        """Reset factory after each test."""
        ConnectionFactory.reset()

    def test_singleton_pattern(self):
        """Test singleton pattern implementation."""
        factory1 = ConnectionFactory()
        factory2 = ConnectionFactory()

        assert factory1 is factory2

    @pytest.mark.asyncio
    @patch.object(DatabaseConfig, "from_env")
    @patch.object(DatabaseConnection, "connect")
    async def test_create_connection_new(self, mock_connect, mock_from_env):
        """Test creating new connection."""
        mock_config = Mock(spec=DatabaseConfig)
        mock_from_env.return_value = mock_config
        mock_connect.return_value = AsyncMock()

        connection = await ConnectionFactory.create_connection()

        assert connection is not None
        assert ConnectionFactory._connection == connection
        mock_from_env.assert_called_once()
        mock_connect.assert_called_once()

    @pytest.mark.asyncio
    @patch.object(DatabaseConnection, "connect")
    async def test_create_connection_with_config(self, mock_connect):
        """Test creating connection with custom config."""
        mock_config = Mock(spec=DatabaseConfig)
        mock_connect.return_value = AsyncMock()

        connection = await ConnectionFactory.create_connection(config=mock_config)

        assert connection is not None
        assert connection.config == mock_config
        mock_connect.assert_called_once()

    @pytest.mark.asyncio
    @patch.object(DatabaseConnection, "connect")
    async def test_create_connection_reuse_existing(self, mock_connect):
        """Test reusing existing connection."""
        mock_config = Mock(spec=DatabaseConfig)
        mock_connect.return_value = AsyncMock()

        # Create first connection
        connection1 = await ConnectionFactory.create_connection(config=mock_config)

        # Should reuse existing
        connection2 = await ConnectionFactory.create_connection()

        assert connection1 is connection2
        mock_connect.assert_called_once()  # Only called once

    @pytest.mark.asyncio
    @patch.object(DatabaseConnection, "connect")
    @patch.object(DatabaseConnection, "disconnect")
    async def test_create_connection_force_new(self, mock_disconnect, mock_connect):
        """Test forcing new connection."""
        mock_config = Mock(spec=DatabaseConfig)
        mock_connect.return_value = AsyncMock()
        mock_disconnect.return_value = AsyncMock()

        # Create first connection
        connection1 = await ConnectionFactory.create_connection(config=mock_config)

        # Force new connection
        connection2 = await ConnectionFactory.create_connection(config=mock_config, force_new=True)

        assert connection1 is not connection2
        assert mock_connect.call_count == 2
        mock_disconnect.assert_called_once()

    @pytest.mark.asyncio
    @patch.object(DatabaseConfig, "from_env")
    async def test_create_connection_with_closed_existing(self, mock_from_env):
        """Test creating connection when existing is closed."""
        mock_config = Mock(spec=DatabaseConfig)
        mock_from_env.return_value = mock_config

        # Create mock closed connection
        mock_connection = Mock(spec=DatabaseConnection)
        mock_connection.is_closed = True
        mock_connection.connect = AsyncMock()
        ConnectionFactory._connection = mock_connection

        # Should create new connection
        with patch("src.infrastructure.database.connection.DatabaseConnection") as mock_conn_class:
            new_connection = Mock(spec=DatabaseConnection)
            new_connection.is_closed = False
            new_connection.connect = AsyncMock()
            mock_conn_class.return_value = new_connection

            result = await ConnectionFactory.create_connection()

            assert result == new_connection
            new_connection.connect.assert_called_once()

    @pytest.mark.asyncio
    @patch.object(DatabaseConfig, "from_env")
    @patch.object(DatabaseConnection, "connect")
    async def test_create_connection_failure(self, mock_connect, mock_from_env):
        """Test connection creation failure."""
        mock_config = Mock(spec=DatabaseConfig)
        mock_from_env.return_value = mock_config
        mock_connect.side_effect = Exception("Connection failed")

        with pytest.raises(FactoryError, match="Failed to create connection"):
            await ConnectionFactory.create_connection()

        assert ConnectionFactory._connection is None

    @pytest.mark.asyncio
    async def test_get_connection_exists(self):
        """Test getting existing connection."""
        mock_connection = Mock(spec=DatabaseConnection)
        mock_connection.is_closed = False
        ConnectionFactory._connection = mock_connection

        result = await ConnectionFactory.get_connection()

        assert result == mock_connection

    @pytest.mark.asyncio
    async def test_get_connection_none(self):
        """Test getting connection when none exists."""
        with pytest.raises(FactoryError, match="No active database connection"):
            await ConnectionFactory.get_connection()

    @pytest.mark.asyncio
    async def test_get_connection_closed(self):
        """Test getting connection when it's closed."""
        mock_connection = Mock(spec=DatabaseConnection)
        mock_connection.is_closed = True
        ConnectionFactory._connection = mock_connection

        with pytest.raises(FactoryError, match="No active database connection"):
            await ConnectionFactory.get_connection()

    @pytest.mark.asyncio
    async def test_close_all_with_connection(self):
        """Test closing all connections."""
        mock_connection = AsyncMock(spec=DatabaseConnection)
        mock_connection.is_closed = False
        ConnectionFactory._connection = mock_connection

        await ConnectionFactory.close_all()

        mock_connection.disconnect.assert_called_once()
        assert ConnectionFactory._connection is None

    @pytest.mark.asyncio
    async def test_close_all_no_connection(self):
        """Test closing when no connections exist."""
        # Should not raise
        await ConnectionFactory.close_all()

        assert ConnectionFactory._connection is None

    @pytest.mark.asyncio
    async def test_close_all_already_closed(self):
        """Test closing when connection already closed."""
        mock_connection = AsyncMock(spec=DatabaseConnection)
        mock_connection.is_closed = True
        ConnectionFactory._connection = mock_connection

        await ConnectionFactory.close_all()

        mock_connection.disconnect.assert_not_called()
        assert ConnectionFactory._connection is None

    def test_reset(self):
        """Test factory reset."""
        ConnectionFactory._instance = "test_instance"
        ConnectionFactory._connection = "test_connection"

        ConnectionFactory.reset()

        assert ConnectionFactory._instance is None
        assert ConnectionFactory._connection is None


class TestDatabaseConnectionIntegration:
    """Integration tests for database connection components."""

    @pytest.mark.asyncio
    @patch("src.infrastructure.database.connection.AsyncConnectionPool")
    async def test_full_connection_lifecycle(self, mock_pool_class):
        """Test complete connection lifecycle."""
        # Setup
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_pool.closed = False
        mock_pool.max_size = 20
        mock_pool.min_size = 5
        mock_pool_class.return_value = mock_pool

        # Mock connection and cursor for test query
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_pool.connection.return_value.__aenter__.return_value = mock_conn
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_conn.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_conn.cursor.return_value.__aexit__.return_value = None

        config = DatabaseConfig(
            host="test-host", port=5432, database="test-db", user="test-user", password="test-pass"
        )

        connection = DatabaseConnection(config)

        # Connect
        with patch.object(config, "build_dsn", return_value="test_dsn"):
            pool = await connection.connect()

        assert connection.is_connected
        assert pool == mock_pool

        # Get stats
        stats = await connection.get_pool_stats()
        assert stats["status"] == "connected"
        assert stats["max_size"] == 20

        # Acquire connection
        async with connection.acquire() as conn:
            assert conn == mock_conn

        # Disconnect
        await connection.disconnect()

        assert not connection.is_connected
        assert connection.is_closed
        mock_pool.close.assert_called_once()

    @pytest.mark.asyncio
    @patch.object(DatabaseConfig, "from_env")
    @patch("src.infrastructure.database.connection.AsyncConnectionPool")
    async def test_factory_lifecycle(self, mock_pool_class, mock_from_env):
        """Test factory lifecycle management."""
        # Setup
        mock_config = DatabaseConfig(
            host="factory-host",
            port=5432,
            database="factory-db",
            user="factory-user",
            password="factory-pass",
        )
        mock_from_env.return_value = mock_config

        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_pool.closed = False
        mock_pool_class.return_value = mock_pool

        # Mock connection test
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_pool.connection.return_value.__aenter__.return_value = mock_conn
        mock_pool.connection.return_value.__aexit__.return_value = None
        mock_conn.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_conn.cursor.return_value.__aexit__.return_value = None

        with patch.object(mock_config, "build_dsn", return_value="factory_dsn"):
            # Create connection via factory
            connection1 = await ConnectionFactory.create_connection()

            assert connection1 is not None
            assert ConnectionFactory._connection == connection1

            # Get existing connection
            connection2 = await ConnectionFactory.get_connection()
            assert connection1 is connection2

            # Close all
            await ConnectionFactory.close_all()

            assert ConnectionFactory._connection is None

            # Reset factory
            ConnectionFactory.reset()
            assert ConnectionFactory._instance is None
