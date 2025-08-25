"""
Unit tests for Database Connection Management.

Tests connection factory, configuration, and pool management functionality.
"""

# Standard library imports
import os
from unittest.mock import AsyncMock, Mock, patch

# Third-party imports
import pytest

# Local imports
from src.application.interfaces.exceptions import ConfigurationError, ConnectionError
from src.infrastructure.database.connection import (
    ConnectionFactory,
    DatabaseConfig,
    DatabaseConnection,
)

# Test constants for passwords - these are test-only values
# Using test fixture names that don't trigger S105
TEST_PASSWORD = "test_password"  # noqa: S105
ENV_PASSWORD = "env_password"  # noqa: S105
CUSTOM_PASSWORD = "custom_password"  # noqa: S105
URL_PASSWORD = "password"  # noqa: S105


@pytest.fixture
def valid_config():
    """Valid database configuration."""
    return DatabaseConfig(
        host="localhost",
        port=5432,
        database="test_db",
        user="test_user",
        password=TEST_PASSWORD,
        min_pool_size=1,
        max_pool_size=10,
        command_timeout=30.0,
    )


@pytest.fixture
def config_from_env():
    """Configuration loaded from environment."""
    return {
        "DATABASE_HOST": "env_host",
        "DATABASE_PORT": "5433",
        "DATABASE_NAME": "env_db",
        "DATABASE_USER": "env_user",
        "DATABASE_PASSWORD": ENV_PASSWORD,
        "DATABASE_MIN_POOL_SIZE": "2",
        "DATABASE_MAX_POOL_SIZE": "20",
        "DATABASE_COMMAND_TIMEOUT": "60.0",
    }


@pytest.mark.unit
class TestDatabaseConfig:
    """Test DatabaseConfig dataclass."""

    def test_config_creation_with_defaults(self):
        """Test config creation with default values."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_db",
            user="test_user",
            password=TEST_PASSWORD,
        )

        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "test_db"
        assert config.user == "test_user"
        assert config.password == TEST_PASSWORD
        assert config.min_pool_size == 1
        assert config.max_pool_size == 10
        assert config.command_timeout == 30.0

    def test_config_creation_with_custom_values(self):
        """Test config creation with custom values."""
        config = DatabaseConfig(
            host="custom_host",
            port=5433,
            database="custom_db",
            user="custom_user",
            password=CUSTOM_PASSWORD,
            min_pool_size=5,
            max_pool_size=50,
            command_timeout=120.0,
        )

        assert config.host == "custom_host"
        assert config.port == 5433
        assert config.min_pool_size == 5
        assert config.max_pool_size == 50
        assert config.command_timeout == 120.0

    def test_config_connection_string(self):
        """Test connection string generation."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_db",
            user="test_user",
            password=TEST_PASSWORD,
        )

        expected = f"postgresql://test_user:{TEST_PASSWORD}@localhost:5432/test_db"
        assert config.connection_string == expected

    def test_config_connection_string_no_password(self):
        """Test connection string without password."""
        config = DatabaseConfig(
            host="localhost", port=5432, database="test_db", user="test_user", password=""
        )

        expected = "postgresql://test_user@localhost:5432/test_db"
        assert config.connection_string == expected

    def test_config_validation_invalid_port(self):
        """Test config validation with invalid port."""
        with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
            DatabaseConfig(
                host="localhost",
                port=0,
                database="test_db",
                user="test_user",
                password=TEST_PASSWORD,
            )

        with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
            DatabaseConfig(
                host="localhost",
                port=70000,
                database="test_db",
                user="test_user",
                password=TEST_PASSWORD,
            )

    def test_config_validation_invalid_pool_size(self):
        """Test config validation with invalid pool sizes."""
        with pytest.raises(ValueError, match="min_pool_size must be at least 1"):
            DatabaseConfig(
                host="localhost",
                port=5432,
                database="test_db",
                user="test_user",
                password=TEST_PASSWORD,
                min_pool_size=0,
            )

        with pytest.raises(ValueError, match="max_pool_size must be greater than min_pool_size"):
            DatabaseConfig(
                host="localhost",
                port=5432,
                database="test_db",
                user="test_user",
                password=TEST_PASSWORD,
                min_pool_size=5,
                max_pool_size=3,
            )

    def test_config_validation_invalid_timeout(self):
        """Test config validation with invalid timeout."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            DatabaseConfig(
                host="localhost",
                port=5432,
                database="test_db",
                user="test_user",
                password=TEST_PASSWORD,
                command_timeout=-1.0,
            )

    def test_config_validation_empty_required_fields(self):
        """Test config validation with empty required fields."""
        with pytest.raises(ValueError, match="host cannot be empty"):
            DatabaseConfig(
                host="", port=5432, database="test_db", user="test_user", password=TEST_PASSWORD
            )

        with pytest.raises(ValueError, match="database cannot be empty"):
            DatabaseConfig(
                host="localhost", port=5432, database="", user="test_user", password=TEST_PASSWORD
            )

        with pytest.raises(ValueError, match="user cannot be empty"):
            DatabaseConfig(
                host="localhost", port=5432, database="test_db", user="", password=TEST_PASSWORD
            )


@pytest.mark.unit
class TestConnectionFactory:
    """Test ConnectionFactory functionality."""

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_missing_required_vars(self):
        """Test config creation from env with missing required variables."""
        with pytest.raises(ConfigurationError, match="Missing required environment variable"):
            ConnectionFactory.from_env()

    @patch.dict(
        os.environ,
        {
            "DATABASE_HOST": "env_host",
            "DATABASE_PORT": "5433",
            "DATABASE_NAME": "env_db",
            "DATABASE_USER": "env_user",
            "DATABASE_PASSWORD": ENV_PASSWORD,
        },
    )
    def test_from_env_with_required_vars(self):
        """Test config creation from env with required variables."""
        config = ConnectionFactory.from_env()

        assert config.host == "env_host"
        assert config.port == 5433
        assert config.database == "env_db"
        assert config.user == "env_user"
        assert config.password == ENV_PASSWORD
        # Defaults should be used for optional vars
        assert config.min_pool_size == 1
        assert config.max_pool_size == 10
        assert config.command_timeout == 30.0

    @patch.dict(
        os.environ,
        {
            "DATABASE_HOST": "env_host",
            "DATABASE_PORT": "5433",
            "DATABASE_NAME": "env_db",
            "DATABASE_USER": "env_user",
            "DATABASE_PASSWORD": ENV_PASSWORD,
            "DATABASE_MIN_POOL_SIZE": "5",
            "DATABASE_MAX_POOL_SIZE": "25",
            "DATABASE_COMMAND_TIMEOUT": "60.0",
        },
    )
    def test_from_env_with_all_vars(self):
        """Test config creation from env with all variables."""
        config = ConnectionFactory.from_env()

        assert config.host == "env_host"
        assert config.port == 5433
        assert config.database == "env_db"
        assert config.user == "env_user"
        assert config.password == ENV_PASSWORD
        assert config.min_pool_size == 5
        assert config.max_pool_size == 25
        assert config.command_timeout == 60.0

    @patch.dict(
        os.environ,
        {
            "DATABASE_HOST": "localhost",
            "DATABASE_PORT": "invalid_port",
            "DATABASE_NAME": "test_db",
            "DATABASE_USER": "test_user",
            "DATABASE_PASSWORD": TEST_PASSWORD,
        },
    )
    def test_from_env_invalid_port_type(self):
        """Test config creation from env with invalid port type."""
        with pytest.raises(ConfigurationError, match="Invalid port value"):
            ConnectionFactory.from_env()

    def test_from_url_valid_postgresql_url(self):
        """Test config creation from valid PostgreSQL URL."""
        url = f"postgresql://user:{URL_PASSWORD}@localhost:5432/database"
        config = ConnectionFactory.from_url(url)

        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "database"
        assert config.user == "user"
        assert config.password == URL_PASSWORD

    def test_from_url_without_password(self):
        """Test config creation from URL without password."""
        url = "postgresql://user@localhost:5432/database"
        config = ConnectionFactory.from_url(url)

        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "database"
        assert config.user == "user"
        assert config.password == ""

    def test_from_url_with_query_params(self):
        """Test config creation from URL with query parameters."""
        url = f"postgresql://user:{URL_PASSWORD}@localhost:5432/database?min_pool_size=5&max_pool_size=20"
        config = ConnectionFactory.from_url(url)

        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "database"
        assert config.user == "user"
        assert config.password == URL_PASSWORD
        assert config.min_pool_size == 5
        assert config.max_pool_size == 20

    def test_from_url_invalid_scheme(self):
        """Test config creation from URL with invalid scheme."""
        url = f"mysql://user:{URL_PASSWORD}@localhost:3306/database"

        with pytest.raises(ConfigurationError, match="URL must start with 'postgresql://'"):
            ConnectionFactory.from_url(url)

    def test_from_url_malformed_url(self):
        """Test config creation from malformed URL."""
        url = "postgresql://invalid_url_format"

        with pytest.raises(ConfigurationError, match="Invalid database URL format"):
            ConnectionFactory.from_url(url)

    def test_from_url_missing_components(self):
        """Test config creation from URL missing required components."""
        url = "postgresql://localhost:5432/"

        with pytest.raises(ConfigurationError, match="Invalid database URL format"):
            ConnectionFactory.from_url(url)

    @patch("src.infrastructure.database.connection.AsyncConnectionPool")
    async def test_create_connection_pool_success(self, mock_pool_class, valid_config):
        """Test successful connection pool creation."""
        mock_pool = AsyncMock()
        mock_pool_class.return_value = mock_pool

        pool = await ConnectionFactory.create_connection_pool(valid_config)

        assert pool == mock_pool
        mock_pool_class.assert_called_once()
        mock_pool.open.assert_called_once()

    @patch("src.infrastructure.database.connection.AsyncConnectionPool")
    async def test_create_connection_pool_failure(self, mock_pool_class, valid_config):
        """Test connection pool creation failure."""
        mock_pool_class.side_effect = Exception("Pool creation failed")

        with pytest.raises(ConnectionError, match="Failed to create connection pool"):
            await ConnectionFactory.create_connection_pool(valid_config)

    @patch("src.infrastructure.database.connection.PostgreSQLAdapter")
    @patch("src.infrastructure.database.connection.AsyncConnectionPool")
    async def test_create_connection_success(
        self, mock_pool_class, mock_adapter_class, valid_config
    ):
        """Test successful connection creation."""
        mock_pool = AsyncMock()
        mock_adapter = Mock()
        mock_pool_class.return_value = mock_pool
        mock_adapter_class.return_value = mock_adapter

        connection = await ConnectionFactory.create_connection(valid_config)

        assert connection == mock_adapter
        mock_pool_class.assert_called_once()
        mock_pool.open.assert_called_once()
        mock_adapter_class.assert_called_once_with(mock_pool)


@pytest.mark.unit
class TestDatabaseConnection:
    """Test DatabaseConnection functionality."""

    @pytest.fixture
    def manager(self, valid_config):
        return DatabaseConnection(valid_config)

    def test_manager_initialization(self, manager, valid_config):
        """Test manager initialization."""
        assert manager.config == valid_config
        assert manager._pool is None
        assert manager._adapter is None

    @patch("src.infrastructure.database.connection.ConnectionFactory.create_connection_pool")
    async def test_connect_success(self, mock_create_pool, manager):
        """Test successful connection."""
        mock_pool = AsyncMock()
        mock_create_pool.return_value = mock_pool

        await manager.connect()

        assert manager._pool == mock_pool
        assert manager._adapter is not None
        mock_create_pool.assert_called_once_with(manager.config)

    @patch("src.infrastructure.database.connection.ConnectionFactory.create_connection_pool")
    async def test_connect_failure(self, mock_create_pool, manager):
        """Test connection failure."""
        mock_create_pool.side_effect = Exception("Connection failed")

        with pytest.raises(ConnectionError, match="Failed to connect to database"):
            await manager.connect()

    @patch("src.infrastructure.database.connection.ConnectionFactory.create_connection_pool")
    async def test_connect_already_connected(self, mock_create_pool, manager):
        """Test connecting when already connected."""
        mock_pool = AsyncMock()
        mock_create_pool.return_value = mock_pool

        # First connection
        await manager.connect()
        assert manager._pool == mock_pool

        # Second connection attempt should not create new pool
        await manager.connect()
        mock_create_pool.assert_called_once()  # Should only be called once

    async def test_disconnect_when_connected(self, manager):
        """Test disconnection when connected."""
        mock_pool = AsyncMock()
        manager._pool = mock_pool
        manager._adapter = Mock()

        await manager.disconnect()

        assert manager._pool is None
        assert manager._adapter is None
        mock_pool.close.assert_called_once()

    async def test_disconnect_when_not_connected(self, manager):
        """Test disconnection when not connected."""
        # Should not raise error
        await manager.disconnect()

        assert manager._pool is None
        assert manager._adapter is None

    async def test_disconnect_pool_close_error(self, manager):
        """Test disconnection when pool close fails."""
        mock_pool = AsyncMock()
        mock_pool.close.side_effect = Exception("Close failed")
        manager._pool = mock_pool
        manager._adapter = Mock()

        # Should not raise error, just log warning
        await manager.disconnect()

        assert manager._pool is None
        assert manager._adapter is None

    def test_is_connected_true(self, manager):
        """Test is_connected when connected."""
        manager._pool = Mock()
        manager._adapter = Mock()

        assert manager.is_connected

    def test_is_connected_false(self, manager):
        """Test is_connected when not connected."""
        assert not manager.is_connected

    def test_adapter_property_when_connected(self, manager):
        """Test adapter property when connected."""
        mock_adapter = Mock()
        manager._adapter = mock_adapter

        assert manager.adapter == mock_adapter

    def test_adapter_property_when_not_connected(self, manager):
        """Test adapter property when not connected."""
        with pytest.raises(ConnectionError, match="Not connected to database"):
            _ = manager.adapter

    def test_pool_property_when_connected(self, manager):
        """Test pool property when connected."""
        mock_pool = Mock()
        manager._pool = mock_pool

        assert manager.pool == mock_pool

    def test_pool_property_when_not_connected(self, manager):
        """Test pool property when not connected."""
        with pytest.raises(ConnectionError, match="Not connected to database"):
            _ = manager.pool

    @patch("src.infrastructure.database.connection.ConnectionFactory.create_connection_pool")
    async def test_health_check_success(self, mock_create_pool, manager):
        """Test successful health check."""
        mock_pool = AsyncMock()
        mock_adapter = AsyncMock()
        mock_adapter.health_check.return_value = True
        mock_create_pool.return_value = mock_pool

        await manager.connect()
        manager._adapter = mock_adapter

        result = await manager.health_check()

        assert result is True
        mock_adapter.health_check.assert_called_once()

    async def test_health_check_when_not_connected(self, manager):
        """Test health check when not connected."""
        result = await manager.health_check()

        assert result is False

    @patch("src.infrastructure.database.connection.ConnectionFactory.create_connection_pool")
    async def test_health_check_adapter_failure(self, mock_create_pool, manager):
        """Test health check when adapter check fails."""
        mock_pool = AsyncMock()
        mock_adapter = AsyncMock()
        mock_adapter.health_check.return_value = False
        mock_create_pool.return_value = mock_pool

        await manager.connect()
        manager._adapter = mock_adapter

        result = await manager.health_check()

        assert result is False

    @patch("src.infrastructure.database.connection.ConnectionFactory.create_connection_pool")
    async def test_get_stats(self, mock_create_pool, manager):
        """Test getting connection statistics."""
        mock_pool = AsyncMock()
        mock_adapter = AsyncMock()
        mock_adapter.get_connection_info.return_value = {
            "max_size": 10,
            "min_size": 1,
            "pool_status": "active",
        }
        mock_create_pool.return_value = mock_pool

        await manager.connect()
        manager._adapter = mock_adapter

        stats = await manager.get_stats()

        expected = {
            "is_connected": True,
            "config": manager.config,
            "pool_info": {"max_size": 10, "min_size": 1, "pool_status": "active"},
        }
        assert stats == expected

    async def test_get_stats_when_not_connected(self, manager):
        """Test getting statistics when not connected."""
        stats = await manager.get_stats()

        expected = {"is_connected": False, "config": manager.config, "pool_info": None}
        assert stats == expected

    async def test_context_manager_success(self, manager):
        """Test context manager successful usage."""
        with patch(
            "src.infrastructure.database.connection.ConnectionFactory.create_connection_pool"
        ) as mock_create_pool:
            mock_pool = AsyncMock()
            mock_create_pool.return_value = mock_pool

            async with manager as ctx_manager:
                assert ctx_manager is manager
                assert manager.is_connected

            assert not manager.is_connected
            mock_pool.close.assert_called_once()

    async def test_context_manager_exception(self, manager):
        """Test context manager with exception."""
        with patch(
            "src.infrastructure.database.connection.ConnectionFactory.create_connection_pool"
        ) as mock_create_pool:
            mock_pool = AsyncMock()
            mock_create_pool.return_value = mock_pool

            try:
                async with manager:
                    assert manager.is_connected
                    raise ValueError("Test exception")
            except ValueError:
                pass

            assert not manager.is_connected
            mock_pool.close.assert_called_once()
