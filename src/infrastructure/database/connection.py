"""
Database Connection Management

Provides connection factory and lifecycle management for PostgreSQL connections.
Implements connection pooling, configuration, and health monitoring.
"""

# Standard library imports
import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
import logging
import os
from typing import Any, Optional
from urllib.parse import urlparse

# Third-party imports
import psycopg
from psycopg_pool import AsyncConnectionPool

# Local imports
from src.application.interfaces.exceptions import ConnectionError, FactoryError

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration parameters."""

    host: str = "localhost"
    port: int = 5432
    database: str = "ai_trader"
    user: str = "zachwade"
    password: str | None = None

    # Pool configuration
    min_pool_size: int = 5
    max_pool_size: int = 20
    max_idle_time: float = 300.0  # 5 minutes
    max_lifetime: float = 3600.0  # 1 hour
    command_timeout: float = 60.0
    server_connection_timeout: float = 30.0

    # SSL configuration
    ssl_mode: str = "prefer"
    ssl_cert_file: str | None = None
    ssl_key_file: str | None = None
    ssl_ca_file: str | None = None

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """
        Create configuration from environment variables.

        Returns:
            DatabaseConfig instance
        """
        return cls(
            host=os.getenv("DATABASE_HOST", "localhost"),
            port=int(os.getenv("DATABASE_PORT", "5432")),
            database=os.getenv("DATABASE_NAME", "ai_trader"),
            user=os.getenv("DATABASE_USER", "zachwade"),
            password=os.getenv("DATABASE_PASSWORD"),
            min_pool_size=int(os.getenv("DATABASE_MIN_POOL_SIZE", "5")),
            max_pool_size=int(os.getenv("DATABASE_MAX_POOL_SIZE", "20")),
            max_idle_time=float(os.getenv("DATABASE_MAX_IDLE_TIME", "300.0")),
            max_lifetime=float(os.getenv("DATABASE_MAX_LIFETIME", "3600.0")),
            command_timeout=float(os.getenv("DATABASE_COMMAND_TIMEOUT", "60.0")),
            server_connection_timeout=float(os.getenv("DATABASE_SERVER_TIMEOUT", "30.0")),
            ssl_mode=os.getenv("DATABASE_SSL_MODE", "prefer"),
            ssl_cert_file=os.getenv("DATABASE_SSL_CERT_FILE"),
            ssl_key_file=os.getenv("DATABASE_SSL_KEY_FILE"),
            ssl_ca_file=os.getenv("DATABASE_SSL_CA_FILE"),
        )

    @classmethod
    def from_url(cls, url: str) -> "DatabaseConfig":
        """
        Create configuration from database URL.

        Args:
            url: Database URL (e.g., postgresql://user:pass@host:port/db)

        Returns:
            DatabaseConfig instance
        """
        parsed = urlparse(url)

        if parsed.scheme not in ("postgresql", "postgres"):
            raise ValueError(f"Invalid database URL scheme: {parsed.scheme}")

        return cls(
            host=parsed.hostname or "localhost",
            port=parsed.port or 5432,
            database=parsed.path.lstrip("/") if parsed.path else "ai_trader",
            user=parsed.username or "zachwade",
            password=parsed.password,
        )

    def build_dsn(self) -> str:
        """
        Build database DSN string.

        Returns:
            DSN string for psycopg3
        """
        dsn_parts = [
            f"postgresql://{self.user}",
        ]

        if self.password:
            dsn_parts[0] += f":{self.password}"

        dsn_parts[0] += f"@{self.host}:{self.port}/{self.database}"

        # Add SSL parameters if configured
        params = []
        if self.ssl_mode != "prefer":
            params.append(f"sslmode={self.ssl_mode}")
        if self.ssl_cert_file:
            params.append(f"sslcert={self.ssl_cert_file}")
        if self.ssl_key_file:
            params.append(f"sslkey={self.ssl_key_file}")
        if self.ssl_ca_file:
            params.append(f"sslrootcert={self.ssl_ca_file}")

        if params:
            dsn_parts.append("?" + "&".join(params))

        return "".join(dsn_parts)


class DatabaseConnection:
    """
    Database connection manager with health monitoring.

    Manages a single psycopg3 connection pool and provides
    health checking and lifecycle management.
    """

    def __init__(self, config: DatabaseConfig) -> None:
        """
        Initialize connection manager.

        Args:
            config: Database configuration
        """
        self.config = config
        self._pool: AsyncConnectionPool | None = None
        self._health_check_task: asyncio.Task | None = None
        self._is_closed = False

    @property
    def is_connected(self) -> bool:
        """Check if connection pool is active."""
        return self._pool is not None and not self._pool.closed

    @property
    def is_closed(self) -> bool:
        """Check if connection has been closed."""
        return self._is_closed

    async def connect(self) -> AsyncConnectionPool:
        """
        Establish database connection pool.

        Returns:
            psycopg3 async connection pool

        Raises:
            ConnectionError: If connection fails
        """
        if self._is_closed:
            raise ConnectionError("Connection manager has been closed")

        if self.is_connected:
            return self._pool

        try:
            logger.info(
                f"Connecting to database: {self.config.host}:{self.config.port}/{self.config.database}"
            )

            # Create connection pool
            self._pool = AsyncConnectionPool(
                conninfo=self.config.build_dsn(),
                min_size=self.config.min_pool_size,
                max_size=self.config.max_pool_size,
                max_idle=self.config.max_idle_time,
                max_lifetime=self.config.max_lifetime,
                timeout=self.config.command_timeout,
                # psycopg3 doesn't have server_settings in the same way
                # these can be set via the connection string instead
            )

            # Open the pool
            await self._pool.open()

            # Test connection
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute("SELECT 1")
                await cur.fetchone()

            logger.info(f"Database connected successfully. Pool max size: {self._pool.max_size}")

            # Start health monitoring
            self._start_health_monitoring()

            return self._pool

        except (TimeoutError, psycopg.OperationalError, OSError) as e:
            logger.error(f"Failed to connect to database: {e}")
            await self._cleanup()
            raise ConnectionError(f"Failed to connect to database: {e}") from e

    async def disconnect(self) -> None:
        """
        Close database connection pool.
        """
        if self._is_closed:
            return

        logger.info("Disconnecting from database...")
        await self._cleanup()
        self._is_closed = True
        logger.info("Database disconnected")

    async def _cleanup(self) -> None:
        """Clean up resources."""
        # Stop health monitoring
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._health_check_task

        # Close connection pool
        if self._pool and not self._pool.closed:
            await self._pool.close()

        self._pool = None
        self._health_check_task = None

    def _start_health_monitoring(self) -> None:
        """Start background health monitoring task."""
        if self._health_check_task and not self._health_check_task.done():
            return

        self._health_check_task = asyncio.create_task(self._health_monitor_loop())

    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop."""
        while not self._is_closed and self.is_connected:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                if not self.is_connected:
                    break

                # Simple health check
                async with self._pool.connection() as conn, conn.cursor() as cur:
                    await cur.execute("SELECT 1")
                    await cur.fetchone()

                logger.debug("Database health check passed")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Database health check failed: {e}")
                # Could trigger reconnection logic here if needed

    async def get_pool_stats(self) -> dict[str, Any]:
        """
        Get connection pool statistics.

        Returns:
            Dictionary with pool statistics
        """
        if not self.is_connected:
            return {"status": "disconnected"}

        return {
            "status": "connected",
            "max_size": self._pool.max_size,
            "min_size": self._pool.min_size,
            "is_closed": self._pool.closed,
        }

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[psycopg.AsyncConnection, None]:
        """
        Acquire a connection from the pool.

        Yields:
            Database connection

        Raises:
            ConnectionError: If no pool is available
        """
        if not self.is_connected:
            raise ConnectionError("Database is not connected")

        async with self._pool.connection() as connection:
            yield connection

    def __str__(self) -> str:
        """String representation."""
        status = "connected" if self.is_connected else "disconnected"
        return f"DatabaseConnection({self.config.host}:{self.config.port}, {status})"


class ConnectionFactory:
    """
    Factory for creating and managing database connections.

    Implements singleton pattern for connection sharing and
    provides configuration-based connection creation.
    """

    _instance: Optional["ConnectionFactory"] = None
    _connection: DatabaseConnection | None = None

    def __new__(cls) -> "ConnectionFactory":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    async def create_connection(
        cls,
        config: DatabaseConfig | None = None,
        force_new: bool = False,
    ) -> DatabaseConnection:
        """
        Create or get database connection.

        Args:
            config: Database configuration (defaults to environment)
            force_new: Force creation of new connection

        Returns:
            DatabaseConnection instance

        Raises:
            FactoryError: If connection cannot be created
        """
        instance = cls()

        if force_new or instance._connection is None or instance._connection.is_closed:
            if instance._connection and not instance._connection.is_closed:
                await instance._connection.disconnect()

            try:
                config = config or DatabaseConfig.from_env()
                instance._connection = DatabaseConnection(config)
                await instance._connection.connect()

                logger.info(f"Created new database connection: {instance._connection}")

            except Exception as e:
                logger.error(f"Failed to create database connection: {e}")
                raise FactoryError("ConnectionFactory", f"Failed to create connection: {e}") from e

        return instance._connection

    @classmethod
    async def get_connection(cls) -> DatabaseConnection:
        """
        Get existing database connection.

        Returns:
            DatabaseConnection instance

        Raises:
            FactoryError: If no connection exists
        """
        instance = cls()

        if instance._connection is None or instance._connection.is_closed:
            raise FactoryError("ConnectionFactory", "No active database connection")

        return instance._connection

    @classmethod
    async def close_all(cls) -> None:
        """Close all connections managed by the factory."""
        instance = cls()

        if instance._connection and not instance._connection.is_closed:
            await instance._connection.disconnect()
            instance._connection = None
            logger.info("All database connections closed")

    @classmethod
    def reset(cls) -> None:
        """Reset the factory (for testing)."""
        cls._instance = None
        cls._connection = None
