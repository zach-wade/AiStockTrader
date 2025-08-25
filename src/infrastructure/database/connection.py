"""
Database Connection Management

Provides connection factory and lifecycle management for PostgreSQL connections.
Implements connection pooling, configuration, and health monitoring.
"""

# Standard library imports
import asyncio
import logging
import os
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import urlparse

# Third-party imports
import psycopg
from psycopg_pool import AsyncConnectionPool

# Local imports
from src.application.interfaces.exceptions import ConnectionError, FactoryError
from src.infrastructure.config import DSNBuilder

# Optional monitoring imports
if TYPE_CHECKING:
    from src.infrastructure.monitoring.metrics import MetricType, TradingMetrics

try:
    from src.infrastructure.monitoring.metrics import MetricType, TradingMetrics

    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    TradingMetrics = None  # type: ignore[misc,assignment]
    MetricType = None  # type: ignore[misc,assignment]

from src.infrastructure.resilience.retry import ExponentialBackoff, RetryConfig

logger = logging.getLogger(__name__)


@dataclass
class ConnectionPoolMetrics:
    """Metrics for connection pool monitoring."""

    active_connections: int = 0
    idle_connections: int = 0
    waiting_requests: int = 0
    total_connections_created: int = 0
    total_connections_closed: int = 0
    connection_errors: int = 0
    connection_timeouts: int = 0
    avg_connection_acquisition_time: float = 0.0
    max_connection_acquisition_time: float = 0.0
    health_check_failures: int = 0
    last_health_check: datetime | None = None
    pool_exhausted_count: int = 0


@dataclass
class DatabaseConfig:
    """Database configuration parameters."""

    host: str = "localhost"
    port: int = 5432
    database: str = "ai_trader"
    user: str = ""  # Must be provided via environment or secrets
    password: str | None = None

    # Pool configuration - Increased for high throughput
    min_pool_size: int = 10
    max_pool_size: int = 100  # Increased from 20 to handle 1000 orders/sec
    max_idle_time: float = 300.0  # 5 minutes
    max_lifetime: float = 3600.0  # 1 hour
    command_timeout: float = 60.0
    server_connection_timeout: float = 30.0

    # Connection validation
    validate_on_checkout: bool = True  # Validate connection before use
    validation_query: str = "SELECT 1"  # Query to validate connection
    max_validation_failures: int = 3  # Max validation failures before discarding connection

    # Retry configuration
    max_retry_attempts: int = 5
    initial_retry_delay: float = 0.5  # seconds
    max_retry_delay: float = 30.0  # seconds
    retry_backoff_multiplier: float = 2.0
    retry_jitter: bool = True

    # Health monitoring
    health_check_interval: float = 10.0  # seconds - More frequent for high throughput
    health_check_timeout: float = 5.0  # seconds
    enable_pool_metrics: bool = True
    metrics_collection_interval: float = 5.0  # seconds

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
        from src.infrastructure.security import SecretsManager

        secrets_manager = SecretsManager.get_instance()
        db_config = secrets_manager.get_database_config()

        return cls(
            host=os.getenv("DATABASE_HOST", db_config.get("host", "localhost")),
            port=int(os.getenv("DATABASE_PORT", db_config.get("port", "5432"))),
            database=os.getenv("DATABASE_NAME", db_config.get("database", "ai_trader")),
            user=os.getenv("DATABASE_USER") or db_config.get("user", "postgres"),
            password=os.getenv("DATABASE_PASSWORD") or db_config.get("password"),
            min_pool_size=int(os.getenv("DATABASE_MIN_POOL_SIZE", "5")),
            max_pool_size=int(os.getenv("DATABASE_MAX_POOL_SIZE", "100")),
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
            user=parsed.username or "",  # No default user - must be provided
            password=parsed.password,
        )

    def build_dsn(self) -> str:
        """
        Build database DSN string using DSNBuilder.

        Returns:
            DSN string for psycopg3
        """
        # Use DSNBuilder for clean separation of concerns
        builder = DSNBuilder()
        return (
            builder.with_host(self.host)
            .with_port(self.port)
            .with_database(self.database)
            .with_credentials(self.user, self.password)
            .with_ssl(self.ssl_mode, self.ssl_cert_file, self.ssl_key_file, self.ssl_ca_file)
            .build()
        )


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
        self._health_check_task: asyncio.Task[None] | None = None
        self._metrics_task: asyncio.Task[None] | None = None
        self._is_closed = False

        # Metrics tracking
        self._metrics = ConnectionPoolMetrics()
        self._connection_acquisition_times: list[float] = []
        self._metrics_lock = asyncio.Lock()

        # Retry configuration
        self._retry_config = RetryConfig(
            max_retries=config.max_retry_attempts,
            initial_delay=config.initial_retry_delay,
            max_delay=config.max_retry_delay,
            backoff_multiplier=config.retry_backoff_multiplier,
            jitter=config.retry_jitter,
            retryable_exceptions=(ConnectionError, psycopg.OperationalError, TimeoutError, OSError),
        )
        self._backoff = ExponentialBackoff(self._retry_config)

        # Metrics collector (if available)
        self._metrics_collector: Any | None = None
        try:
            if METRICS_AVAILABLE and TradingMetrics is not None:
                self._metrics_collector = TradingMetrics.get_instance()  # type: ignore[attr-defined]
                self._setup_metrics()
        except Exception:
            self._metrics_collector = None
            logger.warning("Metrics collector not available, proceeding without metrics")

    @property
    def is_connected(self) -> bool:
        """Check if connection pool is active."""
        return self._pool is not None and not self._pool.closed

    @property
    def is_closed(self) -> bool:
        """Check if connection has been closed."""
        return self._is_closed

    def _setup_metrics(self) -> None:
        """Set up metrics collection."""
        if not self._metrics_collector:
            return

        # Register database connection metrics
        self._metrics_collector.register_metric(
            name="database.pool.active_connections",
            metric_type=MetricType.GAUGE,
            description="Number of active database connections",
        )
        self._metrics_collector.register_metric(
            name="database.pool.idle_connections",
            metric_type=MetricType.GAUGE,
            description="Number of idle database connections",
        )
        self._metrics_collector.register_metric(
            name="database.pool.waiting_requests",
            metric_type=MetricType.GAUGE,
            description="Number of requests waiting for a connection",
        )
        self._metrics_collector.register_metric(
            name="database.pool.connection_errors",
            metric_type=MetricType.COUNTER,
            description="Total number of connection errors",
        )
        self._metrics_collector.register_metric(
            name="database.pool.exhausted_count",
            metric_type=MetricType.COUNTER,
            description="Number of times the pool was exhausted",
        )
        self._metrics_collector.register_metric(
            name="database.connection.acquisition_time",
            metric_type=MetricType.HISTOGRAM,
            description="Time to acquire a connection from pool (ms)",
        )
        self._metrics_collector.register_metric(
            name="database.health_check.failures",
            metric_type=MetricType.COUNTER,
            description="Number of health check failures",
        )

    async def connect(self) -> AsyncConnectionPool:
        """
        Establish database connection pool with retry logic.

        Returns:
            psycopg3 async connection pool

        Raises:
            ConnectionError: If connection fails after all retries
        """
        if self._is_closed:
            raise ConnectionError("Connection manager has been closed")

        if self.is_connected and self._pool is not None:
            return self._pool

        # Implement exponential backoff retry logic
        last_exception: Exception | None = None
        start_time = time.time()

        for attempt in range(self.config.max_retry_attempts):
            try:
                logger.info(
                    f"Connecting to database (attempt {attempt + 1}/{self.config.max_retry_attempts}): "
                    f"{self.config.host}:{self.config.port}/{self.config.database}"
                )

                # Create connection pool
                self._pool = AsyncConnectionPool(
                    conninfo=self.config.build_dsn(),
                    min_size=self.config.min_pool_size,
                    max_size=self.config.max_pool_size,
                    max_idle=self.config.max_idle_time,
                    max_lifetime=self.config.max_lifetime,
                    timeout=self.config.command_timeout,
                )

                # Open the pool with timeout
                await asyncio.wait_for(
                    self._pool.open(), timeout=self.config.server_connection_timeout
                )

                # Test connection
                async with self._pool.connection() as conn, conn.cursor() as cur:
                    await cur.execute("SELECT 1")
                    await cur.fetchone()

                logger.info(
                    f"Database connected successfully. Pool size: {self.config.min_pool_size}-{self.config.max_pool_size}"
                )

                # Record successful connection
                self._metrics.total_connections_created += 1

                # Start monitoring tasks
                self._start_health_monitoring()
                if self.config.enable_pool_metrics:
                    self._start_metrics_collection()

                return self._pool

            except (TimeoutError, psycopg.OperationalError, OSError) as e:
                last_exception = e
                self._metrics.connection_errors += 1

                if attempt < self.config.max_retry_attempts - 1:
                    delay = self._backoff.get_delay(attempt)
                    logger.warning(
                        f"Connection attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    await asyncio.sleep(delay)

                    # Clean up failed pool
                    if self._pool:
                        try:
                            await self._pool.close()
                        except Exception:
                            pass
                        self._pool = None
                else:
                    total_time = time.time() - start_time
                    logger.error(
                        f"Failed to connect to database after {attempt + 1} attempts "
                        f"in {total_time:.2f} seconds: {e}"
                    )

                    # Update metrics if available
                    # Metrics would be recorded here

                    await self._cleanup()
                    raise ConnectionError(
                        f"Failed to connect to database after {attempt + 1} attempts: {e}"
                    ) from e

        # This shouldn't be reached, but handle it just in case
        raise ConnectionError("Failed to establish database connection")

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

        # Stop metrics collection
        if self._metrics_task and not self._metrics_task.done():
            self._metrics_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._metrics_task

        # Close connection pool
        if self._pool and not self._pool.closed:
            await self._pool.close()
            self._metrics.total_connections_closed += 1

        self._pool = None
        self._health_check_task = None
        self._metrics_task = None

    def _start_health_monitoring(self) -> None:
        """Start background health monitoring task."""
        if self._health_check_task and not self._health_check_task.done():
            return

        self._health_check_task = asyncio.create_task(self._health_monitor_loop())

    def _start_metrics_collection(self) -> None:
        """Start background metrics collection task."""
        if self._metrics_task and not self._metrics_task.done():
            return

        self._metrics_task = asyncio.create_task(self._metrics_collection_loop())

    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop with connection validation."""
        consecutive_failures = 0

        while not self._is_closed and self.is_connected:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                # Validate pool availability
                if not self.is_connected or self._pool is None:
                    logger.warning("Pool disconnected during health check")
                    break

                # Perform health check with timeout
                start_time = time.time()
                try:
                    async with asyncio.timeout(self.config.health_check_timeout):
                        async with self._pool.connection() as conn, conn.cursor() as cur:
                            await cur.execute(self.config.validation_query)
                            await cur.fetchone()

                    check_time = (time.time() - start_time) * 1000  # ms
                    self._metrics.last_health_check = datetime.now()
                    consecutive_failures = 0

                    logger.debug(f"Database health check passed in {check_time:.2f}ms")

                    # Update metrics
                    if self._metrics_collector:
                        self._metrics_collector.record_value(
                            "database.health_check.duration", check_time
                        )

                except TimeoutError:
                    consecutive_failures += 1
                    self._metrics.health_check_failures += 1
                    logger.error(
                        f"Health check timeout (consecutive failures: {consecutive_failures})"
                    )

                    if self._metrics_collector:
                        self._metrics_collector.record_value("database.health_check.failures", 1)

                    # Consider pool unhealthy after multiple failures
                    if consecutive_failures >= 3:
                        logger.critical("Database pool appears unhealthy, may need reconnection")
                        # Could trigger automatic reconnection here

            except asyncio.CancelledError:
                break
            except Exception as e:
                consecutive_failures += 1
                self._metrics.health_check_failures += 1
                logger.warning(f"Database health check failed: {e}")

                if self._metrics_collector:
                    self._metrics_collector.record_value("database.health_check.failures", 1)

    async def _metrics_collection_loop(self) -> None:
        """Collect and publish pool metrics."""
        while not self._is_closed and self.is_connected:
            try:
                await asyncio.sleep(self.config.metrics_collection_interval)

                if not self._pool or self._pool.closed:
                    continue

                # Collect pool statistics
                stats = await self._collect_pool_stats()

                # Update internal metrics
                async with self._metrics_lock:
                    self._metrics.active_connections = stats.get("active", 0)
                    self._metrics.idle_connections = stats.get("idle", 0)
                    self._metrics.waiting_requests = stats.get("waiting", 0)

                    # Calculate average acquisition time
                    if self._connection_acquisition_times:
                        self._metrics.avg_connection_acquisition_time = sum(
                            self._connection_acquisition_times
                        ) / len(self._connection_acquisition_times)
                        self._metrics.max_connection_acquisition_time = max(
                            self._connection_acquisition_times
                        )
                        # Keep only last 100 measurements
                        self._connection_acquisition_times = self._connection_acquisition_times[
                            -100:
                        ]

                # Publish to metrics collector
                if self._metrics_collector:
                    self._metrics_collector.record_value(
                        "database.pool.active_connections", self._metrics.active_connections
                    )
                    self._metrics_collector.record_value(
                        "database.pool.idle_connections", self._metrics.idle_connections
                    )
                    self._metrics_collector.record_value(
                        "database.pool.waiting_requests", self._metrics.waiting_requests
                    )

                    # Check for pool exhaustion
                    if self._metrics.active_connections >= self.config.max_pool_size * 0.9:
                        logger.warning(
                            f"Connection pool near exhaustion: {self._metrics.active_connections}/{self.config.max_pool_size}"
                        )
                        self._metrics.pool_exhausted_count += 1
                        self._metrics_collector.record_value("database.pool.exhausted_count", 1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error collecting pool metrics: {e}")

    async def _collect_pool_stats(self) -> dict[str, Any]:
        """Collect current pool statistics."""
        if not self._pool:
            return {}

        try:
            # psycopg3 pool doesn't expose detailed stats directly
            # We'll track what we can
            return {
                "active": self._pool._nconns if hasattr(self._pool, "_nconns") else 0,
                "idle": 0,  # Would need custom tracking
                "waiting": self._pool._waiting if hasattr(self._pool, "_waiting") else 0,
                "max_size": self._pool.max_size,
                "min_size": self._pool.min_size,
                "closed": self._pool.closed,
            }
        except Exception as e:
            logger.error(f"Failed to collect pool stats: {e}")
            return {}

    async def get_pool_stats(self) -> dict[str, Any]:
        """
        Get comprehensive connection pool statistics.

        Returns:
            Dictionary with pool statistics and metrics
        """
        if not self.is_connected:
            return {"status": "disconnected"}

        if self._pool is None:
            return {"status": "disconnected"}

        # Collect current stats
        current_stats = await self._collect_pool_stats()

        # Combine with accumulated metrics
        async with self._metrics_lock:
            return {
                "status": "connected",
                "pool_config": {
                    "max_size": self._pool.max_size,
                    "min_size": self._pool.min_size,
                    "max_idle_time": self.config.max_idle_time,
                    "max_lifetime": self.config.max_lifetime,
                },
                "current_state": {
                    "active_connections": self._metrics.active_connections,
                    "idle_connections": self._metrics.idle_connections,
                    "waiting_requests": self._metrics.waiting_requests,
                    "is_closed": self._pool.closed,
                },
                "performance_metrics": {
                    "avg_acquisition_time_ms": self._metrics.avg_connection_acquisition_time,
                    "max_acquisition_time_ms": self._metrics.max_connection_acquisition_time,
                    "total_connections_created": self._metrics.total_connections_created,
                    "total_connections_closed": self._metrics.total_connections_closed,
                },
                "health_metrics": {
                    "connection_errors": self._metrics.connection_errors,
                    "connection_timeouts": self._metrics.connection_timeouts,
                    "health_check_failures": self._metrics.health_check_failures,
                    "last_health_check": (
                        self._metrics.last_health_check.isoformat()
                        if self._metrics.last_health_check
                        else None
                    ),
                    "pool_exhausted_count": self._metrics.pool_exhausted_count,
                },
                "retry_config": {
                    "max_retries": self.config.max_retry_attempts,
                    "backoff_multiplier": self.config.retry_backoff_multiplier,
                },
            }

    async def _validate_connection(self, conn: psycopg.AsyncConnection) -> bool:
        """
        Validate a connection before use.

        Args:
            conn: Connection to validate

        Returns:
            True if connection is valid, False otherwise
        """
        if not self.config.validate_on_checkout:
            return True

        try:
            async with asyncio.timeout(self.config.health_check_timeout):
                async with conn.cursor() as cur:
                    await cur.execute(self.config.validation_query)
                    result = await cur.fetchone()
                    return result is not None
        except (TimeoutError, psycopg.Error) as e:
            logger.warning(f"Connection validation failed: {e}")
            return False

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[psycopg.AsyncConnection, None]:
        """
        Acquire a validated connection from the pool with retry logic.

        Yields:
            Database connection

        Raises:
            ConnectionError: If no pool is available or connection cannot be acquired
        """
        if not self.is_connected:
            raise ConnectionError("Database is not connected")

        if self._pool is None:
            raise ConnectionError("Database pool is not initialized")

        acquisition_start = time.time()
        validation_failures = 0
        last_exception: Exception | None = None

        # Retry acquiring a valid connection
        for attempt in range(self.config.max_retry_attempts):
            try:
                # Try to acquire connection with timeout
                async with asyncio.timeout(self.config.command_timeout):
                    async with self._pool.connection() as connection:
                        # Validate connection if enabled
                        if self.config.validate_on_checkout:
                            if not await self._validate_connection(connection):
                                validation_failures += 1
                                if validation_failures >= self.config.max_validation_failures:
                                    raise ConnectionError(
                                        f"Connection validation failed {validation_failures} times"
                                    )
                                # Try again with a different connection
                                continue

                        # Track acquisition time
                        acquisition_time = (time.time() - acquisition_start) * 1000  # ms
                        async with self._metrics_lock:
                            self._connection_acquisition_times.append(acquisition_time)

                        # Update metrics
                        if self._metrics_collector:
                            self._metrics_collector.record_value(
                                "database.connection.acquisition_time", acquisition_time
                            )

                        logger.debug(f"Connection acquired in {acquisition_time:.2f}ms")
                        yield connection
                        return

            except TimeoutError as e:
                last_exception = e
                self._metrics.connection_timeouts += 1

                if attempt < self.config.max_retry_attempts - 1:
                    delay = self._backoff.get_delay(attempt)
                    logger.warning(
                        f"Connection acquisition timeout (attempt {attempt + 1}). "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    await asyncio.sleep(delay)
                else:
                    # Metrics would be recorded here
                    raise ConnectionError(
                        f"Failed to acquire connection after {attempt + 1} attempts"
                    ) from e

            except (psycopg.OperationalError, psycopg.Error) as e:
                last_exception = e
                self._metrics.connection_errors += 1

                if attempt < self.config.max_retry_attempts - 1:
                    delay = self._backoff.get_delay(attempt)
                    logger.warning(
                        f"Connection error (attempt {attempt + 1}): {e}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    await asyncio.sleep(delay)
                else:
                    # Metrics would be recorded here
                    raise ConnectionError(f"Failed to acquire valid connection: {e}") from e

        # Should not reach here
        raise ConnectionError(
            f"Failed to acquire connection after {self.config.max_retry_attempts} attempts"
        )

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
        Create or get database connection with automatic retry.

        Args:
            config: Database configuration (defaults to environment)
            force_new: Force creation of new connection

        Returns:
            DatabaseConnection instance

        Raises:
            FactoryError: If connection cannot be created after all retries
        """
        instance = cls()

        if force_new or instance._connection is None or instance._connection.is_closed:
            if instance._connection and not instance._connection.is_closed:
                await instance._connection.disconnect()

            try:
                config = config or DatabaseConfig.from_env()
                instance._connection = DatabaseConnection(config)

                # Connect with built-in retry logic
                await instance._connection.connect()

                logger.info(
                    f"Created new database connection: {instance._connection} "
                    f"(pool size: {config.min_pool_size}-{config.max_pool_size})"
                )

            except ConnectionError as e:
                logger.error(f"Failed to create database connection: {e}")
                instance._connection = None
                raise FactoryError(
                    "ConnectionFactory", f"Failed to create connection after retries: {e}"
                ) from e
            except Exception as e:
                logger.error(f"Unexpected error creating database connection: {e}")
                instance._connection = None
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
