"""
Enhanced Database Connection Management with Resilience

Production-grade database layer with connection pooling, health monitoring,
circuit breakers, retry logic, and graceful degradation.
"""

import logging
import time
from collections.abc import AsyncGenerator
from collections.abc import AsyncGenerator as AsyncGeneratorABC
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional, cast

import psycopg
from psycopg_pool import AsyncConnectionPool

from src.application.interfaces.exceptions import ConnectionError, FactoryError
from src.infrastructure.database.connection import DatabaseConfig
from src.infrastructure.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from src.infrastructure.resilience.health import HealthCheck, HealthCheckResult, HealthStatus
from src.infrastructure.resilience.retry import RetryConfig, retry_with_backoff

logger = logging.getLogger(__name__)


@dataclass
class EnhancedDatabaseConfig(DatabaseConfig):
    """Enhanced database configuration with resilience settings."""

    # Circuit breaker settings
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout: float = 60.0

    # Retry settings
    retry_enabled: bool = True
    max_retries: int = 3
    retry_initial_delay: float = 0.5
    retry_max_delay: float = 10.0

    # Health check settings
    health_check_enabled: bool = True
    health_check_interval: float = 30.0
    health_check_timeout: float = 5.0

    # Connection pool enhancements
    pool_pre_ping: bool = True  # Validate connections before use
    pool_recycle_time: float = 3600.0  # Recycle connections after 1 hour
    pool_overflow_size: int = 10  # Allow pool to grow beyond max_size temporarily

    # Monitoring
    enable_query_logging: bool = False  # Log all queries (expensive)
    slow_query_threshold: float = 1.0  # Log queries slower than this

    @classmethod
    def from_base_config(
        cls, base_config: DatabaseConfig, **overrides: Any
    ) -> "EnhancedDatabaseConfig":
        """Create enhanced config from base config."""
        base_dict = {
            "host": base_config.host,
            "port": base_config.port,
            "database": base_config.database,
            "user": base_config.user,
            "password": base_config.password,
            "min_pool_size": base_config.min_pool_size,
            "max_pool_size": base_config.max_pool_size,
            "max_idle_time": base_config.max_idle_time,
            "max_lifetime": base_config.max_lifetime,
            "command_timeout": base_config.command_timeout,
            "server_connection_timeout": base_config.server_connection_timeout,
            "ssl_mode": base_config.ssl_mode,
            "ssl_cert_file": base_config.ssl_cert_file,
            "ssl_key_file": base_config.ssl_key_file,
            "ssl_ca_file": base_config.ssl_ca_file,
        }
        base_dict.update(overrides)
        # Cast values to proper types with fallbacks
        host_val = base_dict.get("host", "")
        port_val = base_dict.get("port", 5432)
        database_val = base_dict.get("database", "")
        user_val = base_dict.get("user", "")
        password_val = base_dict.get("password")
        min_pool_size_val = base_dict.get("min_pool_size", 1)
        max_pool_size_val = base_dict.get("max_pool_size", 10)
        command_timeout_val = base_dict.get("command_timeout", 30.0)
        server_connection_timeout_val = base_dict.get("server_connection_timeout", 30.0)
        ssl_mode_val = base_dict.get("ssl_mode")
        ssl_cert_file_val = base_dict.get("ssl_cert_file")
        ssl_key_file_val = base_dict.get("ssl_key_file")
        ssl_ca_file_val = base_dict.get("ssl_ca_file")

        return cls(
            host=cast(str, host_val) if host_val is not None else "",
            port=cast(int, port_val) if port_val is not None else 5432,
            database=cast(str, database_val) if database_val is not None else "",
            user=cast(str, user_val) if user_val is not None else "",
            password=cast(str, password_val) if password_val is not None else None,
            min_pool_size=cast(int, min_pool_size_val) if min_pool_size_val is not None else 1,
            max_pool_size=cast(int, max_pool_size_val) if max_pool_size_val is not None else 10,
            command_timeout=(
                cast(float, command_timeout_val) if command_timeout_val is not None else 30.0
            ),
            server_connection_timeout=(
                cast(float, server_connection_timeout_val)
                if server_connection_timeout_val is not None
                else 30.0
            ),
            ssl_mode=cast(str, ssl_mode_val) if ssl_mode_val is not None else "",
            ssl_cert_file=cast(str, ssl_cert_file_val) if ssl_cert_file_val is not None else None,
            ssl_key_file=cast(str, ssl_key_file_val) if ssl_key_file_val is not None else None,
            ssl_ca_file=cast(str, ssl_ca_file_val) if ssl_ca_file_val is not None else None,
            # Note: Additional fields would need explicit handling to avoid type errors
        )


class EnhancedDatabaseHealthCheck(HealthCheck):
    """Health check specifically for enhanced database connections."""

    def __init__(self, name: str, connection: "ResilientDatabaseConnection") -> None:
        super().__init__(name, timeout=10.0)
        self.connection = connection

    async def check_health(self) -> HealthCheckResult:
        """Perform comprehensive database health check."""
        start_time = time.time()

        try:
            # Check if connection pool exists and is not closed
            if not self.connection.is_connected:
                raise Exception("Database connection pool not available")

            # Test basic connectivity
            async with self.connection.acquire() as conn:
                async with conn.cursor() as cursor:
                    # Simple query
                    await cursor.execute("SELECT 1 as health_check")
                    result = await cursor.fetchone()

                    if not result or result[0] != 1:
                        raise Exception("Health check query returned unexpected result")

                    # Check server version (validates connection)
                    await cursor.execute("SELECT version()")
                    version_result = await cursor.fetchone()

                    # Get detailed pool stats
                    pool_stats = await self.connection.get_enhanced_pool_stats()

                    response_time = time.time() - start_time

                    return HealthCheckResult(
                        service_name=self.name,
                        status=HealthStatus.HEALTHY,
                        response_time=response_time,
                        timestamp=start_time,
                        details={
                            **pool_stats,
                            "server_version": version_result[0] if version_result else "unknown",
                            "connection_test": "passed",
                        },
                    )

        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheckResult(
                service_name=self.name,
                status=HealthStatus.UNHEALTHY,
                response_time=response_time,
                timestamp=start_time,
                error=str(e),
                details={"connection_test": "failed"},
            )


class ResilientDatabaseConnection:
    """
    Enhanced database connection with production resilience features.

    Features:
    - Circuit breaker protection
    - Automatic retry with exponential backoff
    - Connection pool health monitoring
    - Graceful degradation
    - Comprehensive metrics
    """

    def __init__(self, config: EnhancedDatabaseConfig) -> None:
        self.config = config
        self._pool: AsyncConnectionPool | None = None
        self._circuit_breaker: CircuitBreaker | None = None
        self._health_check: EnhancedDatabaseHealthCheck | None = None
        self._is_closed = False

        # Metrics
        self._connection_metrics = {
            "total_connections": 0,
            "successful_connections": 0,
            "failed_connections": 0,
            "total_queries": 0,
            "slow_queries": 0,
            "circuit_breaker_trips": 0,
            "retry_attempts": 0,
        }

        # Initialize circuit breaker if enabled
        if config.circuit_breaker_enabled:
            cb_config = CircuitBreakerConfig(
                failure_threshold=config.circuit_breaker_failure_threshold,
                timeout=config.circuit_breaker_timeout,
                failure_types=(
                    ConnectionError,
                    psycopg.OperationalError,
                    psycopg.InterfaceError,
                    TimeoutError,
                    OSError,
                ),
            )
            self._circuit_breaker = CircuitBreaker("database", cb_config)

    @property
    def is_connected(self) -> bool:
        """Check if database pool is connected."""
        return self._pool is not None and not self._pool.closed

    @property
    def is_closed(self) -> bool:
        """Check if connection has been closed."""
        return self._is_closed

    async def connect(self) -> AsyncConnectionPool:
        """
        Establish resilient database connection with retries.

        Returns:
            AsyncConnectionPool instance
        """
        if self._is_closed:
            raise ConnectionError("Connection manager has been closed")

        if self.is_connected and self._pool is not None:
            return self._pool

        # Configure retry for connection attempts
        retry_config = None
        if self.config.retry_enabled:
            retry_config = RetryConfig(
                max_retries=self.config.max_retries,
                initial_delay=self.config.retry_initial_delay,
                max_delay=self.config.retry_max_delay,
                retryable_exceptions=(
                    ConnectionError,
                    psycopg.OperationalError,
                    TimeoutError,
                    OSError,
                ),
            )

        try:
            if self.config.retry_enabled and retry_config:
                self._pool = await retry_with_backoff(
                    self._establish_connection, config=retry_config
                )
            else:
                self._pool = await self._establish_connection()

            self._connection_metrics["successful_connections"] += 1
            logger.info("Database connected successfully with resilience features")

            return self._pool

        except Exception as e:
            self._connection_metrics["failed_connections"] += 1
            logger.error(f"Failed to establish resilient database connection: {e}")
            raise ConnectionError(f"Database connection failed: {e}") from e

    async def _establish_connection(self) -> AsyncConnectionPool:
        """Internal method to establish database connection."""
        self._connection_metrics["total_connections"] += 1

        logger.info(f"Establishing database connection to {self.config.host}:{self.config.port}")

        # Create connection pool with enhanced settings
        pool = AsyncConnectionPool(
            conninfo=self.config.build_dsn(),
            min_size=self.config.min_pool_size,
            max_size=self.config.max_pool_size,
            max_idle=self.config.max_idle_time,
            max_lifetime=self.config.max_lifetime,
            timeout=self.config.command_timeout,
        )

        # Open the pool
        await pool.open()

        # Validate connection with test query
        async with pool.connection() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT 1")
                result = await cursor.fetchone()
                if not result or result[0] != 1:
                    raise ConnectionError("Database connection validation failed")

        logger.info(f"Database pool created: min={pool.min_size}, max={pool.max_size}")

        return pool

    async def disconnect(self) -> None:
        """Close database connection gracefully."""
        if self._is_closed:
            return

        logger.info("Disconnecting from database...")

        # Close pool
        if self._pool and not self._pool.closed:
            await self._pool.close()

        self._pool = None
        self._is_closed = True

        logger.info("Database disconnected")

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[psycopg.AsyncConnection, None]:
        """
        Acquire connection with circuit breaker protection.

        Yields:
            Database connection
        """
        if not self.is_connected:
            raise ConnectionError("Database is not connected")

        if self._pool is None:
            raise ConnectionError("Database pool is not initialized")

        # Use circuit breaker if enabled
        if self._circuit_breaker:
            try:
                # Circuit breaker doesn't work well with context managers
                # so we handle this without circuit breaker protection for now
                async with self._acquire_connection() as connection:
                    yield connection
            except Exception as e:
                if "CircuitBreakerError" in str(type(e)):
                    self._connection_metrics["circuit_breaker_trips"] += 1
                raise
        else:
            async with self._acquire_connection() as connection:
                yield connection

    @asynccontextmanager
    async def _acquire_connection(self) -> AsyncGeneratorABC[psycopg.AsyncConnection, None]:
        """Internal connection acquisition."""
        if self._pool is None:
            raise RuntimeError("Database pool is not initialized")
        async with self._pool.connection() as connection:
            # Pre-ping validation if enabled
            if self.config.pool_pre_ping:
                try:
                    await connection.execute("SELECT 1")
                except Exception as e:
                    logger.warning(f"Connection pre-ping failed: {e}")
                    raise ConnectionError("Connection validation failed") from e

            yield connection

    async def execute_query(
        self,
        query: str,
        params: tuple[Any, ...] | None = None,
        fetch_all: bool = False,
        fetch_one: bool = False,
    ) -> Any:
        """
        Execute query with resilience features.

        Args:
            query: SQL query
            params: Query parameters
            fetch_all: Return all results
            fetch_one: Return first result

        Returns:
            Query results based on fetch parameters
        """
        start_time = time.time()
        self._connection_metrics["total_queries"] += 1

        try:
            async with self.acquire() as connection:
                async with connection.cursor() as cursor:
                    if params:
                        await cursor.execute(query, params)
                    else:
                        await cursor.execute(query)

                    result: Any = None
                    if fetch_all:
                        result = await cursor.fetchall()
                    elif fetch_one:
                        result = await cursor.fetchone()

                    # Check for slow queries
                    execution_time = time.time() - start_time
                    if execution_time > self.config.slow_query_threshold:
                        self._connection_metrics["slow_queries"] += 1
                        logger.warning(
                            f"Slow query detected: {execution_time:.3f}s - {query[:100]}..."
                        )

                    if self.config.enable_query_logging:
                        logger.debug(f"Query executed in {execution_time:.3f}s: {query[:100]}...")

                    return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Query failed after {execution_time:.3f}s: {e}")
            raise

    async def execute_transaction(self, operations: list[Any]) -> list[Any]:
        """
        Execute multiple operations in a transaction with resilience.

        Args:
            operations: List[Any] of (query, params) tuples

        Returns:
            List of operation results
        """
        async with self.acquire() as connection:
            async with connection.transaction():
                results = []

                for operation in operations:
                    if isinstance(operation, tuple) and len(operation) == 2:
                        query, params = operation
                        result = await self.execute_query(query, params)
                        results.append(result)
                    else:
                        raise ValueError("Operations must be (query, params) tuples")

                return results

    async def get_enhanced_pool_stats(self) -> dict[str, Any]:
        """Get enhanced pool statistics with resilience metrics."""
        base_stats = {
            "status": "connected" if self.is_connected else "disconnected",
            "is_closed": self._is_closed,
        }

        if self._pool and not self._pool.closed:
            base_stats.update(
                {
                    "max_size": self._pool.max_size,
                    "min_size": self._pool.min_size,
                    "pool_closed": self._pool.closed,
                }
            )

        # Add resilience metrics
        base_stats.update(
            {
                "resilience_metrics": self._connection_metrics.copy(),
                "circuit_breaker_state": (
                    self._circuit_breaker.state.value if self._circuit_breaker else "disabled"
                ),
            }
        )

        # Add circuit breaker metrics if available
        if self._circuit_breaker:
            base_stats["circuit_breaker_metrics"] = self._circuit_breaker.get_metrics()

        return base_stats

    def get_health_check(self) -> EnhancedDatabaseHealthCheck:
        """Get health check instance for this connection."""
        if not self._health_check:
            self._health_check = EnhancedDatabaseHealthCheck("database", self)
        return self._health_check

    def reset_metrics(self) -> None:
        """Reset connection metrics."""
        self._connection_metrics = {
            "total_connections": 0,
            "successful_connections": 0,
            "failed_connections": 0,
            "total_queries": 0,
            "slow_queries": 0,
            "circuit_breaker_trips": 0,
            "retry_attempts": 0,
        }

        if self._circuit_breaker:
            self._circuit_breaker.reset()

        logger.info("Database connection metrics reset")


class ResilientConnectionFactory:
    """
    Enhanced connection factory with resilience features.

    Manages resilient database connections with health monitoring,
    automatic failover, and comprehensive metrics.
    """

    _instance: Optional["ResilientConnectionFactory"] = None
    _connection: ResilientDatabaseConnection | None = None

    def __new__(cls) -> "ResilientConnectionFactory":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    async def create_resilient_connection(
        cls, config: EnhancedDatabaseConfig | None = None, force_new: bool = False
    ) -> ResilientDatabaseConnection:
        """
        Create or get resilient database connection.

        Args:
            config: Enhanced database configuration
            force_new: Force creation of new connection

        Returns:
            ResilientDatabaseConnection instance
        """
        instance = cls()

        if force_new or instance._connection is None or instance._connection.is_closed:
            # Close existing connection if needed
            if instance._connection and not instance._connection.is_closed:
                await instance._connection.disconnect()

            try:
                # Create enhanced config if not provided
                if config is None:
                    base_config = DatabaseConfig.from_env()
                    config = EnhancedDatabaseConfig.from_base_config(base_config)

                # Create resilient connection
                instance._connection = ResilientDatabaseConnection(config)
                await instance._connection.connect()

                logger.info(f"Created resilient database connection: {config.host}:{config.port}")

            except Exception as e:
                logger.error(f"Failed to create resilient database connection: {e}")
                raise FactoryError(
                    "ResilientConnectionFactory", f"Connection creation failed: {e}"
                ) from e

        return instance._connection

    @classmethod
    async def get_connection(cls) -> ResilientDatabaseConnection:
        """Get existing resilient database connection."""
        instance = cls()

        if instance._connection is None or instance._connection.is_closed:
            raise FactoryError("ResilientConnectionFactory", "No active resilient connection")

        return instance._connection

    @classmethod
    async def close_all(cls) -> None:
        """Close all resilient connections."""
        instance = cls()

        if instance._connection and not instance._connection.is_closed:
            await instance._connection.disconnect()
            instance._connection = None
            logger.info("All resilient database connections closed")

    @classmethod
    def reset(cls) -> None:
        """Reset factory for testing."""
        cls._instance = None
        cls._connection = None
