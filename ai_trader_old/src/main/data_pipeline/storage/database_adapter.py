"""
Database Adapter - Async Implementation

Provides async database operations using existing utilities for pooling,
retry logic, and resilience patterns.
"""

# Standard library imports
import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncContextManager

# Third-party imports
import asyncpg

# Local imports
from main.interfaces.database import IAsyncDatabase
from main.utils.core import get_logger
from main.utils.database.operations import execute_with_retry
from main.utils.database.pool import DatabasePool
from main.utils.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

logger = get_logger(__name__)


class AsyncDatabaseAdapter(IAsyncDatabase):
    """
    Async database adapter using utilities for pooling, retry, and resilience.

    Implements the IAsyncDatabase interface with production-ready patterns.
    """

    def __init__(self, config: Any = None):
        """Initialize the database adapter with utilities."""
        self.config = config
        self._pool: asyncpg.Pool | None = None
        self._closed = False

        # Use existing pool utility
        self.db_pool = DatabasePool()
        self.db_pool.initialize(config=config)

        # Initialize circuit breaker for resilience
        cb_config = CircuitBreakerConfig(
            failure_threshold=5, recovery_timeout=30.0, timeout_seconds=10.0
        )
        self.circuit_breaker = CircuitBreaker(cb_config)

        # Note: Metrics are handled by DatabasePool, not duplicated here

        logger.info("AsyncDatabaseAdapter initialized with utilities")

    async def initialize(self) -> None:
        """Initialize the async connection pool."""
        if self._pool is not None:
            return

        # Get connection string from the DatabasePool utility
        connection_string = self.db_pool._database_url

        async def create_pool():
            return await asyncpg.create_pool(
                connection_string,
                min_size=2,
                max_size=20,
                max_queries=50000,
                max_inactive_connection_lifetime=300,
                command_timeout=60,
            )

        # Use circuit breaker for pool creation
        try:
            self._pool = await self.circuit_breaker.call(create_pool)
            logger.info("Async pool initialized with circuit breaker")
            # Metrics are tracked by DatabasePool, not here
        except Exception as e:
            logger.error(f"Failed to initialize async pool: {e}")
            raise

    async def close(self) -> None:
        """Close the database connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._closed = True
            # Metrics are tracked by DatabasePool, not here
            logger.info("Async pool closed")

    @asynccontextmanager
    async def acquire(self) -> AsyncContextManager[asyncpg.Connection]:
        """Acquire a connection with circuit breaker protection."""
        if self._pool is None:
            await self.initialize()

        async def get_connection():
            return self._pool.acquire()

        # Use circuit breaker for connection acquisition
        connection_ctx = await self.circuit_breaker.call(get_connection)
        async with connection_ctx as connection:
            # Connection metrics are tracked by DatabasePool
            try:
                yield connection
            finally:
                pass  # Metrics handled by pool

    async def execute_query(self, query: str, *args) -> Any:
        """Execute a query with retry logic and circuit breaker."""

        async def operation():
            async with self.acquire() as conn:
                return await conn.execute(query, *args)

        return await execute_with_retry(operation, max_retries=3)

    async def fetch_one(self, query: str, *args) -> dict[str, Any] | None:
        """Fetch one result with retry logic."""

        async def operation():
            async with self.acquire() as conn:
                row = await conn.fetchrow(query, *args)
                return dict(row) if row else None

        return await execute_with_retry(operation, max_retries=2)

    async def fetch_all(self, query: str, *args) -> list[dict[str, Any]]:
        """Fetch all results with retry logic."""

        async def operation():
            async with self.acquire() as conn:
                rows = await conn.fetch(query, *args)
                return [dict(row) for row in rows]

        return await execute_with_retry(operation, max_retries=2)

    async def insert(self, table: str, data: dict[str, Any]) -> bool:
        """Insert data with automatic retry."""
        # Local imports
        from main.utils.security.sql_security import validate_identifier_list, validate_table_name

        # Validate table and column names
        safe_table = validate_table_name(table)
        columns = list(data.keys())
        safe_columns = validate_identifier_list(columns)

        values = list(data.values())
        placeholders = [f"${i+1}" for i in range(len(values))]

        query = f"INSERT INTO {safe_table} ({', '.join(safe_columns)}) VALUES ({', '.join(placeholders)})"

        try:
            await self.execute_query(query, *values)
            return True
        except Exception as e:
            logger.error(f"Insert failed: {e}")
            return False

    async def update(self, table: str, data: dict[str, Any], where: dict[str, Any]) -> bool:
        """Update data with automatic retry."""
        set_clauses = []
        values = []
        param_counter = 1

        for key, value in data.items():
            set_clauses.append(f"{key} = ${param_counter}")
            values.append(value)
            param_counter += 1

        for key, value in where.items():
            values.append(value)

        where_clause = " AND ".join(
            f"{k} = ${i}" for i, k in enumerate(where.keys(), param_counter)
        )
        query = f"UPDATE {table} SET {', '.join(set_clauses)} WHERE {where_clause}"

        try:
            await self.execute_query(query, *values)
            return True
        except Exception as e:
            logger.error(f"Update failed: {e}")
            return False

    async def delete(self, table: str, where: dict[str, Any]) -> bool:
        """Delete data with automatic retry."""
        where_clauses = [f"{k} = ${i}" for i, k in enumerate(where.keys(), 1)]
        query = f"DELETE FROM {table} WHERE {' AND '.join(where_clauses)}"

        try:
            await self.execute_query(query, *list(where.values()))
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    async def execute_many(self, query: str, data: list[tuple]) -> bool:
        """Execute query with multiple parameter sets using retry logic."""

        async def operation():
            async with self.acquire() as conn:
                await conn.executemany(query, data)

        try:
            await execute_with_retry(operation, max_retries=2)
            return True
        except Exception as e:
            logger.error(f"Execute many failed: {e}")
            return False

    async def transaction(self, operations: list[dict[str, Any]] = None) -> AsyncContextManager:
        """Execute operations in a transaction with automatic rollback on error.

        Can be used as a context manager or with a list of operations.

        Args:
            operations: Optional list of operations to execute

        Returns:
            Context manager for transaction if operations is None,
            otherwise bool indicating success
        """
        if operations is None:
            # Return context manager for manual transaction control
            return self._transaction_context()

        # Execute operations in transaction
        async def run_transaction():
            async with self.acquire() as conn:
                transaction = conn.transaction()
                await transaction.start()
                try:
                    for op in operations:
                        await conn.execute(op.get("query"), *op.get("params", []))
                    await transaction.commit()
                    return True
                except Exception as e:
                    await transaction.rollback()
                    logger.error(f"Transaction rolled back due to error: {e}")
                    raise

        try:
            result = await self.circuit_breaker.call(run_transaction)
            return result
        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            return False

    @asynccontextmanager
    async def _transaction_context(self):
        """Context manager for manual transaction control."""
        async with self.acquire() as conn:
            transaction = conn.transaction()
            await transaction.start()
            try:
                yield conn
                await transaction.commit()
            except Exception as e:
                await transaction.rollback()
                logger.error(f"Transaction rolled back: {e}")
                raise

    async def execute_scalar(self, query: str, *args) -> Any:
        """Execute query and return scalar value with retry."""

        async def operation():
            async with self.acquire() as conn:
                return await conn.fetchval(query, *args)

        return await execute_with_retry(operation, max_retries=2)

    async def run_sync(self, func) -> Any:
        """Execute synchronous operation (for interface compatibility)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func)
