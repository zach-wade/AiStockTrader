"""
PostgreSQL Database Adapter

Provides async database operations using psycopg3 for the AI Trading System.
Handles connection management, query execution, and error handling.
"""

# Standard library imports
import builtins
import logging
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from typing import Any

# Third-party imports
import psycopg
from psycopg import AsyncConnection
from psycopg.rows import Row, dict_row
from psycopg_pool import AsyncConnectionPool

# Local imports
from src.application.interfaces.exceptions import (
    ConnectionError,
    IntegrityError,
    RepositoryError,
    TimeoutError,
    TransactionError,
)

logger = logging.getLogger(__name__)


class PostgreSQLAdapter:
    """
    PostgreSQL database adapter using psycopg3.

    Provides high-level database operations with error handling,
    connection management, and transaction support.
    """

    def __init__(self, pool: AsyncConnectionPool) -> None:
        """
        Initialize adapter with connection pool.

        Args:
            pool: psycopg3 async connection pool
        """
        self._pool = pool
        self._connection: AsyncConnection | None = None
        self._transaction: psycopg.AsyncTransaction | None = None

    @property
    def pool(self) -> AsyncConnectionPool:
        """Get the connection pool."""
        return self._pool

    @property
    def has_active_transaction(self) -> bool:
        """Check if there's an active transaction."""
        return self._transaction is not None

    @asynccontextmanager
    async def acquire_connection(self) -> AsyncGenerator[AsyncConnection, None]:
        """
        Acquire a database connection from the pool.

        Yields:
            Database connection

        Raises:
            ConnectionError: If connection cannot be acquired
        """
        if self._connection:
            # Use existing connection if in transaction
            yield self._connection
            return

        try:
            async with self._pool.connection() as connection:
                yield connection
        except psycopg.OperationalError as e:
            logger.error(f"Failed to acquire connection: {e}")
            raise ConnectionError(f"Failed to acquire database connection: {e}") from e
        except builtins.TimeoutError as e:
            logger.error(f"Connection acquisition timed out: {e}")
            raise TimeoutError("acquire_connection", 30.0) from e

    async def execute_query(
        self,
        query: str,
        *args: Any,
        timeout: float | None = None,
    ) -> str:
        """
        Execute a SQL query that doesn't return data.

        Args:
            query: SQL query string
            *args: Query parameters
            timeout: Query timeout in seconds

        Returns:
            Query status string

        Raises:
            RepositoryError: If query execution fails
            TimeoutError: If query times out
        """
        try:
            async with self.acquire_connection() as conn, conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(query, args)
                result = f"EXECUTE {cur.rowcount}"
                logger.debug(f"Query executed: {query[:100]}... | Result: {result}")
                return result
        except psycopg.IntegrityError as e:
            logger.error(f"Integrity constraint violated: {e} | Query: {query[:100]}...")
            raise IntegrityError(str(e), str(e)) from e
        except psycopg.OperationalError as e:
            logger.error(f"Query execution failed: {e} | Query: {query[:100]}...")
            raise RepositoryError(f"Query execution failed: {e}") from e
        except builtins.TimeoutError as e:
            logger.error(f"Query timed out: {query[:100]}...")
            raise TimeoutError("execute_query", timeout or 30.0) from e

    async def fetch_one(
        self,
        query: str,
        *args: Any,
        timeout: float | None = None,
    ) -> Row | None:
        """
        Fetch a single record from the database.

        Args:
            query: SQL query string
            *args: Query parameters
            timeout: Query timeout in seconds

        Returns:
            Record if found, None otherwise

        Raises:
            RepositoryError: If query execution fails
            TimeoutError: If query times out
        """
        try:
            async with self.acquire_connection() as conn, conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(query, args)
                result = await cur.fetchone()
                logger.debug(f"Fetch one query: {query[:100]}... | Found: {result is not None}")
                return result
        except psycopg.OperationalError as e:
            logger.error(f"Fetch one failed: {e} | Query: {query[:100]}...")
            raise RepositoryError(f"Fetch one query failed: {e}") from e
        except builtins.TimeoutError as e:
            logger.error(f"Fetch one timed out: {query[:100]}...")
            raise TimeoutError("fetch_one", timeout or 30.0) from e

    async def fetch_all(
        self,
        query: str,
        *args: Any,
        timeout: float | None = None,
    ) -> list[Row]:
        """
        Fetch all records from the database.

        Args:
            query: SQL query string
            *args: Query parameters
            timeout: Query timeout in seconds

        Returns:
            List of records

        Raises:
            RepositoryError: If query execution fails
            TimeoutError: If query times out
        """
        try:
            async with self.acquire_connection() as conn, conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(query, args)
                result = await cur.fetchall()
                logger.debug(f"Fetch all query: {query[:100]}... | Count: {len(result)}")
                return result
        except psycopg.OperationalError as e:
            logger.error(f"Fetch all failed: {e} | Query: {query[:100]}...")
            raise RepositoryError(f"Fetch all query failed: {e}") from e
        except builtins.TimeoutError as e:
            logger.error(f"Fetch all timed out: {query[:100]}...")
            raise TimeoutError("fetch_all", timeout or 30.0) from e

    async def fetch_values(
        self,
        query: str,
        *args: Any,
        timeout: float | None = None,
    ) -> list[Any]:
        """
        Fetch values from a single column.

        Args:
            query: SQL query string that returns single column
            *args: Query parameters
            timeout: Query timeout in seconds

        Returns:
            List of values from first column

        Raises:
            RepositoryError: If query execution fails
            TimeoutError: If query times out
        """
        try:
            async with self.acquire_connection() as conn, conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(query, args)
                result = await cur.fetchall()
                # Get the first column name from the result
                if result:
                    first_key = list(result[0].keys())[0]
                    values = [record[first_key] for record in result]
                else:
                    values = []
                logger.debug(f"Fetch values query: {query[:100]}... | Count: {len(values)}")
                return values
        except (psycopg.OperationalError, IndexError) as e:
            logger.error(f"Fetch values failed: {e} | Query: {query[:100]}...")
            raise RepositoryError(f"Fetch values query failed: {e}") from e
        except builtins.TimeoutError as e:
            logger.error(f"Fetch values timed out: {query[:100]}...")
            raise TimeoutError("fetch_values", timeout or 30.0) from e

    async def execute_batch(
        self,
        query: str,
        args_list: Sequence[Sequence[Any]],
        timeout: float | None = None,
    ) -> None:
        """
        Execute a query multiple times with different parameters.

        Args:
            query: SQL query string
            args_list: List of parameter tuples
            timeout: Query timeout in seconds

        Raises:
            RepositoryError: If batch execution fails
            TimeoutError: If batch execution times out
        """
        try:
            async with self.acquire_connection() as conn, conn.cursor(row_factory=dict_row) as cur:
                await cur.executemany(query, args_list)
                logger.debug(
                    f"Batch query executed: {query[:100]}... | Batch size: {len(args_list)}"
                )
        except psycopg.IntegrityError as e:
            logger.error(f"Batch integrity constraint violated: {e} | Query: {query[:100]}...")
            raise IntegrityError(str(e), str(e)) from e
        except psycopg.OperationalError as e:
            logger.error(f"Batch execution failed: {e} | Query: {query[:100]}...")
            raise RepositoryError(f"Batch execution failed: {e}") from e
        except builtins.TimeoutError as e:
            logger.error(f"Batch execution timed out: {query[:100]}...")
            raise TimeoutError("execute_batch", timeout or 30.0) from e

    async def begin_transaction(self) -> None:
        """
        Begin a database transaction.

        Raises:
            TransactionError: If transaction cannot be started
        """
        if self.has_active_transaction:
            raise TransactionError("Transaction is already active")

        try:
            if not self._connection:
                self._connection = await self._pool.connection().__aenter__()

            self._transaction = self._connection.transaction()
            await self._transaction.__aenter__()
            logger.debug("Transaction started")

        except psycopg.OperationalError as e:
            logger.error(f"Failed to start transaction: {e}")
            await self._cleanup_transaction()
            raise TransactionError(f"Failed to start transaction: {e}") from e

    async def commit_transaction(self) -> None:
        """
        Commit the current transaction.

        Raises:
            TransactionError: If commit fails or no active transaction
        """
        if not self.has_active_transaction:
            raise TransactionError("No active transaction to commit")

        try:
            await self._transaction.__aexit__(None, None, None)
            logger.debug("Transaction committed")
        except psycopg.OperationalError as e:
            logger.error(f"Failed to commit transaction: {e}")
            raise TransactionError(f"Failed to commit transaction: {e}") from e
        finally:
            await self._cleanup_transaction()

    async def rollback_transaction(self) -> None:
        """
        Rollback the current transaction.

        Raises:
            TransactionError: If rollback fails
        """
        if not self.has_active_transaction:
            logger.warning("No active transaction to rollback")
            return

        try:
            await self._transaction.__aexit__(Exception, Exception(), None)
            logger.debug("Transaction rolled back")
        except psycopg.OperationalError as e:
            logger.error(f"Failed to rollback transaction: {e}")
            raise TransactionError(f"Failed to rollback transaction: {e}") from e
        finally:
            await self._cleanup_transaction()

    async def _cleanup_transaction(self) -> None:
        """Clean up transaction state."""
        if self._connection:
            try:
                await self._connection.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Failed to release connection: {e}")
            finally:
                self._connection = None

        self._transaction = None

    async def health_check(self) -> bool:
        """
        Perform a health check on the database connection.

        Returns:
            True if database is healthy, False otherwise
        """
        try:
            async with self.acquire_connection() as conn, conn.cursor(row_factory=dict_row) as cur:
                await cur.execute("SELECT 1")
                await cur.fetchone()
                return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def get_connection_info(self) -> dict[str, Any]:
        """
        Get information about the connection pool.

        Returns:
            Dictionary with connection pool statistics
        """
        # psycopg3 doesn't expose detailed pool stats like asyncpg
        # Return basic information
        return {
            "max_size": self._pool.max_size,
            "min_size": self._pool.min_size,
            "pool_status": "active" if not self._pool.closed else "closed",
        }

    def __str__(self) -> str:
        """String representation of the adapter."""
        pool_info = f"Pool(max_size={self._pool.max_size})"
        tx_info = "with active transaction" if self.has_active_transaction else "no transaction"
        return f"PostgreSQLAdapter({pool_info}, {tx_info})"
