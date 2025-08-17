"""Database utilities package."""

from .helpers import (
    ConnectionHealthStatus,
    ConnectionPoolMetrics,
    PoolHealthMonitor,
    QueryPerformanceTracker,
    QueryPriority,
    QueryTracker,
    QueryType,
    get_global_tracker,
    track_query,
)
from .operations import (
    BatchOperationResult,
    TransactionStrategy,
    batch_delete,
    batch_upsert,
    execute_with_retry,
    transaction_context,
)
from .pool import DatabasePool

# Global database pool instance
_global_db_pool = None


def get_global_db_pool(config=None):
    """
    Get the global database pool instance.

    Args:
        config: Configuration dictionary (required for first initialization)

    Returns:
        DatabasePool instance

    Raises:
        ValueError: If config is not provided for first initialization
    """
    global _global_db_pool
    if _global_db_pool is None:
        if config is None:
            raise ValueError("Configuration is required for first DatabasePool initialization")
        _global_db_pool = DatabasePool(config)
    return _global_db_pool


def get_default_db_pool(config=None):
    """
    Get the default database pool (alias for get_global_db_pool).

    Args:
        config: Configuration dictionary (required for first initialization)

    Returns:
        DatabasePool instance
    """
    return get_global_db_pool(config)


async def execute_query(query: str, params: tuple = None):
    """
    Execute a database query using the global pool.

    Args:
        query: SQL query string
        params: Query parameters

    Returns:
        Query results
    """
    pool = get_global_db_pool()
    async with pool.acquire() as conn:
        if params:
            result = await conn.fetch(query, *params)
        else:
            result = await conn.fetch(query)
    return result


async def execute_command(query: str, params: tuple = None):
    """
    Execute a database command (INSERT, UPDATE, DELETE) using the global pool.

    Args:
        query: SQL command string
        params: Command parameters

    Returns:
        Command result
    """
    pool = get_global_db_pool()
    async with pool.acquire() as conn:
        if params:
            result = await conn.execute(query, *params)
        else:
            result = await conn.execute(query)
    return result


__all__ = [
    "DatabasePool",
    "ConnectionPoolMetrics",
    "ConnectionHealthStatus",
    "PoolHealthMonitor",
    "QueryTracker",
    "QueryPerformanceTracker",
    "QueryType",
    "QueryPriority",
    "track_query",
    "get_global_tracker",
    "get_global_db_pool",
    "execute_query",
    "execute_command",
    "TransactionStrategy",
    "BatchOperationResult",
    "batch_upsert",
    "batch_delete",
    "execute_with_retry",
    "transaction_context",
]
