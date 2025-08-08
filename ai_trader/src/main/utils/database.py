"""
Database utilities for the AI Trader system.

This module provides a unified interface to all database utilities including:
- Connection pool management for improved performance
- Connection health monitoring and metrics
- Query performance tracking and optimization
- Database operation helpers
- Transaction management utilities

This is the main interface module that imports all database utilities from the 
database/ subdirectory for easy access throughout the system.
"""

# Import and explicitly re-export database utilities
from .database import (
    # Core database components
    DatabasePool,
    
    # Connection management
    ConnectionPoolMetrics,
    ConnectionHealthStatus,
    PoolHealthMonitor,
    
    # Query tracking
    QueryTracker,
    QueryPerformanceTracker,
    QueryType,
    QueryPriority,
    track_query,
    get_global_tracker,
    
    # Batch operations
    TransactionStrategy,
    BatchOperationResult,
    batch_upsert,
    batch_delete,
    execute_with_retry,
    transaction_context
)

# Convenience imports for common patterns
from .database.pool import DatabasePool as ConnectionPool
from .database.helpers import get_global_tracker as get_query_tracker

# Version info
__version__ = "2.0.0"
__author__ = "AI Trader Team"

# Default configuration
DEFAULT_POOL_SIZE = 20
DEFAULT_MAX_OVERFLOW = 10
DEFAULT_POOL_RECYCLE = 3600  # 1 hour
DEFAULT_POOL_TIMEOUT = 30    # 30 seconds