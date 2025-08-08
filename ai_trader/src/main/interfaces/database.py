"""
Database Interfaces

Defines contracts for database operations to ensure clean separation
between async and sync implementations while maintaining consistency.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncContextManager, ContextManager, Protocol
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime


class IDatabase(Protocol):
    """Base interface for all database operations."""
    
    async def initialize(self) -> None:
        """Initialize the database connection."""
        ...
    
    async def close(self) -> None:
        """Close the database connection."""
        ...
    
    async def execute_query(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """Execute a query and return results."""
        ...
    
    async def fetch_one(self, query: str, parameters: Optional[Dict] = None) -> Optional[Dict]:
        """Fetch one result from a query."""
        ...
    
    async def fetch_all(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """Fetch all results from a query."""
        ...
    
    async def insert(self, table: str, data: Dict[str, Any]) -> bool:
        """Insert data into a table."""
        ...
    
    async def update(self, table: str, data: Dict[str, Any], where: Dict[str, Any]) -> bool:
        """Update data in a table."""
        ...
    
    async def delete(self, table: str, where: Dict[str, Any]) -> bool:
        """Delete data from a table."""
        ...
    
    async def execute_many(self, query: str, data: List[Dict[str, Any]]) -> bool:
        """Execute a query with multiple parameter sets."""
        ...
    
    async def transaction(self, operations: List[Dict[str, Any]]) -> bool:
        """Execute multiple operations in a transaction."""
        ...
    
    async def execute_scalar(self, query: str, parameters: Optional[Dict] = None) -> Any:
        """Execute a query and return a single scalar value."""
        ...


class IAsyncDatabase(IDatabase):
    """Interface for async database operations with connection management."""
    
    @asynccontextmanager
    async def acquire(self) -> AsyncContextManager[Any]:
        """Acquire a database connection from the pool."""
        ...
    
    async def run_sync(self, func) -> Any:
        """Execute a synchronous database operation."""
        ...




class IDatabasePool(Protocol):
    """Interface for database connection pooling."""
    
    def initialize(self, database_url: Optional[str] = None, config: Optional[Any] = None) -> None:
        """Initialize the connection pool."""
        ...
    
    def dispose(self) -> None:
        """Dispose of the connection pool."""
        ...
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get the status of the connection pool."""
        ...
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pool performance metrics."""
        ...


class IDatabaseFactory(Protocol):
    """Factory interface for creating database instances."""
    
    def create_async_database(self, config: Any) -> IAsyncDatabase:
        """Create an async database instance."""
        ...
    
    
    def get_database(self, db_type: str, config: Any) -> IAsyncDatabase:
        """Get a database instance based on type."""
        ...