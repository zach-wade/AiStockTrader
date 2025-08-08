"""
Database Factory

Factory for creating database instances based on configuration.
Implements the IDatabaseFactory interface to provide consistent
database instance creation for the new data pipeline.
"""

from typing import Any
from main.interfaces.database import IDatabaseFactory, IAsyncDatabase
from main.utils.core import get_logger

logger = get_logger(__name__)


class DatabaseFactory(IDatabaseFactory):
    """
    Factory for creating database adapter instances.
    
    This factory encapsulates the logic for creating the appropriate
    database adapter based on configuration and requirements.
    """
    
    def create_async_database(self, config: Any) -> IAsyncDatabase:
        """
        Create an async database instance.
        
        Args:
            config: Configuration for the database
            
        Returns:
            IAsyncDatabase instance
        """
        from .database_adapter import AsyncDatabaseAdapter
        
        logger.info("Creating AsyncDatabaseAdapter instance")
        adapter = AsyncDatabaseAdapter(config)
        return adapter
    
    def create_sync_database(self, config: Any) -> Any:
        """
        Create a sync database instance.
        
        Note: Currently not implemented as the system is fully async.
        
        Args:
            config: Configuration for the database
            
        Returns:
            Sync database instance
            
        Raises:
            NotImplementedError: Sync databases are not supported
        """
        raise NotImplementedError("Sync database operations are not supported in the new pipeline")
    
    def get_database(self, db_type: str, config: Any) -> IAsyncDatabase:
        """
        Get a database instance based on type.
        
        Args:
            db_type: Type of database ('async')
            config: Configuration for the database
            
        Returns:
            Database instance (async only)
            
        Raises:
            ValueError: If db_type is not 'async'
        """
        if db_type.lower() == 'async':
            return self.create_async_database(config)
        else:
            raise ValueError(f"Unknown database type: {db_type}. Only 'async' is supported")


# Singleton instance
_database_factory = None


def get_database_factory() -> DatabaseFactory:
    """
    Get the singleton database factory instance.
    
    Returns:
        DatabaseFactory instance
    """
    global _database_factory
    if _database_factory is None:
        _database_factory = DatabaseFactory()
    return _database_factory


def create_database(db_type: str, config: Any) -> IAsyncDatabase:
    """
    Convenience function to create a database instance.
    
    Args:
        db_type: Type of database ('async')
        config: Configuration for the database
        
    Returns:
        Database instance
    """
    factory = get_database_factory()
    return factory.get_database(db_type, config)