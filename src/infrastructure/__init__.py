"""
Infrastructure Layer for AI Trading System

This module provides concrete implementations of the application layer interfaces.
Includes PostgreSQL database repositories, connection management, and data persistence.

The infrastructure layer implements the following components:
- Database connection management with pooling
- PostgreSQL repositories for Orders, Positions, and Portfolios
- Unit of Work pattern for transaction management
- Database migration system for schema versioning
- Error handling and mapping to domain exceptions

Key modules:
- database: Core database connectivity and management
- repositories: Concrete repository implementations

Example usage:
    from src.infrastructure.database import ConnectionFactory
    from src.infrastructure.repositories import PostgreSQLUnitOfWorkFactory

    # Initialize database connection
    connection = await ConnectionFactory.create_connection()

    # Create Unit of Work for transaction management
    factory = PostgreSQLUnitOfWorkFactory()
    uow = await factory.create_unit_of_work_async()

    # Use repositories within transaction
    async with uow:
        order = await uow.orders.get_order_by_id(order_id)
        portfolio = await uow.portfolios.get_current_portfolio()
"""

from .database import (
    ConnectionFactory,
    DatabaseConfig,
    DatabaseConnection,
    MigrationManager,
    PostgreSQLAdapter,
)
from .repositories import (
    PostgreSQLOrderRepository,
    PostgreSQLPortfolioRepository,
    PostgreSQLPositionRepository,
    PostgreSQLTransactionManager,
    PostgreSQLUnitOfWork,
    PostgreSQLUnitOfWorkFactory,
)

__all__ = [
    # Database components
    "ConnectionFactory",
    "DatabaseConfig",
    "DatabaseConnection",
    "MigrationManager",
    "PostgreSQLAdapter",
    # Repository implementations
    "PostgreSQLOrderRepository",
    "PostgreSQLPositionRepository",
    "PostgreSQLPortfolioRepository",
    "PostgreSQLUnitOfWork",
    "PostgreSQLUnitOfWorkFactory",
    "PostgreSQLTransactionManager",
]
