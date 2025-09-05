"""Infrastructure Layer for AI Trading System.

This module provides concrete implementations of the application layer interfaces.
Includes PostgreSQL database repositories, Redis caching, connection management,
and data persistence optimized for high-frequency trading operations.

The infrastructure layer implements the following components:
- Database connection management with pooling
- PostgreSQL repositories for Orders, Positions, and Portfolios
- Unit of Work pattern for transaction management
- Database migration system for schema versioning
- Redis-based caching layer for performance optimization
- Cache decorators for automatic result caching
- Trading-specific cache operations (market data, portfolio, risk)
- Error handling and mapping to domain exceptions
- Circuit breaker integration for resilience

Key modules:
- database: Core database connectivity and management
- repositories: Concrete repository implementations
- cache: Redis caching layer with trading-specific optimizations

Example usage:
    from src.infrastructure.database import ConnectionFactory
    from src.infrastructure.repositories import PostgreSQLUnitOfWorkFactory
    from src.infrastructure.cache import CacheManager, cache_market_data

    # Initialize database connection
    connection = await ConnectionFactory.create_connection()

    # Create Unit of Work for transaction management
    factory = PostgreSQLUnitOfWorkFactory()
    uow = await factory.create_unit_of_work_async()

    # Initialize cache manager
    cache_manager = CacheManager()
    await cache_manager.connect()

    # Use repositories within transaction
    async with uow:
        order = await uow.orders.get_order_by_id(order_id)
        portfolio = await uow.portfolios.get_current_portfolio()

        # Cache portfolio calculation
        await cache_manager.cache_portfolio_calculation(
            portfolio.id, "value", portfolio.total_value
        )

    # Use cache decorators
    @cache_market_data(ttl=60)
    async def get_stock_price(symbol: str) -> Any:
        return await market_data_service.get_price(symbol)
"""

from .cache import CacheConfig as CachingConfig
from .cache import (
    CacheManager,
    RedisCache,
    cache_market_data,
    cache_portfolio_calculation,
    cache_result,
    cache_risk_calculation,
    invalidate_cache,
)
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
    # Cache components
    "CacheManager",
    "RedisCache",
    "CachingConfig",
    "cache_result",
    "invalidate_cache",
    "cache_market_data",
    "cache_portfolio_calculation",
    "cache_risk_calculation",
]
