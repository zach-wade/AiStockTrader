"""
Example Usage of Infrastructure Layer

Demonstrates how to use the PostgreSQL infrastructure implementations
for the AI Trading System. Shows database setup, repository usage,
and transaction management.
"""

# Standard library imports
import asyncio
import logging
from decimal import Decimal

# Local imports
from src.domain.entities.order import Order, OrderSide
from src.domain.entities.portfolio import Portfolio
from src.domain.entities.position import Position
from src.infrastructure.database import (
    ConnectionFactory,
    DatabaseConfig,
    MigrationManager,
    PostgreSQLAdapter,
)
from src.infrastructure.repositories import (
    PostgreSQLTransactionManager,
    PostgreSQLUnitOfWorkFactory,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def setup_database() -> None:
    """Initialize database connection and run migrations."""
    logger.info("Setting up database connection...")

    # Create database configuration
    config = DatabaseConfig.from_env()

    # Establish connection
    connection = await ConnectionFactory.create_connection(config)

    # Create adapter and migration manager
    adapter = PostgreSQLAdapter(connection._pool)
    migration_manager = MigrationManager(adapter)

    # Initialize migration system
    await migration_manager.initialize()

    # Run migrations (in production, you might want to do this separately)
    await migration_manager.migrate_to_latest()

    logger.info("Database setup completed")


async def example_trading_workflow() -> None:
    """
    Example trading workflow demonstrating the infrastructure layer.

    Creates a portfolio, opens a position, places orders, and manages transactions.
    """
    logger.info("Starting example trading workflow...")

    # Create Unit of Work factory and transaction manager
    factory = PostgreSQLUnitOfWorkFactory()
    transaction_manager = PostgreSQLTransactionManager(factory)

    # Example 1: Create a portfolio
    async def create_portfolio(uow):
        """Create a new trading portfolio."""
        portfolio = Portfolio(
            name="Example Trading Portfolio",
            initial_capital=Decimal("100000"),
            cash_balance=Decimal("100000"),
            max_position_size=Decimal("10000"),
            strategy="Example Strategy",
        )

        saved_portfolio = await uow.portfolios.save_portfolio(portfolio)
        logger.info(f"Created portfolio: {saved_portfolio.name}")
        return saved_portfolio

    # Example 2: Place a market order
    async def place_market_order(uow):
        """Place a market buy order."""
        order = Order.create_market_order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            reason="Example market order",
        )

        saved_order = await uow.orders.save_order(order)
        logger.info(f"Placed order: {saved_order}")
        return saved_order

    # Example 3: Open a position
    async def open_position(uow):
        """Open a new trading position."""
        position = Position.open_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            commission=Decimal("1.00"),
            strategy="Example Strategy",
        )

        saved_position = await uow.positions.save_position(position)
        logger.info(f"Opened position: {saved_position}")
        return saved_position

    # Example 4: Complex transaction with multiple operations
    async def complex_trading_operation(uow):
        """Perform multiple operations in a single transaction."""
        # Get the portfolio
        portfolio = await uow.portfolios.get_portfolio_by_name("Example Trading Portfolio")
        if not portfolio:
            raise ValueError("Portfolio not found")

        # Place a limit order
        limit_order = Order.create_limit_order(
            symbol="TSLA",
            quantity=Decimal("50"),
            side=OrderSide.BUY,
            limit_price=Decimal("200.00"),
            reason="Complex operation limit order",
        )
        await uow.orders.save_order(limit_order)

        # Simulate order filling
        limit_order.submit("BROKER123")
        limit_order.fill(Decimal("50"), Decimal("199.50"))
        await uow.orders.update_order(limit_order)

        # Open corresponding position
        position = Position.open_position(
            symbol="TSLA",
            quantity=Decimal("50"),
            entry_price=Decimal("199.50"),
            commission=Decimal("0.50"),
            strategy="Example Strategy",
        )
        await uow.positions.save_position(position)

        # Update portfolio
        portfolio.cash_balance -= Decimal("50") * Decimal("199.50") + Decimal("0.50")
        portfolio.trades_count += 1
        await uow.portfolios.update_portfolio(portfolio)

        logger.info("Complex trading operation completed")
        return {"order": limit_order, "position": position, "portfolio": portfolio}

    try:
        # Execute operations using transaction manager
        await transaction_manager.execute_in_transaction(create_portfolio)
        await transaction_manager.execute_in_transaction(place_market_order)
        await transaction_manager.execute_in_transaction(open_position)

        # Execute complex operation with retry capability
        await transaction_manager.execute_with_retry(
            complex_trading_operation, max_retries=3, backoff_factor=0.5
        )

        logger.info("All operations completed successfully")

        # Retrieve and display final state
        async with await factory.create_unit_of_work_async() as uow:
            all_orders = await uow.orders.get_active_orders()
            all_positions = await uow.positions.get_active_positions()
            current_portfolio = await uow.portfolios.get_current_portfolio()

            logger.info(f"Active orders: {len(all_orders)}")
            logger.info(f"Active positions: {len(all_positions)}")
            logger.info(f"Current portfolio value: ${current_portfolio.get_total_value()}")

    except Exception as e:
        logger.error(f"Trading workflow failed: {e}")
        raise


async def example_query_operations() -> None:
    """Demonstrate various query operations."""
    logger.info("Demonstrating query operations...")

    factory = PostgreSQLUnitOfWorkFactory()

    async with await factory.create_unit_of_work_async() as uow:
        # Repository query examples
        active_orders = await uow.orders.get_active_orders()
        logger.info(f"Found {len(active_orders)} active orders")

        aapl_positions = await uow.positions.get_positions_by_symbol("AAPL")
        logger.info(f"Found {len(aapl_positions)} AAPL positions")

        all_portfolios = await uow.portfolios.get_all_portfolios()
        logger.info(f"Found {len(all_portfolios)} portfolios")

        # Display some details
        for portfolio in all_portfolios:
            logger.info(f"Portfolio: {portfolio.name}, Value: ${portfolio.get_total_value()}")


async def main() -> None:
    """Main example function."""
    try:
        # Setup database
        await setup_database()

        # Run trading workflow
        await example_trading_workflow()

        # Demonstrate queries
        await example_query_operations()

        logger.info("Example completed successfully!")

    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise

    finally:
        # Cleanup connections
        await ConnectionFactory.close_all()


if __name__ == "__main__":
    asyncio.run(main())
