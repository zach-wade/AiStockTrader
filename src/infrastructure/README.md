# Infrastructure Layer - AI Trading System

The infrastructure layer provides concrete implementations of the application layer interfaces, handling data persistence and external integrations for the AI Trading System.

## Architecture Overview

The infrastructure layer follows the Repository pattern and implements:

- **Database Layer**: PostgreSQL connection management and query execution
- **Repository Pattern**: Concrete implementations of domain repository interfaces
- **Unit of Work**: Transaction management across multiple repositories
- **Migration System**: Database schema versioning and evolution

## Key Components

### Database Management

#### PostgreSQL Adapter (`database/adapter.py`)

Core database operations with async support:

- Connection pool management
- Query execution (fetch_one, fetch_all, execute)
- Transaction support (begin, commit, rollback)
- Error handling and timeout management
- Health checks and connection monitoring

#### Connection Factory (`database/connection.py`)

Database connection lifecycle management:

- Connection pooling with configurable limits
- SSL/TLS support for secure connections
- Health monitoring and automatic reconnection
- Configuration from environment or explicit settings
- Graceful shutdown and cleanup

#### Database Configuration

```python
from src.infrastructure.database import DatabaseConfig

# From environment variables
config = DatabaseConfig.from_env()

# From database URL
config = DatabaseConfig.from_url("postgresql://user:pass@host:port/db")

# Manual configuration
config = DatabaseConfig(
    host="localhost",
    port=5432,
    database="ai_trader",
    user="zachwade",
    min_pool_size=5,
    max_pool_size=20
)
```

### Repository Implementations

#### Order Repository (`repositories/order_repository.py`)

PostgreSQL implementation of `IOrderRepository`:

- CRUD operations for trading orders
- Query by symbol, status, broker ID, date range
- Support for all order types (market, limit, stop, stop-limit)
- Order status tracking and updates
- Broker integration with external order IDs

#### Position Repository (`repositories/position_repository.py`)

PostgreSQL implementation of `IPositionRepository`:

- Position lifecycle management (open, update, close)
- P&L tracking (realized and unrealized)
- Risk management fields (stop loss, take profit)
- Strategy association and filtering
- Historical position tracking

#### Portfolio Repository (`repositories/portfolio_repository.py`)

PostgreSQL implementation of `IPortfolioRepository`:

- Portfolio management with risk limits
- Position aggregation and portfolio metrics
- Performance tracking (trades, win/loss ratios)
- Portfolio snapshots for historical analysis
- Strategy-based portfolio organization

### Transaction Management

#### Unit of Work (`repositories/unit_of_work.py`)

Implements atomic operations across repositories:

```python
from src.infrastructure.repositories import PostgreSQLUnitOfWorkFactory

factory = PostgreSQLUnitOfWorkFactory()
uow = await factory.create_unit_of_work_async()

# Using context manager (recommended)
async with uow:
    order = await uow.orders.save_order(new_order)
    position = await uow.positions.save_position(new_position)
    portfolio = await uow.portfolios.update_portfolio(updated_portfolio)
    # Automatic commit on success, rollback on exception

# Manual transaction management
await uow.begin_transaction()
try:
    await uow.orders.save_order(order)
    await uow.commit()
except Exception:
    await uow.rollback()
    raise
```

#### Transaction Manager

High-level transaction patterns:

```python
from src.infrastructure.repositories import PostgreSQLTransactionManager

manager = PostgreSQLTransactionManager(factory)

# Execute with automatic retry
result = await manager.execute_with_retry(
    operation=my_operation,
    max_retries=3,
    backoff_factor=1.0
)

# Execute multiple operations in batch
results = await manager.execute_batch([
    operation1,
    operation2,
    operation3
])
```

### Database Schema

#### Tables

- **orders**: Trading orders with full lifecycle tracking
- **positions**: Position management with P&L calculation
- **portfolios**: Portfolio aggregation and risk management

#### Enums

- `order_side`: BUY, SELL
- `order_type`: MARKET, LIMIT, STOP, STOP_LIMIT
- `order_status`: PENDING, SUBMITTED, PARTIALLY_FILLED, FILLED, CANCELLED, REJECTED, EXPIRED
- `time_in_force`: DAY, GTC, IOC, FOK

#### Indexes

Optimized for trading system query patterns:

- Symbol-based lookups
- Status filtering
- Date range queries
- Active record filtering

### Migration System

#### Migration Manager (`database/migrations.py`)

Schema versioning and evolution:

```python
from src.infrastructure.database import MigrationManager

manager = MigrationManager(adapter)
await manager.initialize()

# Apply all pending migrations
count = await manager.migrate_to_latest()

# Migrate to specific version
count = await manager.migrate_to_version("003")

# Get migration status
status = await manager.get_status()
```

## Usage Examples

### Basic Setup

```python
from src.infrastructure.database import ConnectionFactory, DatabaseConfig
from src.infrastructure.repositories import PostgreSQLUnitOfWorkFactory

# Initialize database connection
config = DatabaseConfig.from_env()
connection = await ConnectionFactory.create_connection(config)

# Create repositories
factory = PostgreSQLUnitOfWorkFactory()
uow = await factory.create_unit_of_work_async()
```

### Order Management

```python
from src.domain.entities.order import Order, OrderSide, OrderType
from decimal import Decimal

# Create a new order
order = Order.create_limit_order(
    symbol="AAPL",
    quantity=Decimal("100"),
    side=OrderSide.BUY,
    limit_price=Decimal("150.00")
)

# Save to database
async with uow:
    saved_order = await uow.orders.save_order(order)

    # Update order status
    order.submit("BROKER-12345")
    updated_order = await uow.orders.update_order(order)

    # Query orders
    active_orders = await uow.orders.get_active_orders()
    apple_orders = await uow.orders.get_orders_by_symbol("AAPL")
```

### Position Tracking

```python
from src.domain.entities.position import Position
from datetime import datetime

# Open a new position
position = Position.open_position(
    symbol="AAPL",
    quantity=Decimal("100"),
    entry_price=Decimal("150.50"),
    commission=Decimal("1.00")
)

async with uow:
    # Save position
    saved_position = await uow.positions.save_position(position)

    # Update with current market price
    position.current_price = Decimal("152.00")
    position.last_updated = datetime.now()
    await uow.positions.update_position(position)

    # Query positions
    active_positions = await uow.positions.get_active_positions()
    apple_position = await uow.positions.get_position_by_symbol("AAPL")
```

### Portfolio Management

```python
from src.domain.entities.portfolio import Portfolio

# Create portfolio
portfolio = Portfolio(
    name="Trading Portfolio",
    initial_capital=Decimal("100000"),
    cash_balance=Decimal("95000"),
    max_position_size=Decimal("10000")
)

async with uow:
    # Save portfolio
    saved_portfolio = await uow.portfolios.save_portfolio(portfolio)

    # Update portfolio metrics
    portfolio.total_realized_pnl = Decimal("1250.00")
    portfolio.trades_count = 15
    portfolio.winning_trades = 10
    await uow.portfolios.update_portfolio(portfolio)

    # Get current portfolio
    current = await uow.portfolios.get_current_portfolio()
```

## Error Handling

The infrastructure layer maps database errors to domain exceptions:

- `ConnectionError`: Database connection failures
- `RepositoryError`: General repository operation failures
- `EntityNotFoundError`: Entity lookup failures
- `TransactionError`: Transaction management failures
- `IntegrityError`: Database constraint violations
- `TimeoutError`: Operation timeout failures

## Performance Considerations

### Connection Pooling

- Configurable pool sizes (min: 5, max: 20 by default)
- Connection reuse and lifecycle management
- Health monitoring and automatic cleanup

### Query Optimization

- Strategic indexes for common query patterns
- Prepared statement usage via asyncpg
- Batch operations for bulk updates

### Transaction Management

- Short-lived transactions to minimize lock contention
- Proper error handling and rollback procedures
- Connection sharing within transaction scope

## Configuration

### Environment Variables

```bash
# Database connection
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=ai_trader
DATABASE_USER=zachwade
DATABASE_PASSWORD=secret

# Connection pooling
DATABASE_MIN_POOL_SIZE=5
DATABASE_MAX_POOL_SIZE=20
DATABASE_MAX_IDLE_TIME=300.0
DATABASE_MAX_LIFETIME=3600.0

# Security
DATABASE_SSL_MODE=prefer
DATABASE_SSL_CERT_FILE=/path/to/cert.pem
DATABASE_SSL_KEY_FILE=/path/to/key.pem
```

### Database URL

```bash
DATABASE_URL=postgresql://user:pass@host:port/database?sslmode=require
```

## Testing

The infrastructure layer supports dependency injection for testing:

```python
# Test with in-memory database or mock
test_adapter = MockPostgreSQLAdapter()
test_repository = PostgreSQLOrderRepository(test_adapter)

# Integration tests with test database
test_config = DatabaseConfig(database="ai_trader_test")
test_connection = await ConnectionFactory.create_connection(test_config)
```

## Dependencies

- `asyncpg`: PostgreSQL async driver
- `asyncio`: Asynchronous programming support
- Python 3.11+ with type hints

## Security

- SSL/TLS support for encrypted connections
- Parameterized queries to prevent SQL injection
- Connection credential management
- Database privilege separation
