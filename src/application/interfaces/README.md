# Repository Layer - Application Interfaces

This directory contains the repository interface definitions following clean architecture principles. The application layer defines what it needs from the infrastructure layer through these contracts.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Interfaces (This Module)               │   │
│  │  • IOrderRepository     • RepositoryError          │   │
│  │  • IPositionRepository  • TransactionError         │   │
│  │  • IPortfolioRepository • IUnitOfWork              │   │
│  │  • ITransactionManager  • IUnitOfWorkFactory       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ implements
                              │
┌─────────────────────────────────────────────────────────────┐
│                  Infrastructure Layer                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Concrete Implementations                  │   │
│  │  • PostgreSQLOrderRepository                       │   │
│  │  • PostgreSQLPositionRepository                    │   │
│  │  • PostgreSQLPortfolioRepository                   │   │
│  │  • PostgreSQLUnitOfWork                            │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Clean Architecture Principles

1. **Dependency Inversion**: Application layer defines interfaces, infrastructure implements them
2. **Single Responsibility**: Each repository handles one aggregate root
3. **Interface Segregation**: Focused interfaces with clear responsibilities
4. **Open/Closed**: Easy to extend with new implementations

## Repository Interfaces

### IOrderRepository

Manages `Order` entities with operations for:

- Saving and updating orders
- Querying by ID, symbol, status, date range
- Finding active orders and orders by broker ID

**Key Methods:**

```python
async def save_order(self, order: Order) -> Order
async def get_order_by_id(self, order_id: UUID) -> Order | None
async def get_active_orders(self) -> list[Order]
async def update_order(self, order: Order) -> Order
```

### IPositionRepository

Manages `Position` entities with operations for:

- Saving and updating positions
- Querying by ID, symbol, strategy
- Finding active and closed positions
- Position lifecycle management

**Key Methods:**

```python
async def save_position(self, position: Position) -> Position
async def get_position_by_symbol(self, symbol: str) -> Position | None
async def get_active_positions(self) -> list[Position]
async def close_position(self, position_id: UUID) -> bool
```

### IPortfolioRepository

Manages `Portfolio` entities with operations for:

- Saving and updating portfolios
- Querying by ID, name, strategy
- Portfolio history and snapshots
- Current portfolio management

**Key Methods:**

```python
async def save_portfolio(self, portfolio: Portfolio) -> Portfolio
async def get_current_portfolio(self) -> Portfolio | None
async def create_portfolio_snapshot(self, portfolio: Portfolio) -> Portfolio
async def get_portfolio_history(self, portfolio_id: UUID, start_date: datetime, end_date: datetime) -> list[Portfolio]
```

## Transaction Management

### IUnitOfWork

Provides atomic operations across multiple repositories:

```python
async with unit_of_work:
    order = await unit_of_work.orders.save_order(order)
    position = await unit_of_work.positions.save_position(position)
    await unit_of_work.commit()
```

**Features:**

- Transaction lifecycle management
- Repository coordination
- Async context manager support
- Error handling and rollback

### ITransactionManager

High-level transaction patterns:

```python
async def trade_operation(uow: IUnitOfWork) -> Order:
    # Business logic using multiple repositories
    return await uow.orders.save_order(order)

result = await transaction_manager.execute_in_transaction(trade_operation)
```

**Patterns:**

- Execute with automatic transaction management
- Retry with exponential backoff
- Batch operations
- Error recovery

## Error Handling

Comprehensive exception hierarchy:

```python
# Base exceptions
RepositoryError          # Base for all repository errors
TransactionError         # Base for transaction errors

# Entity-specific exceptions
OrderNotFoundError       # Order not found
PositionNotFoundError    # Position not found
PortfolioNotFoundError   # Portfolio not found

# Operation exceptions
DuplicateEntityError     # Entity already exists
ConcurrencyError         # Concurrent modification
ValidationError          # Entity validation failed
IntegrityError          # Database constraint violation
```

## Usage Patterns

### Simple Repository Operations

```python
# Save an order
order = Order.create_market_order("AAPL", Decimal("100"), OrderSide.BUY)
async with unit_of_work:
    saved_order = await unit_of_work.orders.save_order(order)
    await unit_of_work.commit()
```

### Complex Multi-Repository Operations

```python
async def execute_trade(order_id: UUID, fill_price: Decimal) -> tuple[Order, Position]:
    async def trade_operation(uow: IUnitOfWork) -> tuple[Order, Position]:
        # Get order
        order = await uow.orders.get_order_by_id(order_id)
        order.fill(fill_quantity, fill_price)

        # Update/create position
        position = await uow.positions.get_position_by_symbol(order.symbol)
        if not position:
            position = Position.open_position(...)
        else:
            position.add_to_position(...)

        # Save changes atomically
        updated_order = await uow.orders.update_order(order)
        updated_position = await uow.positions.save_position(position)

        return updated_order, updated_position

    return await transaction_manager.execute_in_transaction(trade_operation)
```

### Query Operations

```python
# Get portfolio summary
portfolio = await unit_of_work.portfolios.get_current_portfolio()
active_positions = await unit_of_work.positions.get_active_positions()
recent_orders = await unit_of_work.orders.get_orders_by_date_range(start, end)
```

## Implementation Guidelines

### For Infrastructure Layer

When implementing these interfaces:

1. **Use async/await**: All operations are asynchronous
2. **Handle exceptions**: Map database errors to repository exceptions
3. **Maintain entity integrity**: Validate entities before persistence
4. **Support transactions**: Coordinate with Unit of Work
5. **Optimize queries**: Use efficient database operations
6. **Log operations**: Provide observability for debugging

### For Testing

Create mock implementations:

```python
class MockOrderRepository:
    def __init__(self):
        self._orders: dict[UUID, Order] = {}

    async def save_order(self, order: Order) -> Order:
        self._orders[order.id] = order
        return order

    async def get_order_by_id(self, order_id: UUID) -> Order | None:
        return self._orders.get(order_id)
```

## Database Integration

The interfaces are designed to work with:

- **PostgreSQL**: Primary database for transactional data
- **Connection pooling**: Efficient resource management
- **ACID transactions**: Data consistency guarantees
- **Optimistic concurrency**: Version-based conflict resolution

## Type Safety

All interfaces use:

- **Protocol types**: Structural typing for flexibility
- **Generic types**: Type-safe collections
- **Union types**: Nullable return values
- **Comprehensive typing**: Full mypy compatibility

## Performance Considerations

- **Async operations**: Non-blocking I/O
- **Batch operations**: Efficient bulk updates
- **Query optimization**: Indexed access patterns
- **Connection pooling**: Resource efficiency
- **Caching strategies**: Repository-level caching support

## Extension Points

Easy to extend with:

- **New repositories**: Additional aggregate roots
- **Different backends**: Redis, MongoDB, etc.
- **Caching layers**: Transparent caching
- **Audit trails**: Change tracking
- **Read replicas**: Query optimization
