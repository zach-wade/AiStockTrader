# Portfolio Thread Safety and Optimistic Locking Implementation

## Overview

This document summarizes the comprehensive thread safety and optimistic locking implementation for the Portfolio entity and related components in the trading system. The implementation addresses race conditions that could cause financial loss in a concurrent trading environment.

## Architecture Overview

The solution implements both **optimistic locking** at the domain level and **pessimistic locking** at the infrastructure level, following Domain-Driven Design (DDD) principles.

### Key Components

1. **Domain Exceptions** (`src/domain/exceptions.py`)
2. **Enhanced Portfolio Entity** (`src/domain/entities/portfolio.py`)
3. **Concurrency Service** (`src/domain/services/concurrency_service.py`)
4. **Thread-Safe Repository** (`src/infrastructure/repositories/portfolio_repository.py`)
5. **Comprehensive Tests** (unit and integration)

## Implementation Details

### 1. Domain-Level Exceptions

Created `src/domain/exceptions.py` with specialized concurrency exceptions:

- **`StaleDataException`**: Raised when entity version conflicts occur
- **`OptimisticLockException`**: Raised after maximum retries for version conflicts
- **`ConcurrencyException`**: Base exception for general concurrency issues
- **`PessimisticLockException`**: Raised when locks cannot be acquired
- **`DeadlockException`**: Raised when deadlocks are detected

### 2. Portfolio Entity Enhancements

Enhanced `src/domain/entities/portfolio.py` with:

- **Version Increment Methods**: `_increment_version()` automatically called on all mutating operations
- **Version Checking**: `_check_version()` validates expected vs actual versions
- **Optimistic Locking**: All state-changing methods increment version number
- **Automatic Timestamping**: `last_updated` field updated on version changes

#### Mutating Methods with Version Control

- `open_position()` - Increments version when opening new positions
- `close_position()` - Increments version when closing positions
- `update_position_price()` - Increments version on price updates
- `update_all_prices()` - Increments version once for batch updates

### 3. Concurrency Service

Created `src/domain/services/concurrency_service.py` providing:

- **Retry Logic**: Exponential backoff with jitter for version conflicts
- **Locking Primitives**: Both sync and async entity locks
- **Deadlock Detection**: Automatic detection of common deadlock patterns
- **Metrics Tracking**: Monitoring of conflicts, retries, and timeouts
- **Configurable Parameters**: Customizable retry counts, delays, and backoff factors

#### Key Features

```python
# Retry with exponential backoff
service.retry_on_version_conflict(operation, "Portfolio", portfolio_id)

# Entity-level pessimistic locking
with service.entity_lock("Portfolio", portfolio_id):
    # Critical section
    pass

# Async version
async with service.async_entity_lock("Portfolio", portfolio_id):
    # Critical section
    pass
```

### 4. Enhanced Repository Implementation

Updated `src/infrastructure/repositories/portfolio_repository.py` with:

- **SELECT FOR UPDATE**: Pessimistic locking using `FOR UPDATE NOWAIT`
- **Atomic Updates**: `update_portfolio_atomic()` for specific field updates
- **Transaction Support**: Proper transaction boundaries for complex operations
- **Version Conflict Handling**: Automatic retry with concurrency service integration

#### New Repository Methods

- `get_portfolio_for_update()` - Retrieves portfolio with row-level lock
- `update_portfolio_atomic()` - Updates specific fields atomically
- `open_position_transactional()` - Opens positions within transactions
- `batch_update_portfolios()` - Updates multiple portfolios atomically

### 5. Database-Level Optimistic Locking

The implementation uses standard optimistic locking patterns:

```sql
-- Version check in UPDATE statements
UPDATE portfolios
SET field = %s, version = version + 1, last_updated = NOW()
WHERE id = %s AND version = %s
```

```sql
-- Pessimistic locking for critical sections
SELECT * FROM portfolios
WHERE id = %s
FOR UPDATE NOWAIT
```

## Thread Safety Guarantees

### At the Entity Level

1. **Version Consistency**: All mutating operations increment version number
2. **Atomicity**: Version and timestamp updates happen together
3. **Thread-Safe Operations**: Thread-safe version increment for concurrent access
4. **State Validation**: Version checks prevent stale data modifications

### At the Repository Level

1. **Optimistic Locking**: Automatic version conflict detection and retry
2. **Pessimistic Locking**: Row-level locks for critical operations
3. **Transaction Isolation**: Proper transaction boundaries
4. **Deadlock Recovery**: Automatic deadlock detection and retry

### At the Service Level

1. **Retry Mechanisms**: Configurable exponential backoff
2. **Lock Management**: Entity-level lock coordination
3. **Conflict Resolution**: Intelligent retry strategies
4. **Monitoring**: Comprehensive metrics and logging

## Usage Examples

### Basic Portfolio Operations

```python
# The entity automatically handles version increments
portfolio = Portfolio(name="Trading Portfolio", ...)

# This increments version automatically
request = PositionRequest(symbol="AAPL", quantity=100, entry_price=150.00)
portfolio.open_position(request)

# This also increments version
portfolio.update_position_price("AAPL", Decimal("155.00"))
```

### Repository-Level Operations

```python
# Optimistic locking with automatic retry
repository = PostgreSQLPortfolioRepository(adapter, concurrency_service)
updated_portfolio = await repository.update_portfolio(portfolio)

# Pessimistic locking
locked_portfolio = await repository.get_portfolio_for_update(portfolio_id)

# Atomic field updates
await repository.update_portfolio_atomic(
    portfolio_id,
    {"cash_balance": Decimal("95000")},
    expected_version=portfolio.version
)
```

### Using Concurrency Service

```python
service = ConcurrencyService(max_retries=5, base_delay=0.1)

# Automatic retry on version conflicts
def risky_operation():
    # ... operation that might fail due to version conflict
    return result

result = service.retry_on_version_conflict(
    risky_operation,
    "Portfolio",
    portfolio_id
)
```

## Error Handling

The implementation provides detailed error information:

```python
try:
    portfolio.open_position(request)
except StaleDataException as e:
    print(f"Version conflict: expected {e.expected_version}, got {e.actual_version}")
    # Handle stale data - typically reload and retry

except OptimisticLockException as e:
    print(f"Failed after {e.retries} retries")
    # Handle persistent conflicts - may need human intervention

except DeadlockException as e:
    print(f"Deadlock detected involving entities: {e.entities}")
    # Handle deadlock - typically retry with delay
```

## Performance Considerations

### Optimizations Implemented

1. **Batch Updates**: `update_all_prices()` increments version only once
2. **Atomic Operations**: Minimize lock duration with targeted updates
3. **Jittered Backoff**: Reduces thundering herd problems
4. **Lock Timeout**: Prevents indefinite blocking
5. **Metrics Collection**: Performance monitoring without overhead

### Scalability Features

1. **Per-Entity Locking**: Fine-grained locks reduce contention
2. **Configurable Retries**: Tunable for different load patterns
3. **Async Support**: Non-blocking operations for high throughput
4. **Connection Pooling**: Efficient database resource usage

## Testing

### Unit Tests (`tests/unit/domain/entities/test_portfolio_concurrency.py`)

- Version increment behavior
- Concurrent operation handling
- Thread safety validation
- Exception scenarios
- State consistency checks

### Integration Tests (`tests/integration/repositories/test_portfolio_repository_concurrency.py`)

- Database-level optimistic locking
- Pessimistic locking scenarios
- Version conflict handling
- Deadlock simulation
- Performance under load

## Configuration

### Concurrency Service Parameters

```python
ConcurrencyService(
    max_retries=3,           # Maximum retry attempts
    base_delay=0.1,          # Initial delay between retries (seconds)
    max_delay=5.0,           # Maximum delay between retries (seconds)
    backoff_factor=2.0,      # Exponential backoff multiplier
    jitter=True              # Add random jitter to delays
)
```

### Repository Configuration

```python
PostgreSQLPortfolioRepository(
    adapter=db_adapter,
    max_retries=3,           # Repository-level retries
    concurrency_service=service  # Optional custom service
)
```

## Monitoring and Metrics

The implementation provides comprehensive metrics:

```python
metrics = concurrency_service.get_metrics()
# Returns:
{
    "version_conflicts": 42,
    "successful_retries": 38,
    "failed_retries": 4,
    "deadlocks_detected": 1,
    "lock_timeouts": 2
}
```

## Security Considerations

1. **Input Validation**: All parameters validated before processing
2. **SQL Injection Prevention**: Parameterized queries throughout
3. **Resource Limits**: Bounded retry counts and delays
4. **Audit Logging**: All concurrency events logged
5. **Version Tampering**: Version numbers protected from external modification

## Future Enhancements

### Potential Improvements

1. **Distributed Locking**: Redis-based locks for multi-instance deployments
2. **Lock Escalation**: Automatic escalation to table-level locks
3. **Adaptive Backoff**: Machine learning-based retry strategies
4. **Event Sourcing**: Complete audit trail of all changes
5. **Read Replicas**: Separate read/write paths for better performance

### Monitoring Enhancements

1. **Real-time Dashboards**: Live concurrency metrics
2. **Alert Thresholds**: Notifications for high conflict rates
3. **Performance Profiling**: Detailed timing analysis
4. **Capacity Planning**: Predictive scaling based on patterns

## Conclusion

This implementation provides enterprise-grade thread safety and concurrency control for the Portfolio entity, following industry best practices:

- **Multiple Defense Layers**: Entity, repository, and service-level protections
- **Comprehensive Error Handling**: Detailed exceptions with recovery guidance
- **Performance Optimized**: Minimal overhead with maximum safety
- **Thoroughly Tested**: Extensive unit and integration test coverage
- **Production Ready**: Monitoring, logging, and configuration support

The solution eliminates race conditions while maintaining high performance and providing clear error handling for edge cases. It scales well with load and provides the reliability required for financial trading systems.
