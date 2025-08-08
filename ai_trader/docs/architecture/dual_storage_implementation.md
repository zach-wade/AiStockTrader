# Dual Storage Implementation Architecture

## Overview

This document describes the current implementation of the dual storage system in the AI Trader platform. The system implements a "Write to Both, Reader Decides, Janitor Cleans" pattern, where data is written simultaneously to both hot storage (PostgreSQL) and cold storage (Data Lake), with readers choosing their data source based on query patterns.

## Architecture Pattern

### Core Principle: Write to Both, Reader Decides, Janitor Cleans

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Repository    │────▶│ DualStorageWriter│────▶│  Hot Storage    │
│ (11 integrated) │     │                  │     │  (PostgreSQL)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                 │
                                 │ Async/Sync
                                 ▼
                        ┌──────────────────┐
                        │    Event Bus     │
                        └──────────────────┘
                                 │
                                 │ DataWrittenEvent
                                 ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │ ColdStorageConsumer│────▶│  Cold Storage   │
                        └──────────────────┘     │  (Data Lake)    │
                                                 └─────────────────┘
```

## Components

### 1. DualStorageWriter

The central component that manages writes to both storage tiers.

**Location**: `src/main/data_pipeline/storage/dual_storage_writer.py`

**Key Features**:
- Synchronous and asynchronous write modes
- Circuit breaker pattern for fault tolerance
- Metrics tracking for monitoring
- Event publishing for cold storage consistency

**Write Modes**:
- `SYNC`: Write to both hot and cold storage synchronously
- `ASYNC`: Write to hot storage sync, cold storage async
- `EVENT_ONLY`: Write to hot storage, publish event for cold storage

### 2. Repository Integration

All 11 data repositories have been integrated with dual storage support:

| Repository | Storage Attribute | Table Name | Status |
|------------|------------------|------------|---------|
| MarketDataRepository | `_dual_storage_writer` | market_data | ✅ Integrated |
| NewsRepository | `_dual_storage_writer` | news_data | ✅ Integrated |
| CompanyRepository | `_dual_storage_writer` | companies | ✅ Integrated |
| CryptocurrencyRepository | `dual_storage_writer` | cryptocurrencies | ✅ Integrated |
| DividendsRepository | `_dual_storage_writer` | corporate_actions | ✅ Integrated |
| FinancialsRepository | `_dual_storage_writer` | financials_data | ✅ Integrated |
| GuidanceRepository | `_dual_storage_writer` | company_guidance | ✅ Integrated |
| RatingsRepository | `_dual_storage_writer` | analyst_ratings | ✅ Integrated |
| SentimentRepository | `dual_storage_writer` | social_sentiment | ✅ Integrated |
| SocialSentimentRepository | `dual_storage_writer` | social_sentiment_data | ✅ Integrated |
| FeatureRepository | `dual_storage_writer` | feature_store | ✅ Integrated |

### 3. Event System

**DataWrittenEvent**: Published after successful hot storage writes

```python
@dataclass
class DataWrittenEvent(Event):
    table_name: str
    record_count: int
    record_ids: List[str]
    timestamp: datetime
    metadata: Dict[str, Any]
```

**ColdStorageConsumer**: Processes events to write to cold storage
- Handles batch processing
- Implements retry logic
- Tracks processing metrics

### 4. Circuit Breaker Protection

Each storage tier has independent circuit breakers:
- **Hot Storage Circuit Breaker**: Protects PostgreSQL
- **Cold Storage Circuit Breaker**: Protects Data Lake

States:
- `CLOSED`: Normal operation
- `OPEN`: Failures exceeded threshold, requests blocked
- `HALF_OPEN`: Testing if service recovered

## Implementation Details

### Repository Pattern

Each repository overrides `bulk_upsert` to use dual storage:

```python
async def bulk_upsert(self, records: List[Dict[str, Any]], 
                     update_fields: Optional[List[str]] = None, 
                     **kwargs) -> OperationResult:
    if self.dual_storage_writer:
        # Use dual storage writer for hot/cold writes
        write_results = await self.dual_storage_writer.write_batch(
            table_name=self.get_table_name(),
            records=records,
            upsert=True
        )
        
        # Aggregate results from List[WriteResult]
        success_count = sum(1 for r in write_results if r.success)
        failed_count = len(write_results) - success_count
        total_rows = sum(r.row_count for r in write_results)
        
        # Create operation result
        result = OperationResult(
            success=failed_count == 0,
            total_records=total_rows,
            processed_records=success_count,
            failed_records=failed_count,
            operation_type="dual_storage_bulk_upsert"
        )
        
        return result
    else:
        # Fall back to standard BaseRepository bulk_upsert
        return await super().bulk_upsert(records, update_fields, **kwargs)
```

### Attribute Naming Pattern

Due to historical implementation, repositories use two patterns for storing the dual storage writer:
- `self.dual_storage_writer` (newer repositories)
- `self._dual_storage_writer` (older repositories)

Both patterns work correctly; the factory checks for both when verifying dual storage support.

### Configuration

Dual storage is configured through `RepositoryFactory`:

```python
factory = RepositoryFactory(
    db_adapter=db_adapter,
    cold_storage=cold_storage,  # DataArchive instance
    event_bus=event_bus         # EventBus instance
)
```

When both `cold_storage` and `event_bus` are provided, the factory automatically creates a `DualStorageWriter` instance for each repository.

## Data Flow

### Write Flow

1. **Application calls repository method** (e.g., `bulk_upsert`)
2. **Repository checks for dual_storage_writer**
3. **DualStorageWriter.write_batch() called**:
   - Writes to hot storage (PostgreSQL) via CrudExecutor
   - Based on write mode:
     - SYNC: Writes to cold storage synchronously
     - ASYNC: Creates async task for cold storage
     - EVENT_ONLY: Skips direct cold write
   - Publishes DataWrittenEvent
4. **ColdStorageConsumer processes event**:
   - Reads event from queue
   - Writes to cold storage (Data Lake)
   - Updates processing metrics

### Read Flow

Currently, all reads go through hot storage (PostgreSQL). Future enhancements will add:
- Query routing based on data age
- Cold storage direct queries for historical data
- Unified query interface

## Metrics and Monitoring

### Available Metrics

The DualStorageWriter tracks:
- `hot_writes_success`: Successful writes to PostgreSQL
- `hot_writes_failed`: Failed writes to PostgreSQL
- `cold_writes_success`: Successful writes to Data Lake
- `cold_writes_failed`: Failed writes to Data Lake
- `events_published`: Number of events published
- `hot_success_rate`: Calculated success rate for hot storage
- `cold_success_rate`: Calculated success rate for cold storage
- `hot_circuit_breaker_state`: Current state of hot storage circuit breaker
- `cold_circuit_breaker_state`: Current state of cold storage circuit breaker

### Accessing Metrics

```python
# From a repository
metrics = repository.get_dual_storage_metrics()

# From app context
health = await app_context.get_dual_storage_health()
```

## Error Handling

### Failure Scenarios

1. **Hot Storage Failure**:
   - Write fails immediately
   - No attempt to write to cold storage
   - Circuit breaker may open

2. **Cold Storage Failure**:
   - Hot storage write succeeds
   - Event still published
   - Cold storage consumer retries
   - Circuit breaker may open

3. **Event Bus Failure**:
   - Hot storage write succeeds
   - Cold storage async write may succeed
   - Event publishing logged as failed

### Recovery Mechanisms

- **Circuit Breakers**: Automatically recover when service healthy
- **Event Replay**: Failed events can be replayed from hot storage
- **Manual Sync**: Janitor process can sync missing data

## Adding Dual Storage to New Repositories

To add dual storage support to a new repository:

1. **Add constructor parameters**:
```python
def __init__(self, db_adapter: IAsyncDatabase, 
             config: Optional[RepositoryConfig] = None,
             dual_storage_writer: Optional[Any] = None,
             cold_storage: Optional[Any] = None,
             event_bus: Optional[Any] = None):
    super().__init__(db_adapter, ModelClass, config)
    
    # Store dual storage components
    self.dual_storage_writer = dual_storage_writer
    self.cold_storage = cold_storage
    self.event_bus = event_bus
```

2. **Override bulk_upsert method** (see pattern above)

3. **Register in RepositoryFactory**:
```python
self._registry = {
    # ... existing repositories ...
    'new_repo': NewRepository,
}
```

## Testing

### Integration Tests

Comprehensive tests are available in:
- `tests/integration/test_dual_storage_integration.py`
- `tests/integration/test_dual_storage_complete.py`

Tests verify:
- All repositories have dual storage support
- Data flows to both storage tiers
- Events are published correctly
- Circuit breakers function properly
- Concurrent writes work correctly

### Running Tests

```bash
# Run all dual storage tests
pytest tests/integration/test_dual_storage_complete.py -v

# Run with real database
pytest tests/integration/test_dual_storage_complete.py -v --db-real
```

## Future Enhancements

See `docs/enhancements/event-driven-storage-architecture.md` for planned improvements:
- Write-Ahead Log (WAL) pattern
- Event sourcing capabilities
- Advanced query routing
- Data lifecycle automation
- Real-time synchronization