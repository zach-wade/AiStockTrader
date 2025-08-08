# Utils Findings - Batch 51-60: Data Processing and Database Management

## Additional Security Vulnerability Found

### 🚨 Another Pickle Usage in DataProcessor

Found in `/src/main/utils/data/processor.py` line 285:
```python
elif format == 'pickle':
    if path:
        df.to_pickle(path, **kwargs)
    else:
        return base64.b64encode(pickle.dumps(df)).decode()  # SECURITY RISK!
```

This needs to be added to the security migration script and replaced with secure_serializer immediately.

## Data Processing Integration Opportunities

### 1. Replace Custom Data Processing

The data module provides comprehensive DataFrame operations that should replace custom implementations:

```python
from main.utils.data import DataProcessor, DataValidator, DataAnalyzer

# Global instances available
processor = get_global_processor()
validator = get_global_validator()
analyzer = get_global_analyzer()

# Standardize all market data
df = processor.standardize_market_data_columns(df, source='polygon')
df = processor.validate_ohlc_data(df)
df = processor.standardize_financial_timestamps(df)

# Validate data quality
validation_result = validator.validate_dataframe(df, rules=[
    DataValidationRule('open', 'positive'),
    DataValidationRule('volume', 'not_null'),
    DataValidationRule('timestamp', 'increasing')
])

if not validation_result.is_valid:
    logger.error(f"Data validation failed: {validation_result.errors}")

# Analyze for outliers
outliers = analyzer.detect_outliers(df, method='modified_z_score')
df_clean = df[~outliers.any(axis=1)]
```

### 2. Memory-Efficient Data Processing

Replace memory-intensive operations with chunked processing:

```python
# Before:
huge_df = pd.read_parquet('massive_file.parquet')
processed = process_all_at_once(huge_df)  # OOM risk!

# After:
processor = DataProcessor()
for chunk in processor.chunk_dataframe(huge_df, chunk_size=50000):
    processed_chunk = process_chunk(chunk)
    # Save or aggregate results
```

### 3. Data Validation Framework

Replace custom validation with the comprehensive validation system:

```python
# Add custom validators for market data
validator = DataValidator()

# Add market-specific validators
validator.add_custom_validator('valid_price', 
    lambda x: 0 < x < 100000 if pd.notna(x) else True
)

validator.add_custom_validator('valid_symbol',
    lambda x: bool(re.match(r'^[A-Z]{1,5}$', str(x)))
)

# Create validation rules
rules = [
    DataValidationRule('symbol', 'valid_symbol'),
    DataValidationRule('open', 'valid_price'),
    DataValidationRule('high', 'valid_price'),
    DataValidationRule('low', 'valid_price'),
    DataValidationRule('close', 'valid_price'),
    DataValidationRule('volume', 'positive'),
    DataValidationRule('timestamp', 'not_null')
]

# Apply validation
result = validator.validate_dataframe(df, rules)
```

## Database Pool Management Integration

### 1. Global Database Pool with Health Monitoring

Replace all custom database connections:

```python
from main.utils.database import (
    get_global_db_pool,
    PoolHealthMonitor,
    ConnectionPoolMetrics
)

# Use global pool everywhere
pool = get_global_db_pool()

# Add health monitoring
metrics_collector = MetricsCollector()
health_monitor = PoolHealthMonitor(metrics_collector)

# Regular health checks
async def monitor_db_health():
    pool_info = {
        'pool_size': pool.pool.size,
        'max_overflow': pool.pool.overflow,
        'active': pool.pool.checkedout()
    }
    
    health_status = health_monitor.assess_health(pool_info)
    
    if not health_status.is_healthy:
        logger.error(f"Database unhealthy: {health_status.warnings}")
        for recommendation in health_status.recommendations:
            logger.info(f"Recommendation: {recommendation}")
    
    # Check for connection leaks
    leak_check = health_monitor.check_connection_leaks(pool_info)
    if leak_check['potential_leaks']:
        logger.warning(f"Potential connection leaks: {leak_check['indicators']}")
```

### 2. Query Performance Tracking

Implement automatic query tracking:

```python
from main.utils.database import track_query, QueryType, QueryPriority

@track_query(query_type=QueryType.SELECT, priority=QueryPriority.HIGH)
async def get_market_data(symbol: str, start_date: datetime):
    query = """
        SELECT * FROM market_data 
        WHERE symbol = $1 AND timestamp >= $2
        ORDER BY timestamp
    """
    return await pool.fetch(query, symbol, start_date)

# Get performance metrics
tracker = get_global_tracker()
stats = tracker.get_statistics()
print(f"Slow queries: {stats['slow_queries']}")
print(f"Average query time: {stats['avg_execution_time']:.3f}s")
```

### 3. Batch Operations with Monitoring

Use monitored batch operations:

```python
from main.utils.database import batch_upsert, TransactionStrategy

# Batch upsert with monitoring
result = await batch_upsert(
    pool=pool,
    table='market_data',
    records=records,
    unique_columns=['symbol', 'timestamp'],
    batch_size=1000,
    strategy=TransactionStrategy.CHUNKED
)

logger.info(f"Inserted: {result.successful_count}, Failed: {result.failed_count}")
if result.errors:
    logger.error(f"Batch errors: {result.errors}")
```

## Implementation Priority Updates

### Immediate (Security Critical):
1. Fix pickle usage in DataProcessor
2. Add to security migration script
3. Replace with secure_serializer

### Week 1:
1. Implement global database pool with health monitoring
2. Add DataValidator to all data ingestion points
3. Use DataProcessor for all market data standardization

### Week 2:
1. Add query performance tracking
2. Implement connection leak detection
3. Replace custom data processing with utils

### Expected Benefits:
- **Security**: Eliminate another pickle vulnerability
- **Reliability**: Automatic database health monitoring
- **Performance**: Query tracking and optimization recommendations
- **Quality**: Comprehensive data validation
- **Memory**: Efficient chunked processing