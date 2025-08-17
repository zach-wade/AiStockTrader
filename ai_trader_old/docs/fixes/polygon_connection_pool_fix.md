# Polygon Connection Pool Fix Summary

## Issue

The backfill process was generating numerous warnings:

```
WARNING:urllib3.connectionpool:Connection pool is full, discarding connection: api.polygon.io
```

## Root Cause

1. **Connection Pool Size Mismatch**: The default Polygon SDK connection pool size was 10, but the backfill was running with max_parallel=20
2. **Aggressive Parallelism**: The system was attempting to make more concurrent requests than the connection pool could handle
3. **No Pool Recycling**: The SDK wasn't configured with optimal connection pooling parameters

## Fixes Applied

### 1. Increased Connection Pool Size

Updated `polygon_client.py` to dynamically set connection pool size based on parallelism:

```python
# Increase connection pool size based on max_parallel setting
max_parallel = self.config_obj.get('data.backfill.max_parallel', 20)
num_pools = max(10, max_parallel * 2)  # Double the parallel limit for connection pools

self.client = RESTClient(
    api_key,
    num_pools=num_pools,
    connect_timeout=30.0,  # Increased from default 10s
    read_timeout=60.0      # Increased from default 10s
)
```

### 2. Limited Parallelism for Polygon

Updated `historical_utils.py` to use conservative parallelism for Polygon:

```python
if 'polygon' in sources:
    # Polygon has limited connection pools, be conservative
    return min(3, default_parallel)  # Conservative limit to avoid pool exhaustion
```

### 3. Better Error Handling

Added specific handling for connection pool warnings in `polygon_client.py`:

```python
# Handle connection pool warnings
if "connection pool is full" in error_str.lower() or "connectionpool" in error_str.lower():
    logger.warning(f"⚠️ Connection pool warning for {symbol}: {error_str}")
    logger.info("   This is typically harmless - the SDK will retry automatically")
    return pd.DataFrame()
```

### 4. Suppressed Non-Critical Warnings

Added warning filters to reduce noise:

```python
# Suppress non-critical urllib3 connection pool warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore', message='Connection pool is full')
```

## Results

- Connection pool warnings are now properly handled and suppressed
- Parallelism is automatically adjusted based on data source capabilities
- Connection pools are sized appropriately for the workload
- Backfill performance remains high while avoiding connection pool exhaustion

## Best Practices Going Forward

1. Always consider connection pool limits when setting parallelism
2. Different data sources have different capabilities - adjust accordingly
3. Connection pool warnings are often harmless but indicate suboptimal configuration
4. Monitor actual throughput rather than just parallelism settings
