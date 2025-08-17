# Unified Monitoring System

## Overview

The utils monitoring system is the centralized monitoring solution for the AI Trader project. It provides a single source of truth for all monitoring needs including metrics collection, alerting, performance tracking, and health reporting.

## Architecture

```
utils/monitoring/
├── __init__.py          # Main exports and convenience functions
├── types.py             # Common types (MetricType, AlertLevel, etc.)
├── monitor.py           # Base PerformanceMonitor implementation
├── enhanced.py          # Enhanced monitor with DB persistence
├── migration.py         # Migration layer for backward compatibility
├── global_monitor.py    # Global monitor instance management
├── collectors.py        # System metrics collectors
├── alerts.py            # Alert management
├── function_tracker.py  # Function performance tracking
├── memory.py            # Memory profiling
├── dashboard_adapters.py # Adapters for dashboard integration
└── README.md            # This file
```

## Features

### Basic Features (Always Available)

- In-memory metrics storage
- System resource monitoring (CPU, memory, disk, network)
- Function performance tracking
- Basic alerting
- Real-time metric recording

### Enhanced Features (With Database)

- Persistent metric storage
- Advanced aggregation queries
- Threshold-based automatic alerting
- Metric retention policies
- Time-series data queries
- Health scoring

## Usage

### Basic Usage

```python
from main.utils.monitoring import record_metric, timer

# Record a simple metric
record_metric("api.request_count", 1)
record_metric("api.response_time", 150.5, tags={"endpoint": "/users"})

# Time a function
@timer("process_data")
def process_data():
    # Your code here
    pass
```

### Dashboard Integration

```python
from main.utils.monitoring.dashboard_adapters import create_dashboard_adapter

# Create adapter (automatically uses enhanced features if DB available)
adapter = create_dashboard_adapter(db_pool)

# Get system health
health = await adapter.get_system_health_score()

# Get metric values
cpu_avg = await adapter.get_metric_value("system.cpu_usage", "avg", period_minutes=60)

# Get time series
series = await adapter.get_metric_series("api.request_count", period_minutes=120)
```

### Enhanced Features

```python
from main.utils.monitoring.migration import create_monitor

# Create monitor with database support
monitor = create_monitor(db_pool=db_pool)

# Register metric with thresholds
monitor.register_metric_with_thresholds(
    name="api.error_rate",
    warning_threshold=0.05,  # 5% error rate warning
    critical_threshold=0.10,  # 10% error rate critical
    description="API error rate percentage"
)

# Query aggregated metrics
metrics = await monitor.get_aggregated_metrics(
    names=["api.request_count", "api.error_count"],
    aggregation="sum",
    period_minutes=60,
    group_by_interval=5  # 5-minute buckets
)
```

## Migration Guide

### From Old Monitoring Module

If you're using imports from `main.monitoring.*`, migrate to `main.utils.monitoring`:

```python
# OLD
from main.monitoring.metrics.unified_metrics import UnifiedMetrics
from main.monitoring.alerts.unified_alerts import AlertManager

# NEW
from main.utils.monitoring.dashboard_adapters import create_dashboard_adapter
adapter = create_dashboard_adapter(db_pool)
```

### From Basic record_metric

No changes needed! The system is fully backward compatible:

```python
# This still works exactly the same
from main.utils.monitoring import record_metric
record_metric("my_metric", 42.0)
```

## Environment Variables

- `USE_ENHANCED_MONITORING`: Control enhanced features
  - `true`: Always use enhanced features (requires DB)
  - `false`: Use only basic features
  - `auto` (default): Auto-detect based on DB availability

## Best Practices

1. **Metric Naming**: Use dot notation for hierarchical metrics
   - Good: `api.users.request_count`
   - Bad: `api_users_request_count`

2. **Tags**: Use tags for dimensions that you want to filter by

   ```python
   record_metric("api.request_count", 1, tags={
       "endpoint": "/users",
       "method": "GET",
       "status": "200"
   })
   ```

3. **Thresholds**: Set appropriate thresholds for critical metrics

   ```python
   monitor.register_metric_with_thresholds(
       "database.connection_pool.usage",
       warning_threshold=80,   # 80% pool usage
       critical_threshold=95   # 95% pool usage
   )
   ```

4. **Aggregations**: Choose the right aggregation for your metric type
   - Counters: Use `sum`
   - Gauges: Use `avg`, `min`, `max`
   - Timers: Use `avg`, `p95`, `p99`

## Performance Considerations

1. **In-Memory Storage**: Limited by `history_size` (default 10,000 per metric)
2. **Database Storage**: Automatically batches inserts for efficiency
3. **Aggregations**: Pre-computed hourly for common queries
4. **Cleanup**: Old metrics are automatically cleaned up based on retention policy

## Database Schema

When using enhanced features, the following tables are created:

```sql
-- Raw metrics storage
CREATE TABLE metrics (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    tags JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Pre-computed aggregations
CREATE TABLE metric_aggregations (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    aggregation_type VARCHAR(50) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    sample_count INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Troubleshooting

### Metrics Not Persisting

- Check database connection: Ensure `db_pool` is provided
- Check logs for persistence errors
- Verify table creation permissions

### High Memory Usage

- Reduce `history_size` in monitor initialization
- Enable database persistence to offload to storage
- Use retention policies to clean old metrics

### Missing Metrics

- Metrics are auto-registered on first use
- Check metric name spelling and consistency
- Verify tags match when querying

## Future Enhancements

1. **Prometheus Export**: Export metrics in Prometheus format
2. **Grafana Integration**: Direct Grafana datasource support
3. **Machine Learning**: Anomaly detection on metrics
4. **Distributed Tracing**: OpenTelemetry integration
