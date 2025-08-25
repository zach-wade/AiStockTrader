# High-Frequency Trading Database Index Optimization

This directory contains advanced database indexing optimizations designed for high-frequency trading workloads supporting 1000+ orders/sec with sub-millisecond query response times.

## 📁 File Structure

```
src/infrastructure/database/
├── advanced_indexes.sql      # Advanced trading-specific indexes
├── performance_queries.sql   # Performance analysis and monitoring queries
├── index_maintenance.sql     # Automated maintenance procedures
├── query_optimization.sql    # Optimized query functions and helpers
└── README.md                 # This documentation
```

## 🚀 Quick Start

### 1. Apply Advanced Indexes

```sql
-- Apply all advanced indexes for trading performance
\i src/infrastructure/database/advanced_indexes.sql
```

### 2. Set Up Performance Monitoring

```sql
-- Load performance monitoring queries
\i src/infrastructure/database/performance_queries.sql

-- Check current performance metrics
SELECT * FROM v_index_usage_stats;
```

### 3. Configure Automated Maintenance

```sql
-- Load maintenance procedures
\i src/infrastructure/database/index_maintenance.sql

-- Set up automated maintenance (if using pg_cron)
SELECT cron.schedule('trading-maintenance', '*/5 9-16 * * 1-5',
    'SELECT safe_trading_hours_maintenance();');
```

### 4. Deploy Query Optimizations

```sql
-- Apply optimized query functions
\i src/infrastructure/database/query_optimization.sql

-- Test optimized functions
SELECT * FROM get_active_orders_by_symbol('AAPL', 100);
```

## 📊 Performance Targets

| Operation | Target Response Time | Index Strategy |
|-----------|---------------------|----------------|
| Order lookups | < 1ms | Covering indexes + partial indexes |
| Position calculations | < 2ms | Unique active position indexes |
| Market data queries | < 5ms | Time-series optimized indexes |
| Portfolio aggregations | < 10ms | Materialized views + composite indexes |
| Batch operations | > 1000 ops/sec | Optimized batch functions |

## 🔧 Index Categories

### 1. Order Management Indexes

**Critical for HFT order processing**

```sql
-- Most important: Active orders by symbol and status
idx_orders_symbol_status_created_active

-- Broker order ID lookups (unique constraint)
idx_orders_broker_id_unique

-- Order book price-level matching
idx_orders_symbol_side_limit_price_active
```

### 2. Position Tracking Indexes

**Real-time position management**

```sql
-- One active position per symbol (unique)
idx_positions_symbol_active_unique

-- P&L calculations with price joins
idx_positions_pnl_calculation

-- Risk exposure calculations
idx_positions_risk_exposure
```

### 3. Market Data Indexes

**High-performance time-series access**

```sql
-- Latest price lookups (most critical)
idx_market_data_symbol_timeframe_latest

-- Multi-symbol batch price queries
idx_market_data_latest_prices

-- Real-time streaming data (last 1 hour)
idx_market_data_realtime_stream
```

### 4. Risk Management Indexes

**Real-time risk calculations**

```sql
-- Portfolio exposure by strategy
idx_risk_portfolio_exposure

-- Symbol concentration risk
idx_risk_symbol_concentration

-- Leverage calculations
idx_risk_leverage_calculation
```

## 📈 Performance Monitoring

### Real-Time Monitoring (Every Minute)

```sql
-- Check current database activity
SELECT pid, state, query_duration_seconds, query_preview
FROM (SELECT * FROM pg_stat_activity WHERE state = 'active') AS active_queries;

-- Monitor lock contention
SELECT * FROM pg_locks WHERE NOT granted;
```

### High-Frequency Monitoring (Every 5 Minutes)

```sql
-- Order processing performance
SELECT
    COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '5 minutes') as orders_last_5min,
    AVG(EXTRACT(EPOCH FROM (submitted_at - created_at)) * 1000) as avg_submission_latency_ms
FROM orders;

-- Index hit ratios
SELECT indexrelname,
       ROUND(100.0 * idx_blks_hit / (idx_blks_hit + idx_blks_read), 2) as hit_ratio_percent
FROM pg_stat_user_indexes pui
JOIN pg_statio_user_indexes psui ON pui.indexrelid = psui.indexrelid
WHERE schemaname = 'public' AND (idx_blks_hit + idx_blks_read) > 0
ORDER BY hit_ratio_percent ASC;
```

### Daily Analysis

```sql
-- Unused index analysis
SELECT * FROM v_unused_indexes;

-- Performance summary dashboard
SELECT 'Index Hit Ratio' as metric,
       ROUND(100.0 * SUM(idx_blks_hit) / GREATEST(SUM(idx_blks_hit + idx_blks_read), 1), 2)::text || '%' as value
FROM pg_statio_user_indexes;
```

## 🔧 Maintenance Procedures

### During Trading Hours

**Safe operations only (< 100ms impact)**

```sql
-- Fast statistics updates for critical tables
SELECT * FROM safe_trading_hours_maintenance();

-- Quick performance checks
SELECT * FROM detect_performance_issues() WHERE severity = 'CRITICAL';
```

### Maintenance Window (2-6 AM)

**Full maintenance operations**

```sql
-- Comprehensive table analysis
SELECT * FROM auto_analyze_tables();

-- Index rebuilding for fragmented indexes
SELECT * FROM rebuild_fragmented_indexes();

-- Generate maintenance recommendations
SELECT * FROM generate_maintenance_recommendations();
```

### Emergency Procedures

**When performance degrades**

```sql
-- Kill long-running queries (>2 minutes)
SELECT * FROM emergency_kill_long_queries(120);

-- Check for blocking locks
SELECT blocked_pid, blocking_pid, blocked_statement, blocking_statement
FROM pg_catalog.pg_locks blocked_locks
-- [full lock analysis query from performance_queries.sql]
```

## 🎯 Query Optimization Functions

### High-Performance Order Operations

```sql
-- Get active orders for a symbol (< 1ms target)
SELECT * FROM get_active_orders_by_symbol('AAPL', 100);

-- Ultra-fast order status update (< 1ms target)
SELECT update_order_status_fast('BROKER_12345', 'filled', 100.0, 150.25);

-- Batch order processing (> 1000 ops/sec target)
SELECT * FROM process_order_batch('[
    {"symbol": "AAPL", "side": "buy", "quantity": "100", "limit_price": "150.00"},
    {"symbol": "GOOGL", "side": "sell", "quantity": "50", "limit_price": "2800.00"}
]'::jsonb);
```

### Position Management with P&L

```sql
-- Real-time position with P&L calculation (< 2ms target)
SELECT * FROM get_position_with_pnl('AAPL', 155.75);

-- Portfolio exposure summary (< 10ms target)
SELECT * FROM get_portfolio_exposure_summary();
```

### Market Data Access

```sql
-- Latest prices for multiple symbols (< 5ms target)
SELECT * FROM get_latest_prices(ARRAY['AAPL', 'GOOGL', 'MSFT'], '1min');

-- Price history for technical analysis
SELECT * FROM get_price_history('AAPL', '5min', 100);
```

## 📋 Materialized Views

### Real-Time Trading Metrics

```sql
-- Refresh every 5 minutes during trading hours
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_realtime_trading_metrics;

-- Query real-time metrics (< 1ms)
SELECT * FROM mv_realtime_trading_metrics;
```

### Position Summary

```sql
-- Refresh when positions change
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_position_summary;

-- Query position summaries (< 1ms)
SELECT * FROM mv_position_summary WHERE total_exposure > 100000;
```

### Smart Refresh Strategy

```sql
-- Intelligent refresh based on data changes
SELECT * FROM refresh_materialized_views(false);  -- Only refresh if needed
SELECT * FROM refresh_materialized_views(true);   -- Force refresh all
```

## 🚨 Alerting Thresholds

### Critical Alerts (Immediate Action Required)

- Index hit ratio < 95%
- Query execution time > 10ms for critical operations
- Active lock count > 10
- Dead tuple ratio > 50%

### Warning Alerts (Monitor Closely)

- Index hit ratio < 98%
- Query execution time > 5ms for critical operations
- Active lock count > 5
- Dead tuple ratio > 25%

### Performance Degradation Signs

- Sequential scans on large tables
- Index bloat > 50%
- Long-running queries > 60 seconds
- Connection pool exhaustion

## 🔍 Troubleshooting Guide

### Slow Order Lookups

1. Check index usage: `SELECT * FROM v_index_usage_stats WHERE tablename = 'orders';`
2. Verify query plans: `EXPLAIN (ANALYZE) SELECT * FROM orders WHERE symbol = 'AAPL';`
3. Update statistics: `ANALYZE orders;`
4. Rebuild fragmented indexes if needed

### High Lock Contention

1. Identify blocking queries: Use lock analysis queries from `performance_queries.sql`
2. Kill long-running queries: `SELECT * FROM emergency_kill_long_queries(60);`
3. Review transaction boundaries in application code
4. Consider READ COMMITTED isolation for read operations

### Memory Issues

1. Check buffer hit ratios: Should be > 99% for indexes, > 95% for tables
2. Monitor working memory usage: `SHOW work_mem;`
3. Adjust PostgreSQL configuration: `shared_buffers`, `effective_cache_size`
4. Consider connection pooling optimization

### Index Bloat

1. Identify bloated indexes: `SELECT * FROM v_unused_indexes;`
2. Rebuild indexes: `REINDEX INDEX CONCURRENTLY index_name;`
3. Schedule regular maintenance: Use functions from `index_maintenance.sql`
4. Monitor index growth patterns

## 📊 Performance Testing

Run the integration tests to validate performance:

```bash
# Test index performance
pytest tests/integration/database/test_index_performance.py -v

# Test query optimization
pytest tests/integration/database/test_query_optimization.py -v

# Full performance test suite
pytest tests/integration/database/ -v --asyncio-mode=auto
```

### Expected Test Results

- Order lookups: < 1ms (99th percentile)
- Position calculations: < 2ms (99th percentile)
- Market data queries: < 5ms (99th percentile)
- Index hit ratio: > 99%
- Concurrent throughput: > 1000 ops/sec

## 🔧 Configuration Recommendations

### PostgreSQL Settings for HFT

```sql
-- Connection and memory settings
max_connections = 200
shared_buffers = 25% of RAM
effective_cache_size = 75% of RAM
work_mem = 256MB
maintenance_work_mem = 2GB

-- Checkpoint and WAL settings
checkpoint_timeout = 5min
checkpoint_completion_target = 0.9
wal_buffers = 64MB
wal_compression = on

-- Query planner settings
random_page_cost = 1.1  -- For SSD storage
effective_io_concurrency = 200
```

### Index Maintenance Schedule

```sql
-- Every 5 minutes during trading hours (9 AM - 4 PM EST, Mon-Fri)
SELECT cron.schedule('trading-maintenance', '*/5 9-16 * * 1-5',
    'SELECT safe_trading_hours_maintenance();');

-- Daily at 3 AM EST
SELECT cron.schedule('daily-maintenance', '0 3 * * *',
    'SELECT auto_analyze_tables(); SELECT update_critical_table_stats();');

-- Weekly on Sunday at 2 AM EST
SELECT cron.schedule('weekly-maintenance', '0 2 * * 0',
    'SELECT rebuild_fragmented_indexes();');
```

## 📚 Additional Resources

- [PostgreSQL Performance Tuning Guide](https://wiki.postgresql.org/wiki/Performance_Optimization)
- [Index Usage Patterns](https://use-the-index-luke.com/)
- [High-Frequency Trading Database Design](https://www.postgresql.org/docs/current/indexes.html)

## 🤝 Contributing

When adding new indexes or optimizations:

1. **Measure First**: Always benchmark existing performance
2. **Test Impact**: Validate with integration tests
3. **Monitor Usage**: Track index effectiveness with monitoring queries
4. **Document Changes**: Update this README with new procedures
5. **Review Maintenance**: Ensure new indexes are included in maintenance procedures

## 📞 Support

For performance issues or optimization questions:

1. Run diagnostic queries from `performance_queries.sql`
2. Check maintenance recommendations: `SELECT * FROM generate_maintenance_recommendations();`
3. Review recent performance trends
4. Escalate critical performance degradation immediately
