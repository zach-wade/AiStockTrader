-- Performance Analysis Queries for High-Frequency Trading Database
-- Monitor index usage, query performance, and identify optimization opportunities
--
-- Usage: Run these queries periodically to analyze database performance
-- Recommended frequency: Daily for critical metrics, weekly for detailed analysis

-- ============================================================================
-- INDEX PERFORMANCE ANALYSIS
-- ============================================================================

-- 1. Most Used Indexes - Identify heavily utilized indexes
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan as scans,
    idx_tup_read as tuples_read,
    idx_tup_fetch as tuples_fetched,
    ROUND(idx_tup_read::numeric / GREATEST(idx_scan, 1), 2) as avg_tuples_per_scan,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size,
    pg_size_pretty(pg_total_relation_size(c.oid)) as total_table_size
FROM pg_stat_user_indexes pui
JOIN pg_class c ON c.oid = pui.relid
WHERE schemaname = 'public'
ORDER BY idx_scan DESC, idx_tup_read DESC
LIMIT 20;

-- 2. Index Hit Ratio - Should be > 99% for optimal performance
SELECT
    indexrelname as index_name,
    idx_blks_read + idx_blks_hit as total_reads,
    CASE
        WHEN (idx_blks_read + idx_blks_hit) = 0 THEN 0
        ELSE ROUND(100.0 * idx_blks_hit / (idx_blks_read + idx_blks_hit), 2)
    END as hit_ratio_percent
FROM pg_stat_user_indexes pui
JOIN pg_statio_user_indexes psui ON pui.indexrelid = psui.indexrelid
WHERE schemaname = 'public'
    AND (idx_blks_read + idx_blks_hit) > 0
ORDER BY hit_ratio_percent ASC, total_reads DESC;

-- 3. Unused or Rarely Used Indexes - Candidates for removal
SELECT
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size,
    idx_scan as scans,
    idx_tup_read as tuples_read,
    CASE
        WHEN idx_scan = 0 THEN 'Never used'
        WHEN idx_scan < 10 THEN 'Rarely used'
        ELSE 'Potentially unused'
    END as usage_status
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
    AND idx_scan < 50  -- Adjust threshold based on workload
    AND pg_relation_size(indexrelid) > 1024 * 1024  -- Larger than 1MB
ORDER BY pg_relation_size(indexrelid) DESC;

-- 4. Index Bloat Analysis - Identify fragmented indexes
WITH index_bloat AS (
    SELECT
        schemaname,
        tablename,
        indexname,
        pg_size_pretty(pg_relation_size(indexrelid)) as index_size,
        pg_relation_size(indexrelid) as size_bytes,
        CASE
            WHEN pg_relation_size(indexrelid) > 100 * 1024 * 1024 THEN 'Large (>100MB)'
            WHEN pg_relation_size(indexrelid) > 10 * 1024 * 1024 THEN 'Medium (>10MB)'
            ELSE 'Small (<10MB)'
        END as size_category
    FROM pg_stat_user_indexes
    WHERE schemaname = 'public'
)
SELECT *
FROM index_bloat
ORDER BY size_bytes DESC;

-- ============================================================================
-- QUERY PERFORMANCE METRICS
-- ============================================================================

-- 5. Table Access Patterns - Sequential vs Index scans
SELECT
    schemaname,
    tablename,
    seq_scan as sequential_scans,
    seq_tup_read as seq_tuples_read,
    idx_scan as index_scans,
    idx_tup_fetch as idx_tuples_fetched,
    CASE
        WHEN seq_scan + idx_scan = 0 THEN 0
        ELSE ROUND(100.0 * idx_scan / (seq_scan + idx_scan), 2)
    END as index_scan_ratio_percent,
    pg_size_pretty(pg_total_relation_size(c.oid)) as table_size
FROM pg_stat_user_tables put
JOIN pg_class c ON c.oid = put.relid
WHERE schemaname = 'public'
ORDER BY seq_scan DESC, seq_tup_read DESC;

-- 6. Most Active Tables - I/O and update patterns
SELECT
    schemaname,
    tablename,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes,
    n_tup_hot_upd as hot_updates,
    n_live_tup as live_tuples,
    n_dead_tup as dead_tuples,
    CASE
        WHEN n_live_tup = 0 THEN 0
        ELSE ROUND(100.0 * n_dead_tup / (n_live_tup + n_dead_tup), 2)
    END as dead_tuple_ratio_percent,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY (n_tup_ins + n_tup_upd + n_tup_del) DESC;

-- ============================================================================
-- TRADING-SPECIFIC PERFORMANCE QUERIES
-- ============================================================================

-- 7. Order Processing Performance
SELECT
    'Order Lookups' as metric,
    COUNT(*) as total_orders,
    COUNT(*) FILTER (WHERE status IN ('pending', 'submitted', 'partially_filled')) as active_orders,
    AVG(EXTRACT(EPOCH FROM (submitted_at - created_at)) * 1000) as avg_submission_time_ms,
    MAX(EXTRACT(EPOCH FROM (submitted_at - created_at)) * 1000) as max_submission_time_ms,
    COUNT(DISTINCT symbol) as unique_symbols
FROM orders
WHERE created_at >= NOW() - INTERVAL '1 hour'
UNION ALL
SELECT
    'Order Status Updates' as metric,
    COUNT(*) FILTER (WHERE status = 'filled') as filled_orders,
    COUNT(*) FILTER (WHERE status = 'cancelled') as cancelled_orders,
    AVG(EXTRACT(EPOCH FROM (filled_at - submitted_at)) * 1000) as avg_fill_time_ms,
    MAX(EXTRACT(EPOCH FROM (filled_at - submitted_at)) * 1000) as max_fill_time_ms,
    NULL as unique_symbols
FROM orders
WHERE created_at >= NOW() - INTERVAL '1 hour'
    AND (filled_at IS NOT NULL OR cancelled_at IS NOT NULL);

-- 8. Position Management Performance
WITH position_metrics AS (
    SELECT
        symbol,
        COUNT(*) as position_count,
        SUM(ABS(quantity * average_entry_price)) as total_exposure,
        AVG(ABS(quantity * average_entry_price)) as avg_position_size,
        MAX(ABS(quantity * average_entry_price)) as max_position_size
    FROM positions
    WHERE closed_at IS NULL
    GROUP BY symbol
)
SELECT
    COUNT(*) as unique_symbols_with_positions,
    SUM(position_count) as total_active_positions,
    pg_size_pretty(SUM(total_exposure)::bigint) as total_portfolio_exposure,
    pg_size_pretty(AVG(avg_position_size)::bigint) as avg_position_size,
    pg_size_pretty(MAX(max_position_size)::bigint) as largest_position
FROM position_metrics;

-- 9. Market Data Access Performance
SELECT
    timeframe,
    COUNT(*) as total_bars,
    COUNT(DISTINCT symbol) as unique_symbols,
    MIN(timestamp) as oldest_data,
    MAX(timestamp) as newest_data,
    AVG(volume) as avg_volume,
    pg_size_pretty(
        COUNT(*) * (
            8 + -- timestamp
            20 + -- symbol varchar
            10 + -- timeframe varchar
            8 * 5 + -- OHLC + volume decimals
            36 -- UUID
        )
    ) as estimated_size
FROM market_data
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY timeframe
ORDER BY timeframe;

-- ============================================================================
-- REAL-TIME PERFORMANCE MONITORING
-- ============================================================================

-- 10. Current Database Activity - Active queries and locks
SELECT
    pid,
    usename,
    application_name,
    client_addr,
    state,
    EXTRACT(EPOCH FROM (NOW() - query_start))::int as query_duration_seconds,
    LEFT(query, 100) as query_preview,
    wait_event_type,
    wait_event
FROM pg_stat_activity
WHERE state = 'active'
    AND pid != pg_backend_pid()
    AND query NOT LIKE '%pg_stat_activity%'
ORDER BY query_start;

-- 11. Lock Analysis - Identify blocking queries
SELECT
    blocked_locks.pid AS blocked_pid,
    blocked_activity.usename AS blocked_user,
    blocking_locks.pid AS blocking_pid,
    blocking_activity.usename AS blocking_user,
    blocked_activity.query AS blocked_statement,
    blocking_activity.query AS blocking_statement,
    blocked_activity.application_name AS blocked_application,
    blocking_activity.application_name AS blocking_application,
    blocked_locks.mode AS blocked_mode,
    blocking_locks.mode AS blocking_mode,
    blocked_locks.locktype AS blocked_locktype,
    blocking_locks.locktype AS blocking_locktype
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks
    ON blocking_locks.locktype = blocked_locks.locktype
    AND blocking_locks.DATABASE IS NOT DISTINCT FROM blocked_locks.DATABASE
    AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
    AND blocking_locks.page IS NOT DISTINCT FROM blocked_locks.page
    AND blocking_locks.tuple IS NOT DISTINCT FROM blocked_locks.tuple
    AND blocking_locks.virtualxid IS NOT DISTINCT FROM blocked_locks.virtualxid
    AND blocking_locks.transactionid IS NOT DISTINCT FROM blocked_locks.transactionid
    AND blocking_locks.classid IS NOT DISTINCT FROM blocked_locks.classid
    AND blocking_locks.objid IS NOT DISTINCT FROM blocked_locks.objid
    AND blocking_locks.objsubid IS NOT DISTINCT FROM blocked_locks.objsubid
    AND blocking_locks.pid != blocked_locks.pid
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.GRANTED;

-- ============================================================================
-- PERFORMANCE BENCHMARKING QUERIES
-- ============================================================================

-- 12. Order Query Performance Test
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT o.id, o.symbol, o.status, o.quantity, o.limit_price
FROM orders o
WHERE o.symbol = 'AAPL'
    AND o.status IN ('pending', 'submitted', 'partially_filled')
ORDER BY o.created_at DESC
LIMIT 100;

-- 13. Position Lookup Performance Test
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT p.symbol, p.quantity, p.average_entry_price, p.current_price,
       (p.quantity * p.current_price) - (p.quantity * p.average_entry_price) as unrealized_pnl
FROM positions p
WHERE p.closed_at IS NULL
    AND p.symbol IN ('AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN');

-- 14. Market Data Latest Price Performance Test
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT DISTINCT ON (md.symbol)
    md.symbol, md.close, md.volume, md.timestamp
FROM market_data md
WHERE md.timeframe = '1min'
    AND md.timestamp >= NOW() - INTERVAL '1 hour'
ORDER BY md.symbol, md.timestamp DESC;

-- ============================================================================
-- MAINTENANCE RECOMMENDATIONS
-- ============================================================================

-- 15. Tables Needing VACUUM/ANALYZE
SELECT
    schemaname,
    tablename,
    n_dead_tup as dead_tuples,
    n_live_tup as live_tuples,
    ROUND(100.0 * n_dead_tup / GREATEST(n_live_tup + n_dead_tup, 1), 2) as dead_tuple_percent,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze,
    CASE
        WHEN n_dead_tup > n_live_tup * 0.1 THEN 'VACUUM needed'
        WHEN last_analyze < NOW() - INTERVAL '1 day' THEN 'ANALYZE needed'
        WHEN last_vacuum < NOW() - INTERVAL '7 days' THEN 'VACUUM recommended'
        ELSE 'OK'
    END as maintenance_recommendation
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY dead_tuple_percent DESC;

-- ============================================================================
-- SUMMARY PERFORMANCE DASHBOARD
-- ============================================================================

-- 16. High-Level Performance Summary
WITH performance_summary AS (
    SELECT 'Index Hit Ratio' as metric,
           ROUND(100.0 * SUM(idx_blks_hit) / GREATEST(SUM(idx_blks_hit + idx_blks_read), 1), 2)::text || '%' as value
    FROM pg_statio_user_indexes
    UNION ALL
    SELECT 'Table Hit Ratio' as metric,
           ROUND(100.0 * SUM(heap_blks_hit) / GREATEST(SUM(heap_blks_hit + heap_blks_read), 1), 2)::text || '%' as value
    FROM pg_statio_user_tables
    UNION ALL
    SELECT 'Active Connections' as metric,
           COUNT(*)::text as value
    FROM pg_stat_activity
    WHERE state = 'active'
    UNION ALL
    SELECT 'Total Database Size' as metric,
           pg_size_pretty(SUM(pg_database_size(datname))) as value
    FROM pg_database
    WHERE datname = current_database()
    UNION ALL
    SELECT 'Orders Last Hour' as metric,
           COUNT(*)::text as value
    FROM orders
    WHERE created_at >= NOW() - INTERVAL '1 hour'
    UNION ALL
    SELECT 'Active Positions' as metric,
           COUNT(*)::text as value
    FROM positions
    WHERE closed_at IS NULL
)
SELECT * FROM performance_summary;

-- ============================================================================
-- USAGE INSTRUCTIONS
-- ============================================================================

/*
Performance Monitoring Schedule:

1. Real-time (every minute):
   - Query #10: Current Database Activity
   - Query #11: Lock Analysis

2. High frequency (every 5 minutes):
   - Query #7: Order Processing Performance
   - Query #16: Summary Performance Dashboard

3. Regular monitoring (hourly):
   - Query #1: Most Used Indexes
   - Query #2: Index Hit Ratio
   - Query #6: Most Active Tables

4. Daily analysis:
   - Query #3: Unused Indexes
   - Query #5: Table Access Patterns
   - Query #15: Maintenance Recommendations

5. Weekly deep analysis:
   - Query #4: Index Bloat Analysis
   - Queries #12-14: Performance Benchmarking
   - Query #8-9: Trading-specific metrics

Performance Targets:
- Index hit ratio: >99%
- Table hit ratio: >95%
- Order lookup time: <1ms
- Position lookup time: <2ms
- Market data queries: <5ms
*/
