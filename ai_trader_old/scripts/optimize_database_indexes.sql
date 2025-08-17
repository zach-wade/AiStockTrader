-- Hunter-Killer Strategy Database Index Optimizations
-- Run this script to add performance-critical indexes for scanner queries

-- ============================================================
-- Scanner Performance Indexes
-- ============================================================

-- Composite index for Layer 3 scanner queries
-- This dramatically speeds up finding qualified symbols with high scores
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_scanner_qualified
ON companies(layer3_qualified, premarket_score DESC, liquidity_score DESC)
WHERE is_active = true;

-- Index for Layer 2 catalyst scanner
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_layer2_catalyst
ON companies(layer2_qualified, catalyst_score DESC, layer2_updated)
WHERE is_active = true;

-- Index for Layer 1 liquidity filter
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_layer1_liquidity
ON companies(layer1_qualified, liquidity_score DESC, avg_dollar_volume DESC)
WHERE is_active = true;

-- Composite index for scanner layer progression
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_layer_progression
ON companies(layer1_qualified, layer2_qualified, layer3_qualified)
WHERE is_active = true;

-- ============================================================
-- Market Data Performance Indexes
-- ============================================================

-- Composite index for symbol + timestamp queries (with included columns)
-- This is critical for fast feature calculation and OHLCV queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_symbol_timestamp_composite
ON market_data(symbol, timestamp DESC)
INCLUDE (open, high, low, close, volume, vwap);

-- Index for time-based market data queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_timestamp_only
ON market_data(timestamp DESC);

-- Partial index for recent market data (last 30 days)
-- Speeds up queries for recent data which is most frequently accessed
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_recent
ON market_data(symbol, timestamp DESC)
WHERE timestamp > NOW() - INTERVAL '30 days';

-- ============================================================
-- Scanner Alert Performance Indexes
-- ============================================================

-- Composite index for active high-priority alerts
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scanner_alerts_active_priority_time
ON scanner_alerts(is_active, priority, timestamp DESC)
WHERE is_active = true;

-- Index for symbol-based alert queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scanner_alerts_symbol_active
ON scanner_alerts(symbol, timestamp DESC)
WHERE is_active = true;

-- ============================================================
-- Order Management Indexes
-- ============================================================

-- Index for active order monitoring
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_active_monitoring
ON orders(status, created_at DESC)
WHERE status IN ('pending', 'submitted', 'partial');

-- Index for symbol-based order queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_symbol_status
ON orders(symbol, status, created_at DESC);

-- ============================================================
-- Feature Store Indexes
-- ============================================================

-- Composite index for feature retrieval
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_feature_store_retrieval
ON feature_store(symbol, timestamp DESC, feature_set, version);

-- Index for latest features per symbol
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_feature_store_latest
ON feature_store(symbol, feature_set, version, timestamp DESC);

-- ============================================================
-- Strategy Performance Indexes
-- ============================================================

-- Index for strategy performance queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_strategy_performance_lookup
ON strategy_performance(strategy_name, timestamp DESC);

-- ============================================================
-- News Data Indexes (for catalyst detection)
-- ============================================================

-- Index for recent news by symbol
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_news_data_symbol_recent
ON news_data(symbol, published_date DESC)
WHERE published_date > NOW() - INTERVAL '7 days';

-- ============================================================
-- Aggregate Table Indexes
-- ============================================================

-- Composite index for aggregate queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_aggregates_symbol_date_timeframe
ON aggregates(symbol, date DESC, timeframe);

-- ============================================================
-- Performance Monitoring
-- ============================================================

-- Create extension for monitoring if not exists
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- View to monitor slow queries
CREATE OR REPLACE VIEW slow_queries AS
SELECT
    query,
    calls,
    total_exec_time,
    mean_exec_time,
    stddev_exec_time,
    max_exec_time
FROM pg_stat_statements
WHERE mean_exec_time > 100  -- Queries averaging > 100ms
ORDER BY mean_exec_time DESC
LIMIT 50;

-- ============================================================
-- Index Usage Statistics
-- ============================================================

-- View to monitor index usage
CREATE OR REPLACE VIEW index_usage_stats AS
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- ============================================================
-- Table Statistics Update
-- ============================================================

-- Update table statistics for query planner
ANALYZE companies;
ANALYZE market_data;
ANALYZE scanner_alerts;
ANALYZE orders;
ANALYZE feature_store;

-- ============================================================
-- Connection Pool Recommendations
-- ============================================================

-- Check current connection settings
-- Run these queries to verify connection pool optimization

-- Current max connections
SHOW max_connections;

-- Current connection count
SELECT count(*) FROM pg_stat_activity;

-- Connections by state
SELECT state, count(*)
FROM pg_stat_activity
GROUP BY state;

-- Long running queries
SELECT
    pid,
    now() - pg_stat_activity.query_start AS duration,
    query,
    state
FROM pg_stat_activity
WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes'
ORDER BY duration DESC;

-- ============================================================
-- Maintenance Tasks
-- ============================================================

-- Schedule these for off-hours

-- Vacuum analyze for performance
-- VACUUM ANALYZE companies;
-- VACUUM ANALYZE market_data;
-- VACUUM ANALYZE scanner_alerts;

-- Reindex if needed (locks table, run during maintenance window)
-- REINDEX TABLE CONCURRENTLY companies;
-- REINDEX TABLE CONCURRENTLY market_data;
