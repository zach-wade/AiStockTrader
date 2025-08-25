-- Advanced Trading Indexes for High-Frequency Performance
-- Optimized for 1000+ orders/sec with sub-millisecond response times
--
-- Index Strategy:
-- 1. Composite indexes for multi-column query patterns
-- 2. Partial indexes for frequently filtered data subsets
-- 3. Functional indexes for calculated fields
-- 4. Covering indexes to avoid table lookups
-- 5. Time-series optimizations for market data

-- ============================================================================
-- ORDER MANAGEMENT INDEXES - Critical for HFT Order Processing
-- ============================================================================

-- High-frequency order lookup patterns
-- Covers: Finding active orders by symbol and status (most critical HFT query)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_symbol_status_created_active
    ON orders(symbol, status, created_at DESC)
    WHERE status IN ('pending', 'submitted', 'partially_filled');

-- Order book management - fast price-level matching
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_symbol_side_limit_price_active
    ON orders(symbol, side, limit_price, created_at)
    WHERE status IN ('pending', 'submitted', 'partially_filled')
    AND limit_price IS NOT NULL;

-- Broker order ID lookups (extremely frequent in HFT)
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_broker_id_unique
    ON orders(broker_order_id)
    WHERE broker_order_id IS NOT NULL;

-- Order status updates by time ranges
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_status_submitted_at
    ON orders(status, submitted_at DESC)
    WHERE submitted_at IS NOT NULL;

-- Covering index for order summary queries (avoids table lookups)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_symbol_status_covering
    ON orders(symbol, status)
    INCLUDE (quantity, filled_quantity, limit_price, stop_price, created_at)
    WHERE status IN ('pending', 'submitted', 'partially_filled');

-- Partial fill tracking (critical for order management)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_partial_fills
    ON orders(symbol, filled_quantity, quantity, status)
    WHERE status = 'partially_filled' AND filled_quantity > 0;

-- Time-based order expiry checks
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_time_in_force_created
    ON orders(time_in_force, created_at)
    WHERE status IN ('pending', 'submitted', 'partially_filled');

-- Order rejection analysis
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_rejected_reason
    ON orders(status, created_at DESC, symbol)
    WHERE status = 'rejected';

-- ============================================================================
-- POSITION MANAGEMENT INDEXES - Real-time Position Tracking
-- ============================================================================

-- Real-time position lookups by symbol (most frequent position query)
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_symbol_active_unique
    ON positions(symbol)
    WHERE closed_at IS NULL;

-- Position P&L calculations requiring price joins
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_pnl_calculation
    ON positions(symbol, quantity, average_entry_price, current_price)
    WHERE closed_at IS NULL;

-- Position risk analysis - exposure calculations
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_risk_exposure
    ON positions(symbol, quantity, average_entry_price, opened_at)
    WHERE closed_at IS NULL AND quantity != 0;

-- Portfolio position aggregations
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_portfolio_aggregation
    ON positions(strategy, symbol, quantity, realized_pnl)
    WHERE closed_at IS NULL;

-- Stop loss and take profit monitoring
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_stop_levels
    ON positions(symbol, stop_loss_price, take_profit_price, current_price)
    WHERE closed_at IS NULL
    AND (stop_loss_price IS NOT NULL OR take_profit_price IS NOT NULL);

-- Position performance analysis
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_performance_analysis
    ON positions(strategy, opened_at DESC, realized_pnl, commission_paid)
    WHERE closed_at IS NOT NULL;

-- Covering index for position summaries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_summary_covering
    ON positions(symbol)
    INCLUDE (quantity, average_entry_price, current_price, realized_pnl, opened_at)
    WHERE closed_at IS NULL;

-- ============================================================================
-- MARKET DATA INDEXES - High-Performance Time-Series Access
-- ============================================================================

-- Latest price lookups (most critical market data query)
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_symbol_timeframe_latest
    ON market_data(symbol, timeframe, timestamp DESC);

-- Multi-symbol latest price batch queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_latest_prices
    ON market_data(timeframe, timestamp DESC, symbol)
    INCLUDE (close, volume, vwap);

-- Historical price ranges for analysis
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_symbol_time_range
    ON market_data(symbol, timeframe, timestamp)
    INCLUDE (open, high, low, close, volume);

-- Volume-weighted price calculations
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_vwap_calculation
    ON market_data(symbol, timeframe, timestamp)
    INCLUDE (close, volume, vwap, trade_count)
    WHERE vwap IS NOT NULL;

-- Intraday data optimization (most active trading hours)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_intraday_active
    ON market_data(symbol, timestamp DESC)
    WHERE timeframe IN ('1min', '5min', '15min')
    AND timestamp > NOW() - INTERVAL '1 day';

-- Real-time streaming data (last 1 hour)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_realtime_stream
    ON market_data(symbol, timeframe, timestamp DESC)
    WHERE timestamp > NOW() - INTERVAL '1 hour';

-- Cross-timeframe analysis
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_multi_timeframe
    ON market_data(symbol, timestamp DESC, timeframe)
    INCLUDE (close, volume);

-- ============================================================================
-- RISK MANAGEMENT INDEXES - Real-time Risk Calculations
-- ============================================================================

-- Portfolio exposure calculations
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_risk_portfolio_exposure
    ON positions(strategy, symbol, quantity, average_entry_price)
    WHERE closed_at IS NULL;

-- Symbol concentration risk
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_risk_symbol_concentration
    ON positions(symbol, quantity, average_entry_price)
    WHERE closed_at IS NULL AND quantity != 0;

-- Leverage calculations
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_risk_leverage_calculation
    ON positions(quantity, average_entry_price, current_price)
    WHERE closed_at IS NULL AND quantity != 0;

-- Daily risk metrics
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_risk_daily_metrics
    ON positions(opened_at::date, realized_pnl, commission_paid)
    WHERE opened_at >= CURRENT_DATE - INTERVAL '30 days';

-- ============================================================================
-- COMPLIANCE AND AUDIT INDEXES - Regulatory Requirements
-- ============================================================================

-- Order audit trail
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_order_trail
    ON orders(created_at DESC, symbol, side, quantity, limit_price)
    INCLUDE (broker_order_id, status, filled_quantity);

-- Position audit trail
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_position_trail
    ON positions(opened_at DESC, symbol, quantity, strategy)
    INCLUDE (average_entry_price, realized_pnl, closed_at);

-- Daily trading activity
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_daily_activity
    ON orders(created_at::date DESC, status)
    WHERE status IN ('filled', 'partially_filled');

-- ============================================================================
-- PERFORMANCE OPTIMIZATION INDEXES - Query Acceleration
-- ============================================================================

-- Order count aggregations
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_perf_order_counts
    ON orders(status, created_at::date)
    WHERE created_at >= CURRENT_DATE - INTERVAL '90 days';

-- Position count aggregations
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_perf_position_counts
    ON positions(strategy, opened_at::date)
    WHERE opened_at >= CURRENT_DATE - INTERVAL '90 days';

-- Market data volume analysis
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_perf_volume_analysis
    ON market_data(symbol, timestamp::date, volume)
    WHERE timestamp >= CURRENT_DATE - INTERVAL '30 days';

-- ============================================================================
-- FUNCTIONAL INDEXES - Calculated Field Optimizations
-- ============================================================================

-- Order fill percentage calculations
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_func_order_fill_percentage
    ON orders((filled_quantity / quantity * 100))
    WHERE status IN ('partially_filled', 'filled') AND quantity > 0;

-- Position value calculations
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_func_position_value
    ON positions((ABS(quantity) * average_entry_price))
    WHERE closed_at IS NULL AND quantity != 0;

-- Market data price change calculations
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_func_price_change
    ON market_data(symbol, timestamp, ((close - open) / open * 100))
    WHERE timestamp >= CURRENT_DATE - INTERVAL '7 days';

-- ============================================================================
-- JSONB INDEXES - Tag and Metadata Optimization
-- ============================================================================

-- Order tags using GIN index for flexible querying
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_tags_gin
    ON orders USING GIN (tags)
    WHERE tags IS NOT NULL AND tags != '{}';

-- Position tags using GIN index
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_tags_gin
    ON positions USING GIN (tags)
    WHERE tags IS NOT NULL AND tags != '{}';

-- Portfolio tags using GIN index
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolios_tags_gin
    ON portfolios USING GIN (tags)
    WHERE tags IS NOT NULL AND tags != '{}';

-- ============================================================================
-- CLUSTER INDEXES - Physical Data Organization
-- ============================================================================

-- Cluster orders by symbol and creation time for sequential access
-- Note: Only one cluster index per table, choose the most important access pattern
-- CLUSTER orders USING idx_orders_symbol_status_created_active;

-- ============================================================================
-- INDEX USAGE MONITORING VIEWS
-- ============================================================================

-- Create view to monitor index usage
CREATE OR REPLACE VIEW v_index_usage_stats AS
SELECT
    schemaname,
    tablename,
    indexname,
    idx_tup_read,
    idx_tup_fetch,
    idx_scan,
    idx_tup_read::float / GREATEST(idx_scan, 1) as avg_tuples_per_scan,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY idx_scan DESC, idx_tup_read DESC;

-- Create view to identify unused indexes
CREATE OR REPLACE VIEW v_unused_indexes AS
SELECT
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size,
    idx_scan,
    idx_tup_read
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
    AND idx_scan < 10  -- Adjust threshold as needed
    AND pg_relation_size(indexrelid) > 1024 * 1024  -- Larger than 1MB
ORDER BY pg_relation_size(indexrelid) DESC;

-- ============================================================================
-- COMMENTS AND DOCUMENTATION
-- ============================================================================

COMMENT ON INDEX idx_orders_symbol_status_created_active IS
'Critical HFT index: Fast lookup of active orders by symbol and status';

COMMENT ON INDEX idx_orders_broker_id_unique IS
'Unique constraint: Ensures broker order ID uniqueness and fast lookups';

COMMENT ON INDEX idx_positions_symbol_active_unique IS
'Unique constraint: One active position per symbol, fast position lookups';

COMMENT ON INDEX idx_market_data_symbol_timeframe_latest IS
'Latest price index: Optimized for real-time price queries';

-- ============================================================================
-- INDEX MAINTENANCE RECOMMENDATIONS
-- ============================================================================

-- These indexes should be monitored and maintained regularly:
-- 1. REINDEX CONCURRENTLY during low-activity periods
-- 2. Update statistics frequently: ANALYZE tables
-- 3. Monitor index bloat and fragmentation
-- 4. Consider partitioning for large tables (market_data, orders)
-- 5. Use pg_stat_user_indexes to monitor usage patterns

-- Recommended maintenance schedule:
-- Daily: ANALYZE orders, positions, market_data
-- Weekly: Check index usage statistics
-- Monthly: REINDEX unused or fragmented indexes
-- Quarterly: Review and optimize index strategy based on query patterns
