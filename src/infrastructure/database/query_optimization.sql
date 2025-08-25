-- Query Optimization Helpers for High-Frequency Trading
-- Optimized queries and helper functions for common trading operations
--
-- This file contains:
-- 1. Pre-optimized query templates for common HFT operations
-- 2. Stored procedures for frequently executed operations
-- 3. Materialized views for aggregated data
-- 4. Query plan optimization utilities

-- ============================================================================
-- OPTIMIZED QUERY TEMPLATES FOR COMMON HFT OPERATIONS
-- ============================================================================

-- 1. High-Performance Order Lookup Functions
CREATE OR REPLACE FUNCTION get_active_orders_by_symbol(
    p_symbol VARCHAR(20),
    p_limit INTEGER DEFAULT 100
) RETURNS TABLE(
    id UUID,
    symbol VARCHAR(20),
    side order_side,
    order_type order_type,
    status order_status,
    quantity DECIMAL(18, 8),
    limit_price DECIMAL(18, 8),
    stop_price DECIMAL(18, 8),
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    -- Optimized query using covering index
    RETURN QUERY
    SELECT
        o.id, o.symbol, o.side, o.order_type, o.status,
        o.quantity, o.limit_price, o.stop_price, o.created_at
    FROM orders o
    WHERE o.symbol = p_symbol
        AND o.status IN ('pending', 'submitted', 'partially_filled')
    ORDER BY o.created_at DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- 2. Ultra-Fast Order Status Update
CREATE OR REPLACE FUNCTION update_order_status_fast(
    p_broker_order_id VARCHAR(100),
    p_new_status order_status,
    p_filled_quantity DECIMAL(18, 8) DEFAULT NULL,
    p_average_fill_price DECIMAL(18, 8) DEFAULT NULL
) RETURNS BOOLEAN AS $$
DECLARE
    updated_count INTEGER;
BEGIN
    -- Single query update with minimal index lookups
    UPDATE orders
    SET
        status = p_new_status,
        filled_quantity = COALESCE(p_filled_quantity, filled_quantity),
        average_fill_price = COALESCE(p_average_fill_price, average_fill_price),
        filled_at = CASE WHEN p_new_status = 'filled' THEN NOW() ELSE filled_at END,
        cancelled_at = CASE WHEN p_new_status = 'cancelled' THEN NOW() ELSE cancelled_at END
    WHERE broker_order_id = p_broker_order_id;

    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RETURN updated_count > 0;
END;
$$ LANGUAGE plpgsql;

-- 3. Batch Order Processing for High Throughput
CREATE OR REPLACE FUNCTION process_order_batch(
    p_orders JSONB
) RETURNS TABLE(
    order_id UUID,
    success BOOLEAN,
    error_message TEXT
) AS $$
DECLARE
    order_data JSONB;
    new_order_id UUID;
BEGIN
    -- Process multiple orders in a single transaction
    FOR order_data IN SELECT * FROM jsonb_array_elements(p_orders)
    LOOP
        BEGIN
            INSERT INTO orders (
                symbol, side, order_type, quantity, limit_price, stop_price, time_in_force
            ) VALUES (
                (order_data->>'symbol')::VARCHAR(20),
                (order_data->>'side')::order_side,
                (order_data->>'order_type')::order_type,
                (order_data->>'quantity')::DECIMAL(18, 8),
                NULLIF(order_data->>'limit_price', '')::DECIMAL(18, 8),
                NULLIF(order_data->>'stop_price', '')::DECIMAL(18, 8),
                COALESCE((order_data->>'time_in_force')::time_in_force, 'day')
            ) RETURNING id INTO new_order_id;

            RETURN QUERY SELECT new_order_id, TRUE, NULL::TEXT;

        EXCEPTION WHEN OTHERS THEN
            RETURN QUERY SELECT NULL::UUID, FALSE, SQLERRM;
        END;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- POSITION MANAGEMENT OPTIMIZATIONS
-- ============================================================================

-- 4. Real-Time Position Lookup with P&L
CREATE OR REPLACE FUNCTION get_position_with_pnl(
    p_symbol VARCHAR(20),
    p_current_price DECIMAL(18, 8)
) RETURNS TABLE(
    id UUID,
    symbol VARCHAR(20),
    quantity DECIMAL(18, 8),
    average_entry_price DECIMAL(18, 8),
    current_price DECIMAL(18, 8),
    unrealized_pnl DECIMAL(18, 8),
    realized_pnl DECIMAL(18, 8),
    total_pnl DECIMAL(18, 8),
    pnl_percentage DECIMAL(8, 4),
    position_value DECIMAL(18, 8)
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.id,
        p.symbol,
        p.quantity,
        p.average_entry_price,
        p_current_price as current_price,
        -- Calculate unrealized P&L
        CASE
            WHEN p.quantity > 0 THEN
                (p_current_price - p.average_entry_price) * p.quantity
            WHEN p.quantity < 0 THEN
                (p.average_entry_price - p_current_price) * ABS(p.quantity)
            ELSE 0
        END as unrealized_pnl,
        p.realized_pnl,
        -- Total P&L
        p.realized_pnl + CASE
            WHEN p.quantity > 0 THEN
                (p_current_price - p.average_entry_price) * p.quantity
            WHEN p.quantity < 0 THEN
                (p.average_entry_price - p_current_price) * ABS(p.quantity)
            ELSE 0
        END as total_pnl,
        -- P&L percentage
        CASE
            WHEN p.average_entry_price > 0 AND p.quantity != 0 THEN
                ((p.realized_pnl +
                  CASE
                      WHEN p.quantity > 0 THEN (p_current_price - p.average_entry_price) * p.quantity
                      ELSE (p.average_entry_price - p_current_price) * ABS(p.quantity)
                  END) / (ABS(p.quantity) * p.average_entry_price)) * 100
            ELSE 0
        END as pnl_percentage,
        -- Current position value
        ABS(p.quantity) * p_current_price as position_value
    FROM positions p
    WHERE p.symbol = p_symbol
        AND p.closed_at IS NULL;
END;
$$ LANGUAGE plpgsql STABLE;

-- 5. Portfolio Exposure Summary
CREATE OR REPLACE FUNCTION get_portfolio_exposure_summary()
RETURNS TABLE(
    total_positions INTEGER,
    total_exposure DECIMAL(18, 8),
    total_unrealized_pnl DECIMAL(18, 8),
    total_realized_pnl DECIMAL(18, 8),
    largest_position_symbol VARCHAR(20),
    largest_position_value DECIMAL(18, 8),
    risk_concentration_ratio DECIMAL(8, 4)
) AS $$
BEGIN
    RETURN QUERY
    WITH position_summary AS (
        SELECT
            COUNT(*) as position_count,
            SUM(ABS(quantity * COALESCE(current_price, average_entry_price))) as total_exposure_calc,
            SUM(CASE
                WHEN quantity > 0 AND current_price IS NOT NULL THEN
                    (current_price - average_entry_price) * quantity
                WHEN quantity < 0 AND current_price IS NOT NULL THEN
                    (average_entry_price - current_price) * ABS(quantity)
                ELSE 0
            END) as total_unrealized_calc,
            SUM(realized_pnl) as total_realized_calc
        FROM positions
        WHERE closed_at IS NULL
    ),
    largest_position AS (
        SELECT
            symbol,
            ABS(quantity * COALESCE(current_price, average_entry_price)) as position_value
        FROM positions
        WHERE closed_at IS NULL
        ORDER BY ABS(quantity * COALESCE(current_price, average_entry_price)) DESC
        LIMIT 1
    )
    SELECT
        ps.position_count::INTEGER,
        ps.total_exposure_calc,
        ps.total_unrealized_calc,
        ps.total_realized_calc,
        lp.symbol,
        lp.position_value,
        CASE
            WHEN ps.total_exposure_calc > 0 THEN
                ROUND((lp.position_value / ps.total_exposure_calc * 100)::NUMERIC, 4)
            ELSE 0
        END as risk_concentration_ratio
    FROM position_summary ps
    CROSS JOIN largest_position lp;
END;
$$ LANGUAGE plpgsql STABLE;

-- ============================================================================
-- MARKET DATA OPTIMIZATION
-- ============================================================================

-- 6. Latest Price Lookup (Ultra-Fast)
CREATE OR REPLACE FUNCTION get_latest_prices(
    p_symbols VARCHAR(20)[],
    p_timeframe VARCHAR(10) DEFAULT '1min'
) RETURNS TABLE(
    symbol VARCHAR(20),
    price DECIMAL(18, 8),
    volume BIGINT,
    timestamp TIMESTAMP WITH TIME ZONE,
    vwap DECIMAL(18, 8)
) AS $$
BEGIN
    RETURN QUERY
    SELECT DISTINCT ON (md.symbol)
        md.symbol,
        md.close as price,
        md.volume,
        md.timestamp,
        md.vwap
    FROM market_data md
    WHERE md.symbol = ANY(p_symbols)
        AND md.timeframe = p_timeframe
        AND md.timestamp > NOW() - INTERVAL '1 hour'
    ORDER BY md.symbol, md.timestamp DESC;
END;
$$ LANGUAGE plpgsql STABLE;

-- 7. Price History for Technical Analysis
CREATE OR REPLACE FUNCTION get_price_history(
    p_symbol VARCHAR(20),
    p_timeframe VARCHAR(10),
    p_bars INTEGER DEFAULT 100
) RETURNS TABLE(
    timestamp TIMESTAMP WITH TIME ZONE,
    open DECIMAL(18, 8),
    high DECIMAL(18, 8),
    low DECIMAL(18, 8),
    close DECIMAL(18, 8),
    volume BIGINT,
    vwap DECIMAL(18, 8)
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        md.timestamp,
        md.open,
        md.high,
        md.low,
        md.close,
        md.volume,
        md.vwap
    FROM market_data md
    WHERE md.symbol = p_symbol
        AND md.timeframe = p_timeframe
    ORDER BY md.timestamp DESC
    LIMIT p_bars;
END;
$$ LANGUAGE plpgsql STABLE;

-- ============================================================================
-- MATERIALIZED VIEWS FOR AGGREGATED DATA
-- ============================================================================

-- 8. Real-Time Trading Metrics (Refreshed frequently)
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_realtime_trading_metrics AS
SELECT
    COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '1 hour') as orders_last_hour,
    COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '1 day') as orders_last_day,
    COUNT(*) FILTER (WHERE status = 'filled' AND filled_at >= NOW() - INTERVAL '1 hour') as fills_last_hour,
    COUNT(*) FILTER (WHERE status = 'cancelled' AND cancelled_at >= NOW() - INTERVAL '1 hour') as cancels_last_hour,
    COUNT(DISTINCT symbol) as unique_symbols_traded,
    AVG(EXTRACT(EPOCH FROM (submitted_at - created_at)) * 1000) FILTER (
        WHERE submitted_at IS NOT NULL
        AND created_at >= NOW() - INTERVAL '1 hour'
    ) as avg_submission_latency_ms,
    AVG(EXTRACT(EPOCH FROM (filled_at - submitted_at)) * 1000) FILTER (
        WHERE filled_at IS NOT NULL
        AND submitted_at IS NOT NULL
        AND filled_at >= NOW() - INTERVAL '1 hour'
    ) as avg_fill_latency_ms,
    NOW() as last_updated
FROM orders;

-- Create unique index for concurrent refresh
CREATE UNIQUE INDEX IF NOT EXISTS mv_realtime_trading_metrics_unique
    ON mv_realtime_trading_metrics(last_updated);

-- 9. Position Summary View (Refreshed less frequently)
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_position_summary AS
SELECT
    symbol,
    SUM(quantity) as net_quantity,
    COUNT(*) as position_count,
    AVG(average_entry_price) as avg_entry_price,
    SUM(realized_pnl) as total_realized_pnl,
    SUM(commission_paid) as total_commission,
    MIN(opened_at) as first_position_opened,
    MAX(opened_at) as last_position_opened,
    STRING_AGG(DISTINCT strategy, ', ') as strategies,
    SUM(ABS(quantity * average_entry_price)) as total_exposure
FROM positions
WHERE closed_at IS NULL
GROUP BY symbol
HAVING SUM(quantity) != 0;

-- Create unique index for concurrent refresh
CREATE UNIQUE INDEX IF NOT EXISTS mv_position_summary_symbol
    ON mv_position_summary(symbol);

-- 10. Daily Trading Statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_daily_trading_stats AS
SELECT
    DATE_TRUNC('day', created_at) as trading_date,
    COUNT(*) as total_orders,
    COUNT(*) FILTER (WHERE status = 'filled') as filled_orders,
    COUNT(*) FILTER (WHERE status = 'cancelled') as cancelled_orders,
    COUNT(*) FILTER (WHERE status = 'rejected') as rejected_orders,
    COUNT(DISTINCT symbol) as unique_symbols,
    SUM(quantity) FILTER (WHERE status = 'filled' AND side = 'buy') as total_buy_quantity,
    SUM(quantity) FILTER (WHERE status = 'filled' AND side = 'sell') as total_sell_quantity,
    AVG(filled_quantity * average_fill_price) FILTER (WHERE status = 'filled') as avg_trade_value,
    SUM(filled_quantity * average_fill_price) FILTER (WHERE status = 'filled') as total_trade_value
FROM orders
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', created_at)
ORDER BY trading_date DESC;

-- Create unique index for concurrent refresh
CREATE UNIQUE INDEX IF NOT EXISTS mv_daily_trading_stats_date
    ON mv_daily_trading_stats(trading_date);

-- ============================================================================
-- REFRESH FUNCTIONS FOR MATERIALIZED VIEWS
-- ============================================================================

-- 11. Smart refresh function that only refreshes when needed
CREATE OR REPLACE FUNCTION refresh_materialized_views(
    force_refresh BOOLEAN DEFAULT FALSE
) RETURNS TABLE(
    view_name TEXT,
    refresh_time_ms BIGINT,
    rows_affected BIGINT
) AS $$
DECLARE
    start_time TIMESTAMP;
    end_time TIMESTAMP;
    execution_ms BIGINT;
    row_count BIGINT;
BEGIN
    -- Refresh real-time metrics (always refresh if recent activity)
    IF force_refresh OR EXISTS (
        SELECT 1 FROM orders WHERE created_at >= NOW() - INTERVAL '5 minutes'
    ) THEN
        start_time := clock_timestamp();
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_realtime_trading_metrics;
        end_time := clock_timestamp();
        execution_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;

        SELECT COUNT(*) INTO row_count FROM mv_realtime_trading_metrics;

        RETURN QUERY SELECT
            'mv_realtime_trading_metrics'::TEXT,
            execution_ms,
            row_count;
    END IF;

    -- Refresh position summary (if positions changed)
    IF force_refresh OR EXISTS (
        SELECT 1 FROM positions
        WHERE last_updated >= NOW() - INTERVAL '10 minutes' OR opened_at >= NOW() - INTERVAL '10 minutes'
    ) THEN
        start_time := clock_timestamp();
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_position_summary;
        end_time := clock_timestamp();
        execution_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;

        SELECT COUNT(*) INTO row_count FROM mv_position_summary;

        RETURN QUERY SELECT
            'mv_position_summary'::TEXT,
            execution_ms,
            row_count;
    END IF;

    -- Refresh daily stats (once per hour is sufficient)
    IF force_refresh OR NOT EXISTS (
        SELECT 1 FROM mv_daily_trading_stats
        WHERE trading_date = CURRENT_DATE
    ) THEN
        start_time := clock_timestamp();
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_daily_trading_stats;
        end_time := clock_timestamp();
        execution_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;

        SELECT COUNT(*) INTO row_count FROM mv_daily_trading_stats;

        RETURN QUERY SELECT
            'mv_daily_trading_stats'::TEXT,
            execution_ms,
            row_count;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- QUERY PLAN OPTIMIZATION UTILITIES
-- ============================================================================

-- 12. Function to analyze query performance
CREATE OR REPLACE FUNCTION analyze_query_performance(
    p_query TEXT,
    p_iterations INTEGER DEFAULT 10
) RETURNS TABLE(
    iteration INTEGER,
    execution_time_ms NUMERIC,
    total_cost NUMERIC,
    rows_returned BIGINT
) AS $$
DECLARE
    i INTEGER;
    start_time TIMESTAMP;
    end_time TIMESTAMP;
    execution_ms NUMERIC;
    plan_json JSONB;
    total_cost_val NUMERIC;
    rows_val BIGINT;
BEGIN
    -- Run query multiple times to get average performance
    FOR i IN 1..p_iterations LOOP
        start_time := clock_timestamp();

        -- Execute the query (this is a simplified version)
        -- In practice, you'd need to handle different query types
        EXECUTE p_query;

        end_time := clock_timestamp();
        execution_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;

        -- Get query plan information
        EXECUTE 'EXPLAIN (FORMAT JSON) ' || p_query INTO plan_json;

        total_cost_val := (plan_json->0->'Plan'->>'Total Cost')::NUMERIC;
        rows_val := (plan_json->0->'Plan'->>'Plan Rows')::BIGINT;

        RETURN QUERY SELECT i, execution_ms, total_cost_val, rows_val;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- 13. Index recommendation function
CREATE OR REPLACE FUNCTION recommend_indexes_for_query(
    p_query TEXT
) RETURNS TABLE(
    table_name TEXT,
    column_names TEXT,
    index_type TEXT,
    estimated_benefit TEXT
) AS $$
BEGIN
    -- This is a simplified version - in practice, you'd analyze the query plan
    -- and identify missing indexes based on sequential scans and filter conditions

    RETURN QUERY
    SELECT
        'Example recommendation'::TEXT,
        'symbol, status'::TEXT,
        'BTREE'::TEXT,
        'Would eliminate sequential scan'::TEXT;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- PREPARED STATEMENTS FOR MAXIMUM PERFORMANCE
-- ============================================================================

-- 14. Prepare commonly used statements
DO $$
BEGIN
    -- Prepare statement for order lookups
    PREPARE get_order_by_broker_id (VARCHAR(100)) AS
    SELECT id, symbol, status, quantity, filled_quantity, limit_price
    FROM orders
    WHERE broker_order_id = $1;

    -- Prepare statement for position lookups
    PREPARE get_position_by_symbol (VARCHAR(20)) AS
    SELECT id, quantity, average_entry_price, current_price, realized_pnl
    FROM positions
    WHERE symbol = $1 AND closed_at IS NULL;

    -- Prepare statement for latest price lookup
    PREPARE get_latest_price (VARCHAR(20), VARCHAR(10)) AS
    SELECT close, volume, timestamp
    FROM market_data
    WHERE symbol = $1 AND timeframe = $2
    ORDER BY timestamp DESC
    LIMIT 1;
END $$;

-- ============================================================================
-- PERFORMANCE MONITORING FOR OPTIMIZED QUERIES
-- ============================================================================

-- 15. Track query execution statistics
CREATE TABLE IF NOT EXISTS query_performance_log (
    id SERIAL PRIMARY KEY,
    query_name VARCHAR(100) NOT NULL,
    execution_time_ms NUMERIC NOT NULL,
    rows_affected BIGINT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    parameters JSONB
);

-- Index for performance log queries
CREATE INDEX IF NOT EXISTS idx_query_performance_log_name_time
    ON query_performance_log(query_name, timestamp DESC);

-- Function to log query performance
CREATE OR REPLACE FUNCTION log_query_performance(
    p_query_name VARCHAR(100),
    p_execution_time_ms NUMERIC,
    p_rows_affected BIGINT DEFAULT NULL,
    p_parameters JSONB DEFAULT NULL
) RETURNS VOID AS $$
BEGIN
    INSERT INTO query_performance_log (
        query_name, execution_time_ms, rows_affected, parameters
    ) VALUES (
        p_query_name, p_execution_time_ms, p_rows_affected, p_parameters
    );
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- QUERY OPTIMIZATION BEST PRACTICES
-- ============================================================================

/*
Performance Optimization Guidelines:

1. Use prepared statements for frequently executed queries
2. Leverage covering indexes to avoid table lookups
3. Use materialized views for expensive aggregations
4. Implement proper connection pooling
5. Use batch operations for high-throughput scenarios
6. Monitor and tune query plans regularly
7. Use appropriate data types and constraints
8. Implement proper indexing strategy based on query patterns

Query Performance Targets:
- Order lookups: < 1ms
- Position calculations: < 2ms
- Market data queries: < 5ms
- Portfolio aggregations: < 10ms
- Batch operations: > 1000 ops/sec

Monitoring and Alerting:
- Track query execution times
- Monitor index usage patterns
- Alert on performance degradation
- Analyze slow query logs
- Review query plans for optimization opportunities
*/
