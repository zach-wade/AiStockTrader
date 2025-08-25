-- Index Maintenance Procedures for High-Frequency Trading Database
-- Automated and manual procedures for maintaining optimal index performance
--
-- This file contains:
-- 1. Automated maintenance functions
-- 2. Manual maintenance procedures
-- 3. Monitoring and alerting functions
-- 4. Performance optimization procedures

-- ============================================================================
-- AUTOMATED INDEX MAINTENANCE FUNCTIONS
-- ============================================================================

-- 1. Function to automatically analyze tables based on activity
CREATE OR REPLACE FUNCTION auto_analyze_tables(
    activity_threshold INTEGER DEFAULT 1000,
    analyze_threshold_hours INTEGER DEFAULT 24
) RETURNS TABLE(
    table_name TEXT,
    action_taken TEXT,
    rows_affected BIGINT,
    execution_time_ms BIGINT
) AS $$
DECLARE
    rec RECORD;
    start_time TIMESTAMP;
    end_time TIMESTAMP;
    execution_ms BIGINT;
BEGIN
    -- Analyze tables with significant activity since last analyze
    FOR rec IN
        SELECT
            schemaname,
            tablename,
            n_tup_ins + n_tup_upd + n_tup_del as total_activity,
            n_live_tup + n_dead_tup as total_tuples
        FROM pg_stat_user_tables
        WHERE schemaname = 'public'
            AND (n_tup_ins + n_tup_upd + n_tup_del) > activity_threshold
            AND (last_analyze IS NULL OR last_analyze < NOW() - (analyze_threshold_hours || ' hours')::INTERVAL)
        ORDER BY total_activity DESC
    LOOP
        start_time := clock_timestamp();

        -- Execute ANALYZE
        EXECUTE format('ANALYZE %I.%I', rec.schemaname, rec.tablename);

        end_time := clock_timestamp();
        execution_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;

        RETURN QUERY SELECT
            rec.tablename::TEXT,
            'ANALYZE'::TEXT,
            rec.total_tuples,
            execution_ms;

        -- Log the operation
        RAISE NOTICE 'ANALYZE completed for %.% in % ms', rec.schemaname, rec.tablename, execution_ms;
    END LOOP;

    RETURN;
END;
$$ LANGUAGE plpgsql;

-- 2. Function to rebuild fragmented indexes
CREATE OR REPLACE FUNCTION rebuild_fragmented_indexes(
    size_threshold_mb INTEGER DEFAULT 100,
    fragmentation_threshold_percent INTEGER DEFAULT 20
) RETURNS TABLE(
    index_name TEXT,
    table_name TEXT,
    action_taken TEXT,
    old_size TEXT,
    new_size TEXT,
    execution_time_ms BIGINT
) AS $$
DECLARE
    rec RECORD;
    start_time TIMESTAMP;
    end_time TIMESTAMP;
    execution_ms BIGINT;
    old_size BIGINT;
    new_size BIGINT;
    old_size_pretty TEXT;
    new_size_pretty TEXT;
BEGIN
    -- Find indexes that need rebuilding based on size and estimated fragmentation
    FOR rec IN
        SELECT
            i.indexname,
            i.tablename,
            i.schemaname,
            pg_relation_size(i.indexrelid) as index_size_bytes
        FROM pg_stat_user_indexes i
        WHERE i.schemaname = 'public'
            AND pg_relation_size(i.indexrelid) > size_threshold_mb * 1024 * 1024
            AND i.idx_scan > 0  -- Only rebuild used indexes
        ORDER BY pg_relation_size(i.indexrelid) DESC
    LOOP
        old_size := pg_relation_size(
            (rec.schemaname || '.' || rec.indexname)::regclass
        );
        old_size_pretty := pg_size_pretty(old_size);

        start_time := clock_timestamp();

        -- Rebuild index concurrently to avoid blocking
        BEGIN
            EXECUTE format('REINDEX INDEX CONCURRENTLY %I.%I', rec.schemaname, rec.indexname);

            end_time := clock_timestamp();
            execution_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;

            new_size := pg_relation_size(
                (rec.schemaname || '.' || rec.indexname)::regclass
            );
            new_size_pretty := pg_size_pretty(new_size);

            RETURN QUERY SELECT
                rec.indexname::TEXT,
                rec.tablename::TEXT,
                'REINDEX CONCURRENTLY'::TEXT,
                old_size_pretty,
                new_size_pretty,
                execution_ms;

            RAISE NOTICE 'REINDEX completed for % in % ms (% -> %)',
                rec.indexname, execution_ms, old_size_pretty, new_size_pretty;

        EXCEPTION
            WHEN OTHERS THEN
                RAISE WARNING 'Failed to rebuild index %: %', rec.indexname, SQLERRM;
                CONTINUE;
        END;
    END LOOP;

    RETURN;
END;
$$ LANGUAGE plpgsql;

-- 3. Function to update table statistics during trading hours
CREATE OR REPLACE FUNCTION update_critical_table_stats()
RETURNS TABLE(
    table_name TEXT,
    action_taken TEXT,
    execution_time_ms BIGINT
) AS $$
DECLARE
    critical_tables TEXT[] := ARRAY['orders', 'positions', 'market_data'];
    table_name_var TEXT;
    start_time TIMESTAMP;
    end_time TIMESTAMP;
    execution_ms BIGINT;
BEGIN
    -- Update statistics for critical trading tables
    FOREACH table_name_var IN ARRAY critical_tables
    LOOP
        start_time := clock_timestamp();

        -- Fast statistics update (sample-based)
        EXECUTE format('ANALYZE %I', table_name_var);

        end_time := clock_timestamp();
        execution_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;

        RETURN QUERY SELECT
            table_name_var,
            'FAST_ANALYZE'::TEXT,
            execution_ms;

        RAISE NOTICE 'Fast ANALYZE completed for % in % ms', table_name_var, execution_ms;
    END LOOP;

    RETURN;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- INDEX MONITORING AND ALERTING FUNCTIONS
-- ============================================================================

-- 4. Function to detect performance degradation
CREATE OR REPLACE FUNCTION detect_performance_issues()
RETURNS TABLE(
    issue_type TEXT,
    table_name TEXT,
    index_name TEXT,
    severity TEXT,
    description TEXT,
    recommended_action TEXT
) AS $$
BEGIN
    -- Check for low index hit ratios
    RETURN QUERY
    SELECT
        'Low Index Hit Ratio'::TEXT,
        pui.tablename::TEXT,
        pui.indexrelname::TEXT,
        CASE
            WHEN hit_ratio < 90 THEN 'CRITICAL'
            WHEN hit_ratio < 95 THEN 'HIGH'
            ELSE 'MEDIUM'
        END::TEXT,
        ('Hit ratio: ' || hit_ratio::TEXT || '%')::TEXT,
        'Investigate query patterns and consider index optimization'::TEXT
    FROM (
        SELECT
            pui.tablename,
            pui.indexrelname,
            CASE
                WHEN (psui.idx_blks_hit + psui.idx_blks_read) = 0 THEN 100
                ELSE ROUND(100.0 * psui.idx_blks_hit / (psui.idx_blks_hit + psui.idx_blks_read), 2)
            END as hit_ratio
        FROM pg_stat_user_indexes pui
        JOIN pg_statio_user_indexes psui ON pui.indexrelid = psui.indexrelid
        WHERE pui.schemaname = 'public'
            AND (psui.idx_blks_hit + psui.idx_blks_read) > 1000
    ) pui
    WHERE hit_ratio < 98;

    -- Check for unused large indexes
    RETURN QUERY
    SELECT
        'Unused Large Index'::TEXT,
        tablename::TEXT,
        indexname::TEXT,
        CASE
            WHEN pg_relation_size(indexrelid) > 100 * 1024 * 1024 THEN 'HIGH'
            WHEN pg_relation_size(indexrelid) > 10 * 1024 * 1024 THEN 'MEDIUM'
            ELSE 'LOW'
        END::TEXT,
        ('Size: ' || pg_size_pretty(pg_relation_size(indexrelid)) || ', Scans: ' || idx_scan::TEXT)::TEXT,
        'Consider dropping if truly unused'::TEXT
    FROM pg_stat_user_indexes
    WHERE schemaname = 'public'
        AND idx_scan < 10
        AND pg_relation_size(indexrelid) > 1024 * 1024;

    -- Check for tables with high dead tuple ratios
    RETURN QUERY
    SELECT
        'High Dead Tuple Ratio'::TEXT,
        tablename::TEXT,
        NULL::TEXT,
        CASE
            WHEN dead_ratio > 50 THEN 'CRITICAL'
            WHEN dead_ratio > 25 THEN 'HIGH'
            ELSE 'MEDIUM'
        END::TEXT,
        ('Dead tuples: ' || dead_ratio::TEXT || '%')::TEXT,
        'Schedule VACUUM operation'::TEXT
    FROM (
        SELECT
            tablename,
            CASE
                WHEN n_live_tup + n_dead_tup = 0 THEN 0
                ELSE ROUND(100.0 * n_dead_tup / (n_live_tup + n_dead_tup), 2)
            END as dead_ratio
        FROM pg_stat_user_tables
        WHERE schemaname = 'public'
            AND n_dead_tup > 1000
    ) t
    WHERE dead_ratio > 15;

    RETURN;
END;
$$ LANGUAGE plpgsql;

-- 5. Function to generate maintenance recommendations
CREATE OR REPLACE FUNCTION generate_maintenance_recommendations()
RETURNS TABLE(
    priority TEXT,
    recommendation TEXT,
    rationale TEXT,
    estimated_impact TEXT
) AS $$
BEGIN
    -- High priority recommendations
    RETURN QUERY
    SELECT
        'HIGH'::TEXT,
        ('VACUUM ' || tablename)::TEXT,
        ('Dead tuple ratio: ' ||
         ROUND(100.0 * n_dead_tup / GREATEST(n_live_tup + n_dead_tup, 1), 1)::TEXT || '%')::TEXT,
        'Improve query performance and reclaim space'::TEXT
    FROM pg_stat_user_tables
    WHERE schemaname = 'public'
        AND n_dead_tup > GREATEST(n_live_tup * 0.2, 10000);

    -- Medium priority recommendations
    RETURN QUERY
    SELECT
        'MEDIUM'::TEXT,
        ('ANALYZE ' || tablename)::TEXT,
        ('Last analyzed: ' ||
         COALESCE(last_analyze::TEXT, 'Never'))::TEXT,
        'Update query planner statistics'::TEXT
    FROM pg_stat_user_tables
    WHERE schemaname = 'public'
        AND (last_analyze IS NULL OR last_analyze < NOW() - INTERVAL '7 days')
        AND (n_tup_ins + n_tup_upd + n_tup_del) > 1000;

    -- Low priority recommendations
    RETURN QUERY
    SELECT
        'LOW'::TEXT,
        ('Consider partitioning ' || tablename)::TEXT,
        ('Table size: ' || pg_size_pretty(pg_total_relation_size(c.oid)))::TEXT,
        'Improve query performance for large tables'::TEXT
    FROM pg_stat_user_tables put
    JOIN pg_class c ON c.oid = put.relid
    WHERE put.schemaname = 'public'
        AND pg_total_relation_size(c.oid) > 1024 * 1024 * 1024  -- > 1GB
        AND put.tablename IN ('orders', 'market_data');

    RETURN;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- MAINTENANCE SCHEDULING FUNCTIONS
-- ============================================================================

-- 6. Function to check if maintenance window is active
CREATE OR REPLACE FUNCTION is_maintenance_window(
    maintenance_start_hour INTEGER DEFAULT 2,  -- 2 AM
    maintenance_end_hour INTEGER DEFAULT 6     -- 6 AM
) RETURNS BOOLEAN AS $$
DECLARE
    current_hour INTEGER;
BEGIN
    current_hour := EXTRACT(HOUR FROM NOW());

    -- Check if current time is within maintenance window
    IF maintenance_start_hour <= maintenance_end_hour THEN
        RETURN current_hour >= maintenance_start_hour AND current_hour < maintenance_end_hour;
    ELSE
        -- Handle overnight maintenance window (e.g., 22:00 to 06:00)
        RETURN current_hour >= maintenance_start_hour OR current_hour < maintenance_end_hour;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- 7. Function to perform safe maintenance during trading hours
CREATE OR REPLACE FUNCTION safe_trading_hours_maintenance()
RETURNS TABLE(
    operation TEXT,
    table_name TEXT,
    execution_time_ms BIGINT,
    rows_affected BIGINT
) AS $$
DECLARE
    max_execution_time_ms INTEGER := 100;  -- Maximum 100ms per operation
    start_time TIMESTAMP;
    end_time TIMESTAMP;
    execution_ms BIGINT;
BEGIN
    -- Only perform very fast operations during trading hours

    -- Quick statistics update for critical tables
    FOR table_name IN SELECT unnest(ARRAY['orders', 'positions'])
    LOOP
        start_time := clock_timestamp();

        -- Sample-based analyze (faster than full analyze)
        EXECUTE format('ANALYZE %I', table_name);

        end_time := clock_timestamp();
        execution_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;

        -- Only return if execution was fast enough
        IF execution_ms <= max_execution_time_ms THEN
            RETURN QUERY SELECT
                'FAST_ANALYZE'::TEXT,
                table_name::TEXT,
                execution_ms,
                0::BIGINT;
        END IF;
    END LOOP;

    RETURN;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- PERFORMANCE OPTIMIZATION PROCEDURES
-- ============================================================================

-- 8. Procedure to optimize query plans
CREATE OR REPLACE FUNCTION optimize_query_plans()
RETURNS TABLE(
    table_name TEXT,
    optimization TEXT,
    before_cost NUMERIC,
    after_cost NUMERIC,
    improvement_percent NUMERIC
) AS $$
DECLARE
    rec RECORD;
BEGIN
    -- Update statistics for all trading tables
    FOR rec IN
        SELECT tablename
        FROM pg_stat_user_tables
        WHERE schemaname = 'public'
            AND tablename IN ('orders', 'positions', 'portfolios', 'market_data')
    LOOP
        EXECUTE format('ANALYZE %I', rec.tablename);

        RETURN QUERY SELECT
            rec.tablename::TEXT,
            'STATISTICS_UPDATE'::TEXT,
            0::NUMERIC,
            0::NUMERIC,
            0::NUMERIC;
    END LOOP;

    RETURN;
END;
$$ LANGUAGE plpgsql;

-- 9. Function to create missing indexes based on query patterns
CREATE OR REPLACE FUNCTION suggest_missing_indexes()
RETURNS TABLE(
    table_name TEXT,
    suggested_index TEXT,
    rationale TEXT,
    estimated_benefit TEXT
) AS $$
BEGIN
    -- Analyze query patterns and suggest indexes
    -- This is a simplified version - in practice, you'd analyze pg_stat_statements

    -- Check for frequently filtered columns without indexes
    RETURN QUERY
    WITH missing_indexes AS (
        SELECT
            'orders'::TEXT as table_name,
            'CREATE INDEX idx_orders_symbol_side ON orders(symbol, side)'::TEXT as suggested_index,
            'Frequent filtering by symbol and side'::TEXT as rationale,
            'High - Used in order book queries'::TEXT as estimated_benefit
        WHERE NOT EXISTS (
            SELECT 1 FROM pg_indexes
            WHERE tablename = 'orders'
            AND indexdef LIKE '%symbol%side%'
        )

        UNION ALL

        SELECT
            'positions'::TEXT,
            'CREATE INDEX idx_positions_strategy_symbol ON positions(strategy, symbol) WHERE closed_at IS NULL'::TEXT,
            'Strategy-based position lookups'::TEXT,
            'Medium - Used in portfolio analysis'::TEXT
        WHERE NOT EXISTS (
            SELECT 1 FROM pg_indexes
            WHERE tablename = 'positions'
            AND indexdef LIKE '%strategy%symbol%'
        )
    )
    SELECT * FROM missing_indexes;

    RETURN;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- EMERGENCY MAINTENANCE PROCEDURES
-- ============================================================================

-- 10. Emergency procedure to kill long-running queries
CREATE OR REPLACE FUNCTION emergency_kill_long_queries(
    max_duration_seconds INTEGER DEFAULT 300
) RETURNS TABLE(
    killed_pid INTEGER,
    query_duration_seconds INTEGER,
    query_preview TEXT
) AS $$
DECLARE
    rec RECORD;
BEGIN
    -- Find and kill queries running longer than threshold
    FOR rec IN
        SELECT
            pid,
            EXTRACT(EPOCH FROM (NOW() - query_start))::INTEGER as duration_seconds,
            LEFT(query, 100) as query_preview
        FROM pg_stat_activity
        WHERE state = 'active'
            AND pid != pg_backend_pid()
            AND query_start < NOW() - (max_duration_seconds || ' seconds')::INTERVAL
            AND query NOT LIKE '%pg_stat_activity%'
    LOOP
        -- Attempt to cancel the query first
        PERFORM pg_cancel_backend(rec.pid);

        -- Wait a moment, then terminate if still running
        PERFORM pg_sleep(1);

        IF EXISTS (SELECT 1 FROM pg_stat_activity WHERE pid = rec.pid AND state = 'active') THEN
            PERFORM pg_terminate_backend(rec.pid);
        END IF;

        RETURN QUERY SELECT
            rec.pid,
            rec.duration_seconds,
            rec.query_preview;

        RAISE NOTICE 'Killed long-running query: PID %, Duration %s', rec.pid, rec.duration_seconds;
    END LOOP;

    RETURN;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- MAINTENANCE AUTOMATION VIEWS
-- ============================================================================

-- View for monitoring maintenance status
CREATE OR REPLACE VIEW v_maintenance_status AS
SELECT
    'Current Time' as metric,
    NOW()::TEXT as value,
    'System timestamp' as description
UNION ALL
SELECT
    'Maintenance Window',
    CASE WHEN is_maintenance_window() THEN 'ACTIVE' ELSE 'INACTIVE' END,
    'Whether maintenance operations can run'
UNION ALL
SELECT
    'Tables Needing VACUUM',
    COUNT(*)::TEXT,
    'Tables with >20% dead tuples'
FROM pg_stat_user_tables
WHERE schemaname = 'public'
    AND n_dead_tup > GREATEST(n_live_tup * 0.2, 1000)
UNION ALL
SELECT
    'Tables Needing ANALYZE',
    COUNT(*)::TEXT,
    'Tables not analyzed in last 24 hours'
FROM pg_stat_user_tables
WHERE schemaname = 'public'
    AND (last_analyze IS NULL OR last_analyze < NOW() - INTERVAL '24 hours')
    AND (n_tup_ins + n_tup_upd + n_tup_del) > 100;

-- ============================================================================
-- USAGE EXAMPLES AND SCHEDULING
-- ============================================================================

/*
Maintenance Schedule Examples:

1. During trading hours (every 5 minutes):
   SELECT * FROM safe_trading_hours_maintenance();

2. Maintenance window (daily at 3 AM):
   SELECT * FROM auto_analyze_tables();
   SELECT * FROM update_critical_table_stats();

3. Weekly maintenance (Sunday 2 AM):
   SELECT * FROM rebuild_fragmented_indexes();
   SELECT * FROM generate_maintenance_recommendations();

4. Continuous monitoring (every minute):
   SELECT * FROM detect_performance_issues();
   SELECT * FROM v_maintenance_status;

5. Emergency procedures (as needed):
   SELECT * FROM emergency_kill_long_queries(120);

Automated Scheduling with pg_cron (if available):

-- Every 5 minutes during trading hours
SELECT cron.schedule('trading-maintenance', '*/5 9-16 * * 1-5',
    'SELECT safe_trading_hours_maintenance();');

-- Daily at 3 AM
SELECT cron.schedule('daily-maintenance', '0 3 * * *',
    'SELECT auto_analyze_tables(); SELECT update_critical_table_stats();');

-- Weekly on Sunday at 2 AM
SELECT cron.schedule('weekly-maintenance', '0 2 * * 0',
    'SELECT rebuild_fragmented_indexes();');
*/

-- ============================================================================
-- PERFORMANCE IMPACT MONITORING
-- ============================================================================

-- Create a table to log maintenance operations
CREATE TABLE IF NOT EXISTS maintenance_log (
    id SERIAL PRIMARY KEY,
    operation_type VARCHAR(50) NOT NULL,
    table_name VARCHAR(100),
    index_name VARCHAR(100),
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    execution_time_ms BIGINT,
    rows_affected BIGINT,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for maintenance log queries
CREATE INDEX IF NOT EXISTS idx_maintenance_log_operation_time
    ON maintenance_log(operation_type, start_time DESC);

-- Function to log maintenance operations
CREATE OR REPLACE FUNCTION log_maintenance_operation(
    p_operation_type VARCHAR(50),
    p_table_name VARCHAR(100) DEFAULT NULL,
    p_index_name VARCHAR(100) DEFAULT NULL,
    p_start_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    p_end_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    p_rows_affected BIGINT DEFAULT NULL,
    p_success BOOLEAN DEFAULT TRUE,
    p_error_message TEXT DEFAULT NULL
) RETURNS VOID AS $$
BEGIN
    INSERT INTO maintenance_log (
        operation_type, table_name, index_name, start_time, end_time,
        execution_time_ms, rows_affected, success, error_message
    ) VALUES (
        p_operation_type, p_table_name, p_index_name, p_start_time, p_end_time,
        EXTRACT(EPOCH FROM (p_end_time - p_start_time)) * 1000,
        p_rows_affected, p_success, p_error_message
    );
END;
$$ LANGUAGE plpgsql;
