-- Hunter-Killer Strategy Database Index Optimizations (Fixed)
-- Run this script to add missing performance-critical indexes

-- ============================================================
-- Market Data Performance Indexes (without CONCURRENTLY for TimescaleDB)
-- ============================================================

-- Check if market_data is a hypertable
DO $$
BEGIN
    -- Try to create index without CONCURRENTLY for hypertables
    IF EXISTS (
        SELECT 1 FROM _timescaledb_catalog.hypertable 
        WHERE table_name = 'market_data'
    ) THEN
        -- For hypertables, create without CONCURRENTLY
        CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp_composite 
        ON market_data(symbol, timestamp DESC);
        
        CREATE INDEX IF NOT EXISTS idx_market_data_timestamp_only
        ON market_data(timestamp DESC);
        
        -- Partial index for recent market data (last 30 days)
        CREATE INDEX IF NOT EXISTS idx_market_data_recent
        ON market_data(symbol, timestamp DESC)
        WHERE timestamp > NOW() - INTERVAL '30 days';
        
        RAISE NOTICE 'Created TimescaleDB-compatible indexes for market_data';
    ELSE
        -- For regular tables, use CONCURRENTLY
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_symbol_timestamp_composite 
        ON market_data(symbol, timestamp DESC);
        
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_timestamp_only
        ON market_data(timestamp DESC);
        
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_recent
        ON market_data(symbol, timestamp DESC)
        WHERE timestamp > NOW() - INTERVAL '30 days';
        
        RAISE NOTICE 'Created regular indexes for market_data';
    END IF;
EXCEPTION
    WHEN undefined_table THEN
        RAISE NOTICE 'TimescaleDB catalog not found, creating regular indexes';
        
        CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp_composite 
        ON market_data(symbol, timestamp DESC);
        
        CREATE INDEX IF NOT EXISTS idx_market_data_timestamp_only
        ON market_data(timestamp DESC);
        
        CREATE INDEX IF NOT EXISTS idx_market_data_recent
        ON market_data(symbol, timestamp DESC)
        WHERE timestamp > NOW() - INTERVAL '30 days';
END $$;

-- ============================================================
-- News Data Indexes (using correct column name)
-- ============================================================

-- Index for recent news by symbol
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_news_data_symbol_recent
ON news_data(symbols, timestamp DESC)
WHERE timestamp > NOW() - INTERVAL '7 days';

-- ============================================================
-- Additional Performance Indexes for Existing Tables
-- ============================================================

-- Orders table - for execution monitoring
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_symbol_status_time
ON orders(symbol, status, created_at DESC);

-- Strategy performance index
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_strategy_performance_lookup
ON strategy_performance(strategy_name, timestamp DESC);

-- Model predictions index
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_model_predictions_symbol_time
ON model_predictions(symbol, timestamp DESC);

-- Social sentiment index
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_social_sentiment_symbol_time
ON social_sentiment(symbol, timestamp DESC);

-- ============================================================
-- View for Index Usage Monitoring (corrected)
-- ============================================================

-- Drop and recreate the index usage view with correct column names
DROP VIEW IF EXISTS index_usage_stats;
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
WHERE schemaname = 'public'
ORDER BY idx_scan DESC;

-- ============================================================
-- Update Statistics for Query Planner
-- ============================================================

-- Update table statistics for all existing tables
ANALYZE companies;
ANALYZE market_data;
ANALYZE orders;
ANALYZE news_data;
ANALYZE strategy_performance;
ANALYZE model_predictions;
ANALYZE social_sentiment;

-- ============================================================
-- Performance Verification
-- ============================================================

-- Show created indexes
SELECT 
    tablename,
    indexname,
    indexdef
FROM pg_indexes 
WHERE tablename IN (
    'companies', 'market_data', 'orders', 'news_data', 
    'strategy_performance', 'model_predictions', 'social_sentiment'
)
AND indexname LIKE 'idx_%'
ORDER BY tablename, indexname;