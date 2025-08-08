-- Migration: Create features table for ML feature storage
-- Date: 2025-01-10
-- Purpose: Store calculated features for machine learning models

-- Create features table if it doesn't exist
CREATE TABLE IF NOT EXISTS features (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    features JSONB NOT NULL,  -- Store features as JSON for flexibility
    feature_count INTEGER DEFAULT 0,
    metadata JSONB,  -- Optional metadata about the features
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Create unique constraint on symbol and timestamp
    CONSTRAINT unique_symbol_timestamp UNIQUE(symbol, timestamp)
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_features_symbol ON features(symbol);
CREATE INDEX IF NOT EXISTS idx_features_timestamp ON features(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_features_symbol_timestamp ON features(symbol, timestamp DESC);

-- Create GIN index for JSON features for efficient JSON queries
CREATE INDEX IF NOT EXISTS idx_features_jsonb ON features USING GIN (features);

-- Add comments for documentation
COMMENT ON TABLE features IS 'Stores calculated ML features for each symbol and timestamp';
COMMENT ON COLUMN features.symbol IS 'Stock symbol (e.g., AAPL, MSFT)';
COMMENT ON COLUMN features.timestamp IS 'Timestamp when features were calculated';
COMMENT ON COLUMN features.features IS 'JSON object containing all calculated features';
COMMENT ON COLUMN features.feature_count IS 'Number of features in the JSON object';
COMMENT ON COLUMN features.metadata IS 'Optional metadata about feature calculation';
COMMENT ON COLUMN features.created_at IS 'When the record was created';

-- Grant appropriate permissions (using current user)
-- Note: Permissions will be granted to the user running the migration

-- Verify table creation
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'features') THEN
        RAISE NOTICE 'Features table created successfully';
    ELSE
        RAISE EXCEPTION 'Failed to create features table';
    END IF;
END $$;