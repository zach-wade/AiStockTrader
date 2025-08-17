-- Scanner Qualifications Table Schema
-- This table tracks symbol qualification history and scanner decisions
-- Used to determine which symbols qualify for different data layers

CREATE TABLE IF NOT EXISTS scanner_qualifications (
    -- Primary key
    id SERIAL PRIMARY KEY,

    -- Symbol identification
    symbol VARCHAR(10) NOT NULL,

    -- Qualification details
    layer_qualified INTEGER NOT NULL,  -- Layer level (0-3) the symbol qualified for
    qualification_date DATE NOT NULL,  -- Date of qualification

    -- Qualification metrics at time of qualification
    liquidity_score NUMERIC(10,2),
    avg_daily_volume BIGINT,
    avg_dollar_volume NUMERIC(20,2),
    affinity_score NUMERIC(5,4),      -- Score for user interest/affinity
    catalyst_score NUMERIC(5,4),      -- Score for potential catalysts

    -- Retention configuration
    retention_days INTEGER NOT NULL DEFAULT 30,  -- How long to retain data
    extended_retention BOOLEAN DEFAULT false,    -- Whether retention was extended
    extended_until DATE,                        -- Date retention is extended until

    -- Metadata and tracking
    metadata JSONB,                             -- Additional qualification metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Unique constraint to prevent duplicate qualifications on same date
    CONSTRAINT scanner_qualifications_symbol_qualification_date_key
        UNIQUE (symbol, qualification_date)
);

-- Performance indexes
CREATE INDEX idx_scanner_qual_symbol ON scanner_qualifications(symbol);
CREATE INDEX idx_scanner_qual_date ON scanner_qualifications(qualification_date DESC);
CREATE INDEX idx_scanner_qual_layer ON scanner_qualifications(layer_qualified);
CREATE INDEX idx_scanner_qual_symbol_date ON scanner_qualifications(symbol, qualification_date DESC);

-- Extended retention tracking
CREATE INDEX idx_scanner_qual_extended ON scanner_qualifications(extended_retention)
    WHERE extended_retention = true;

-- Comments
COMMENT ON TABLE scanner_qualifications IS 'Historical record of symbol qualifications for different data layers';
COMMENT ON COLUMN scanner_qualifications.symbol IS 'Stock ticker symbol';
COMMENT ON COLUMN scanner_qualifications.layer_qualified IS 'Data layer the symbol qualified for (0=Basic, 1=Liquid, 2=Catalyst, 3=Active)';
COMMENT ON COLUMN scanner_qualifications.qualification_date IS 'Date when the qualification was determined';
COMMENT ON COLUMN scanner_qualifications.liquidity_score IS 'Liquidity score at time of qualification';
COMMENT ON COLUMN scanner_qualifications.avg_daily_volume IS 'Average daily trading volume at qualification';
COMMENT ON COLUMN scanner_qualifications.avg_dollar_volume IS 'Average daily dollar volume at qualification';
COMMENT ON COLUMN scanner_qualifications.affinity_score IS 'User interest/watchlist affinity score';
COMMENT ON COLUMN scanner_qualifications.catalyst_score IS 'Potential catalyst event score';
COMMENT ON COLUMN scanner_qualifications.retention_days IS 'Number of days to retain detailed data for this symbol';
COMMENT ON COLUMN scanner_qualifications.extended_retention IS 'Flag indicating if retention period was extended';
COMMENT ON COLUMN scanner_qualifications.extended_until IS 'Date until which extended retention is valid';
COMMENT ON COLUMN scanner_qualifications.metadata IS 'Additional JSON metadata about the qualification';

-- Example metadata structure:
-- {
--   "qualification_reason": "High volume spike detected",
--   "scanner_type": "volume_scanner",
--   "threshold_values": {
--     "volume_threshold": 1000000,
--     "liquidity_threshold": 0.8
--   },
--   "source_scanners": ["volume_scanner", "liquidity_scanner"],
--   "promotion_from_layer": 0,
--   "metrics_snapshot": {
--     "spread_avg": 0.02,
--     "trades_per_minute": 150
--   }
-- }
