-- Migration script to add scanner_metadata column for flexible scanner data storage
-- This provides a clean way to store scanner-specific data without polluting the layer system

-- Add the scanner_metadata column
ALTER TABLE companies 
ADD COLUMN IF NOT EXISTS scanner_metadata JSONB DEFAULT '{}';

-- Add GIN index for efficient JSONB queries
CREATE INDEX IF NOT EXISTS idx_companies_scanner_metadata 
ON companies USING GIN (scanner_metadata);

-- Add index for common query patterns (e.g., finding layer 1.5 qualified symbols)
CREATE INDEX IF NOT EXISTS idx_companies_scanner_metadata_layer15 
ON companies ((scanner_metadata->'layer1_5'->>'qualified'))
WHERE scanner_metadata->'layer1_5'->>'qualified' = 'true';

-- Add comments for documentation
COMMENT ON COLUMN companies.scanner_metadata IS 
'Scanner-specific metadata including layer 1.5 affinity scores, catalyst data, and other enrichment. Structure varies by scanner type.';

-- Example structure for reference:
-- {
--   "layer1_5": {
--     "qualified": true,
--     "strategy_affinity": {
--       "momentum": 0.85,
--       "breakout": 0.72,
--       "mean_reversion": 0.45
--     },
--     "timestamp": "2025-08-07T10:30:00Z"
--   },
--   "catalyst_scanner": {
--     "earnings_date": "2025-08-15",
--     "catalyst_score": 0.78,
--     "timestamp": "2025-08-07T10:30:00Z"
--   }
-- }

-- Verify the migration
DO $$
BEGIN
    -- Check if column was added
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'companies' 
        AND column_name = 'scanner_metadata'
    ) THEN
        RAISE NOTICE 'scanner_metadata column successfully added to companies table';
    ELSE
        RAISE EXCEPTION 'Failed to add scanner_metadata column';
    END IF;
    
    -- Check if indexes were created
    IF EXISTS (
        SELECT 1 FROM pg_indexes 
        WHERE tablename = 'companies' 
        AND indexname = 'idx_companies_scanner_metadata'
    ) THEN
        RAISE NOTICE 'GIN index on scanner_metadata successfully created';
    END IF;
END
$$;