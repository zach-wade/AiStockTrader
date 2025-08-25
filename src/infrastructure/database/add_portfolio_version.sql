-- Add version column to portfolios table for optimistic locking
-- This migration adds thread-safety support for concurrent portfolio operations

-- Add version column if it doesn't exist
ALTER TABLE portfolios
ADD COLUMN IF NOT EXISTS version INTEGER DEFAULT 1 NOT NULL;

-- Create index on version for faster lookups during concurrent updates
CREATE INDEX IF NOT EXISTS idx_portfolios_id_version
ON portfolios(id, version);

-- Update existing portfolios to have version 1
UPDATE portfolios
SET version = 1
WHERE version IS NULL;

-- Add comment explaining the column
COMMENT ON COLUMN portfolios.version IS 'Optimistic locking version number for concurrent update safety';
