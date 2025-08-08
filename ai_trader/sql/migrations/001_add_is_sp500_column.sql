-- Migration: Add is_sp500 column to companies table
-- Date: 2025-07-30
-- Description: Adds is_sp500 boolean column to track S&P 500 membership

-- Add the is_sp500 column with default value false
ALTER TABLE companies 
ADD COLUMN IF NOT EXISTS is_sp500 BOOLEAN DEFAULT FALSE;

-- Add index for performance on S&P 500 queries
CREATE INDEX IF NOT EXISTS idx_companies_is_sp500 ON companies(is_sp500);

-- Add composite index for common queries (active S&P 500 stocks)
CREATE INDEX IF NOT EXISTS idx_companies_sp500_active ON companies(is_sp500, is_active) 
WHERE is_sp500 = TRUE;

-- Add comment to document the column
COMMENT ON COLUMN companies.is_sp500 IS 'Indicates if the company is currently a member of the S&P 500 index';

-- Verify the migration
DO $$
BEGIN
    -- Check if column was added successfully
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'companies' 
        AND column_name = 'is_sp500'
    ) THEN
        RAISE EXCEPTION 'Migration failed: is_sp500 column was not created';
    END IF;
    
    -- Check if index was created successfully
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes 
        WHERE tablename = 'companies' 
        AND indexname = 'idx_companies_is_sp500'
    ) THEN
        RAISE EXCEPTION 'Migration failed: idx_companies_is_sp500 index was not created';
    END IF;
    
    RAISE NOTICE 'Migration completed successfully: is_sp500 column and indexes added';
END $$;