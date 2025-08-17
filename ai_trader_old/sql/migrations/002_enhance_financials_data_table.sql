-- Migration: Enhance financials_data table for Polygon quarterly data support
-- Date: 2025-08-03
-- Purpose: Add additional columns to store key financial metrics and metadata from Polygon API

-- Add filing date and source tracking columns
ALTER TABLE financials_data
ADD COLUMN IF NOT EXISTS filing_date DATE,
ADD COLUMN IF NOT EXISTS source VARCHAR(50),
ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP;

-- Add key financial metrics columns
ALTER TABLE financials_data
ADD COLUMN IF NOT EXISTS gross_profit BIGINT,
ADD COLUMN IF NOT EXISTS operating_income BIGINT,
ADD COLUMN IF NOT EXISTS eps_basic NUMERIC(10, 4),
ADD COLUMN IF NOT EXISTS eps_diluted NUMERIC(10, 4),
ADD COLUMN IF NOT EXISTS current_assets BIGINT,
ADD COLUMN IF NOT EXISTS current_liabilities BIGINT,
ADD COLUMN IF NOT EXISTS stockholders_equity BIGINT;

-- Create index on filing_date for temporal queries
CREATE INDEX IF NOT EXISTS idx_financials_data_filing_date
ON financials_data(filing_date DESC);

-- Create index on source for filtering by data provider
CREATE INDEX IF NOT EXISTS idx_financials_data_source
ON financials_data(source);

-- Create composite index for common query patterns
CREATE INDEX IF NOT EXISTS idx_financials_data_symbol_filing_date
ON financials_data(symbol, filing_date DESC);

-- Add comment to table documenting the enhancement
COMMENT ON TABLE financials_data IS 'Financial statements data from Yahoo (annual) and Polygon (quarterly/annual). Enhanced to support Polygon quarterly data with additional metrics.';

-- Add comments to new columns
COMMENT ON COLUMN financials_data.filing_date IS 'SEC filing date for the financial statement';
COMMENT ON COLUMN financials_data.source IS 'Data source: yahoo, polygon, etc.';
COMMENT ON COLUMN financials_data.gross_profit IS 'Gross profit in cents (Revenue - Cost of Revenue)';
COMMENT ON COLUMN financials_data.operating_income IS 'Operating income/EBIT in cents';
COMMENT ON COLUMN financials_data.eps_basic IS 'Basic earnings per share';
COMMENT ON COLUMN financials_data.eps_diluted IS 'Diluted earnings per share';
COMMENT ON COLUMN financials_data.current_assets IS 'Current assets in cents';
COMMENT ON COLUMN financials_data.current_liabilities IS 'Current liabilities in cents';
COMMENT ON COLUMN financials_data.stockholders_equity IS 'Total stockholders equity in cents';
