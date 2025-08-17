-- Migration: Update news_data table schema to support both Polygon and Alpaca providers
-- Date: 2025-07-31

-- First, backup the existing table if it has data
CREATE TABLE IF NOT EXISTS news_data_backup AS SELECT * FROM news_data;

-- Drop the old table
DROP TABLE IF EXISTS news_data;

-- Create the new schema that supports both providers
CREATE TABLE news_data (
    -- Core identifiers
    news_id VARCHAR PRIMARY KEY,  -- Format: {source}_{id}

    -- Content fields
    headline TEXT NOT NULL,
    summary TEXT,
    content TEXT,  -- Full article content

    -- Timestamps
    published_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE,  -- From Alpaca
    updated_at TIMESTAMP WITH TIME ZONE,   -- From Alpaca

    -- Source information
    source VARCHAR NOT NULL CHECK (source IN ('polygon', 'alpaca', 'yahoo')),
    publisher VARCHAR,  -- Publisher name (from Polygon)
    author VARCHAR,

    -- URLs
    url TEXT,
    image_url TEXT,
    amp_url TEXT,  -- From Polygon

    -- Associated symbols and categorization
    symbols JSONB NOT NULL DEFAULT '[]'::jsonb,  -- Array of symbols
    keywords JSONB DEFAULT '[]'::jsonb,  -- Array of keywords (from Polygon)

    -- Sentiment analysis fields
    sentiment_score FLOAT CHECK (sentiment_score >= -1 AND sentiment_score <= 1),
    sentiment_label VARCHAR CHECK (sentiment_label IN ('positive', 'negative', 'neutral')),
    sentiment_magnitude FLOAT CHECK (sentiment_magnitude >= 0),
    insights JSONB,  -- Polygon's LLM-extracted insights

    -- Relevance scoring
    relevance_score FLOAT CHECK (relevance_score >= 0 AND relevance_score <= 1),

    -- Original data storage
    raw_data JSONB,  -- Store original API response for reference

    -- Database tracking
    created_at_db TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at_db TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_news_symbols ON news_data USING GIN (symbols);
CREATE INDEX idx_news_published ON news_data (published_at DESC);
CREATE INDEX idx_news_source ON news_data (source);
CREATE INDEX idx_news_created ON news_data (created_at_db DESC);
CREATE INDEX idx_news_sentiment ON news_data (sentiment_score) WHERE sentiment_score IS NOT NULL;

-- Create composite index for symbol + date queries
-- Note: GIN doesn't support INCLUDE, so we create a regular btree index for date filtering
CREATE INDEX idx_news_published_symbol ON news_data (published_at)
    WHERE symbols IS NOT NULL;

-- Add comments for documentation
COMMENT ON TABLE news_data IS 'Unified news storage supporting multiple providers (Polygon, Alpaca)';
COMMENT ON COLUMN news_data.news_id IS 'Unique identifier: {source}_{original_id}';
COMMENT ON COLUMN news_data.symbols IS 'JSON array of stock symbols mentioned in the article';
COMMENT ON COLUMN news_data.keywords IS 'JSON array of keywords (primarily from Polygon)';
COMMENT ON COLUMN news_data.insights IS 'JSON object with LLM-extracted insights (from Polygon)';
COMMENT ON COLUMN news_data.sentiment_score IS 'Sentiment score from -1 (negative) to 1 (positive)';
COMMENT ON COLUMN news_data.raw_data IS 'Original API response stored for reference and reprocessing';

-- Create trigger to update updated_at_db
CREATE OR REPLACE FUNCTION update_updated_at_db()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at_db = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_news_data_updated_at
    BEFORE UPDATE ON news_data
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_db();

-- Grant permissions (adjust as needed)
-- GRANT SELECT ON news_data TO readonly_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON news_data TO readwrite_user;
