-- Create news_data table for storing news articles
CREATE TABLE IF NOT EXISTS news_data (
    id SERIAL PRIMARY KEY,
    article_id VARCHAR(255) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    summary TEXT,
    url TEXT,
    author VARCHAR(255),
    publisher VARCHAR(255),
    published_at TIMESTAMP WITH TIME ZONE NOT NULL,
    symbols TEXT[],  -- Array of stock symbols mentioned
    sentiment_score NUMERIC(5, 4),  -- Sentiment score between -1 and 1
    sentiment_label VARCHAR(50),    -- e.g., 'positive', 'negative', 'neutral'
    keywords TEXT[],                -- Array of keywords
    source VARCHAR(50) NOT NULL,    -- e.g., 'polygon', 'alpaca', 'benzinga'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_news_data_symbols ON news_data USING GIN(symbols);
CREATE INDEX IF NOT EXISTS idx_news_data_published_at ON news_data(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_data_source ON news_data(source);
CREATE INDEX IF NOT EXISTS idx_news_data_article_id ON news_data(article_id);
CREATE INDEX IF NOT EXISTS idx_news_data_sentiment_score ON news_data(sentiment_score);

-- Create compound index for common queries
CREATE INDEX IF NOT EXISTS idx_news_data_symbol_published 
ON news_data(published_at DESC) 
WHERE symbols IS NOT NULL;

-- Add trigger to update updated_at
CREATE OR REPLACE FUNCTION update_news_data_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_news_data_timestamp
BEFORE UPDATE ON news_data
FOR EACH ROW
EXECUTE FUNCTION update_news_data_updated_at();

-- Add comment
COMMENT ON TABLE news_data IS 'Stores news articles and sentiment data for stocks';