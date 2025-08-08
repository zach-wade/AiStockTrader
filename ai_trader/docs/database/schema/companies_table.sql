-- Companies Table Schema
-- This table stores information about all tracked companies/symbols
-- including their market data, fundamental metrics, and qualification status

CREATE TABLE IF NOT EXISTS companies (
    -- Primary identifier
    symbol VARCHAR(10) NOT NULL PRIMARY KEY,
    
    -- Basic company information
    name VARCHAR(255),
    exchange VARCHAR(50),
    sector VARCHAR(100),
    industry VARCHAR(100),
    ipo_date DATE,
    
    -- Market metrics
    market_cap NUMERIC(20),
    avg_daily_volume BIGINT,
    avg_dollar_volume NUMERIC(20,2),
    
    -- Status flags
    is_active BOOLEAN DEFAULT true,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Company details
    quote_type VARCHAR(50),
    website VARCHAR(255),
    country VARCHAR(100),
    state VARCHAR(50),
    city VARCHAR(100),
    zip VARCHAR(20),
    full_time_employees INTEGER,
    currency VARCHAR(10),
    
    -- Valuation metrics
    enterprise_value NUMERIC(20),
    trailing_pe NUMERIC(12,4),
    forward_pe NUMERIC(12,4),
    price_to_book NUMERIC(20,6),
    beta NUMERIC(10,3),
    
    -- Dividend information
    dividend_rate NUMERIC(10,4),
    dividend_yield NUMERIC(10,6),
    payout_ratio NUMERIC(10,4),
    
    -- Volume and shares data
    average_volume NUMERIC(20),
    float_shares NUMERIC(20),
    shares_outstanding NUMERIC(20),
    short_percent_of_float NUMERIC(10,6),
    
    -- Analyst recommendations
    recommendation_key VARCHAR(20),
    recommendation_mean NUMERIC(5,2),
    number_of_analysts INTEGER,
    target_mean_price NUMERIC(12,2),
    
    -- Price data
    data_last_updated TIMESTAMP WITH TIME ZONE,
    current_price NUMERIC(12,4),
    previous_close NUMERIC(12,4),
    price_last_updated TIMESTAMP WITH TIME ZONE,
    
    -- Trading characteristics
    liquidity_score NUMERIC(10,4),
    easy_to_borrow BOOLEAN DEFAULT false,
    marginable BOOLEAN DEFAULT false,
    shortable BOOLEAN DEFAULT false,
    
    -- Scanner scores and metadata
    catalyst_score NUMERIC(10,2) DEFAULT 0,
    premarket_score DOUBLE PRECISION,
    rvol DOUBLE PRECISION,
    
    -- Data coverage tracking
    data_coverage_metadata JSONB,
    data_coverage_last_analyzed TIMESTAMP WITH TIME ZONE,
    
    -- Index membership
    is_sp500 BOOLEAN DEFAULT false,
    
    -- Layer system (0-3 qualification levels)
    layer INTEGER DEFAULT 0,
    layer_updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    layer_reason TEXT,
    
    -- Additional scanner metadata
    scanner_metadata JSONB
);

-- Performance indexes
CREATE INDEX idx_companies_active ON companies(is_active);
CREATE INDEX idx_companies_layer ON companies(layer) WHERE is_active = true;
CREATE INDEX idx_companies_layer_active ON companies(layer, is_active);
CREATE INDEX idx_companies_layer_updated ON companies(layer_updated_at);
CREATE INDEX idx_companies_market_cap ON companies(market_cap DESC);
CREATE INDEX idx_companies_volume ON companies(avg_dollar_volume DESC);
CREATE INDEX idx_companies_liquidity_score ON companies(liquidity_score DESC NULLS LAST);
CREATE INDEX idx_companies_catalyst_score ON companies(catalyst_score DESC) WHERE catalyst_score > 0;

-- Sector/industry indexes
CREATE INDEX idx_companies_sector ON companies(sector);
CREATE INDEX idx_companies_country ON companies(country);
CREATE INDEX idx_companies_quote_type ON companies(quote_type);

-- Financial metrics indexes
CREATE INDEX idx_companies_beta ON companies(beta);
CREATE INDEX idx_companies_dividend_yield ON companies(dividend_yield DESC);
CREATE INDEX idx_companies_recommendation ON companies(recommendation_mean);
CREATE INDEX idx_companies_current_price ON companies(current_price);

-- Data tracking indexes
CREATE INDEX idx_companies_ipo_date ON companies(ipo_date);
CREATE INDEX idx_companies_data_updated ON companies(data_last_updated);
CREATE INDEX idx_companies_coverage_last_analyzed ON companies(data_coverage_last_analyzed);

-- S&P 500 index
CREATE INDEX idx_companies_is_sp500 ON companies(is_sp500);
CREATE INDEX idx_companies_sp500_active ON companies(is_sp500, is_active) WHERE is_sp500 = true;

-- JSONB indexes for metadata
CREATE INDEX idx_companies_coverage_metadata ON companies USING gin(data_coverage_metadata);

-- Comments
COMMENT ON TABLE companies IS 'Master table for all tracked companies and symbols';
COMMENT ON COLUMN companies.symbol IS 'Stock ticker symbol (primary key)';
COMMENT ON COLUMN companies.layer IS 'Data layer qualification (0=Basic, 1=Liquid, 2=Catalyst, 3=Active)';
COMMENT ON COLUMN companies.liquidity_score IS 'Calculated liquidity score based on volume and spread';
COMMENT ON COLUMN companies.catalyst_score IS 'Score indicating likelihood of price movement catalysts';
COMMENT ON COLUMN companies.rvol IS 'Relative volume indicator';
COMMENT ON COLUMN companies.scanner_metadata IS 'JSON metadata from various scanner systems';
COMMENT ON COLUMN companies.data_coverage_metadata IS 'JSON tracking which data types are available';