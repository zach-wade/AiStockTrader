-- AI Trading System Database Schema
-- PostgreSQL schema definitions for trading entities

-- Create required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enum types for orders
CREATE TYPE order_side AS ENUM ('buy', 'sell');
CREATE TYPE order_type AS ENUM ('market', 'limit', 'stop', 'stop_limit');
CREATE TYPE order_status AS ENUM ('pending', 'submitted', 'partially_filled', 'filled', 'cancelled', 'rejected', 'expired');
CREATE TYPE time_in_force AS ENUM ('day', 'gtc', 'ioc', 'fok');

-- Orders table
CREATE TABLE IF NOT EXISTS orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    side order_side NOT NULL,
    order_type order_type NOT NULL,
    status order_status NOT NULL DEFAULT 'pending',
    quantity DECIMAL(18, 8) NOT NULL CHECK (quantity > 0),
    limit_price DECIMAL(18, 8) NULL CHECK (limit_price IS NULL OR limit_price > 0),
    stop_price DECIMAL(18, 8) NULL CHECK (stop_price IS NULL OR stop_price > 0),
    time_in_force time_in_force NOT NULL DEFAULT 'day',
    broker_order_id VARCHAR(100) NULL,
    filled_quantity DECIMAL(18, 8) NOT NULL DEFAULT 0 CHECK (filled_quantity >= 0),
    average_fill_price DECIMAL(18, 8) NULL CHECK (average_fill_price IS NULL OR average_fill_price > 0),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    submitted_at TIMESTAMP WITH TIME ZONE NULL,
    filled_at TIMESTAMP WITH TIME ZONE NULL,
    cancelled_at TIMESTAMP WITH TIME ZONE NULL,
    reason TEXT NULL,
    tags JSONB NULL DEFAULT '{}',

    -- Constraints
    CONSTRAINT filled_quantity_le_quantity CHECK (filled_quantity <= quantity),
    CONSTRAINT limit_order_has_price CHECK (order_type != 'limit' OR limit_price IS NOT NULL),
    CONSTRAINT stop_order_has_price CHECK (order_type != 'stop' OR stop_price IS NOT NULL),
    CONSTRAINT stop_limit_has_prices CHECK (order_type != 'stop_limit' OR (stop_price IS NOT NULL AND limit_price IS NOT NULL))
);

-- Positions table
CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    quantity DECIMAL(18, 8) NOT NULL,  -- Positive for long, negative for short
    average_entry_price DECIMAL(18, 8) NOT NULL CHECK (average_entry_price >= 0),
    current_price DECIMAL(18, 8) NULL CHECK (current_price IS NULL OR current_price > 0),
    last_updated TIMESTAMP WITH TIME ZONE NULL,
    realized_pnl DECIMAL(18, 8) NOT NULL DEFAULT 0,
    commission_paid DECIMAL(18, 8) NOT NULL DEFAULT 0 CHECK (commission_paid >= 0),
    stop_loss_price DECIMAL(18, 8) NULL CHECK (stop_loss_price IS NULL OR stop_loss_price > 0),
    take_profit_price DECIMAL(18, 8) NULL CHECK (take_profit_price IS NULL OR take_profit_price > 0),
    max_position_value DECIMAL(18, 8) NULL CHECK (max_position_value IS NULL OR max_position_value > 0),
    opened_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    closed_at TIMESTAMP WITH TIME ZONE NULL,
    strategy VARCHAR(100) NULL,
    tags JSONB NULL DEFAULT '{}',

    -- Position can't be zero quantity unless closed
    CONSTRAINT position_zero_if_closed CHECK (quantity != 0 OR closed_at IS NOT NULL)
);

-- Portfolios table
CREATE TABLE IF NOT EXISTS portfolios (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL UNIQUE,
    initial_capital DECIMAL(18, 8) NOT NULL CHECK (initial_capital > 0),
    cash_balance DECIMAL(18, 8) NOT NULL CHECK (cash_balance >= 0),
    max_position_size DECIMAL(18, 8) NOT NULL CHECK (max_position_size > 0),
    max_portfolio_risk DECIMAL(5, 4) NOT NULL CHECK (max_portfolio_risk > 0 AND max_portfolio_risk <= 1),
    max_positions INTEGER NOT NULL CHECK (max_positions > 0),
    max_leverage DECIMAL(5, 2) NOT NULL DEFAULT 1.0 CHECK (max_leverage >= 1.0),
    total_realized_pnl DECIMAL(18, 8) NOT NULL DEFAULT 0,
    total_commission_paid DECIMAL(18, 8) NOT NULL DEFAULT 0 CHECK (total_commission_paid >= 0),
    trades_count INTEGER NOT NULL DEFAULT 0 CHECK (trades_count >= 0),
    winning_trades INTEGER NOT NULL DEFAULT 0 CHECK (winning_trades >= 0),
    losing_trades INTEGER NOT NULL DEFAULT 0 CHECK (losing_trades >= 0),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    last_updated TIMESTAMP WITH TIME ZONE NULL,
    strategy VARCHAR(100) NULL,
    tags JSONB NULL DEFAULT '{}',

    -- Logical constraints
    CONSTRAINT winning_losing_le_total CHECK (winning_trades + losing_trades <= trades_count)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_broker_id ON orders(broker_order_id) WHERE broker_order_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at);
CREATE INDEX IF NOT EXISTS idx_orders_active ON orders(status) WHERE status IN ('pending', 'submitted', 'partially_filled');

CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_strategy ON positions(strategy) WHERE strategy IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_positions_opened_at ON positions(opened_at);
CREATE INDEX IF NOT EXISTS idx_positions_active ON positions(closed_at) WHERE closed_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_portfolios_name ON portfolios(name);
CREATE INDEX IF NOT EXISTS idx_portfolios_strategy ON portfolios(strategy) WHERE strategy IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_portfolios_created_at ON portfolios(created_at);

-- Market data table for storing OHLCV bars
CREATE TABLE IF NOT EXISTS market_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,  -- e.g., '1min', '5min', '1hour', '1day'
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    open DECIMAL(18, 8) NOT NULL CHECK (open > 0),
    high DECIMAL(18, 8) NOT NULL CHECK (high > 0),
    low DECIMAL(18, 8) NOT NULL CHECK (low > 0),
    close DECIMAL(18, 8) NOT NULL CHECK (close > 0),
    volume BIGINT NOT NULL CHECK (volume >= 0),
    vwap DECIMAL(18, 8) NULL CHECK (vwap IS NULL OR vwap > 0),
    trade_count INTEGER NULL CHECK (trade_count IS NULL OR trade_count >= 0),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- Data integrity constraints
    CONSTRAINT high_gte_low CHECK (high >= low),
    CONSTRAINT high_gte_open CHECK (high >= open),
    CONSTRAINT high_gte_close CHECK (high >= close),
    CONSTRAINT low_lte_open CHECK (low <= open),
    CONSTRAINT low_lte_close CHECK (low <= close),

    -- Unique constraint to prevent duplicate bars
    CONSTRAINT unique_bar UNIQUE (symbol, timeframe, timestamp)
);

-- Indexes for market data performance
CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol);
CREATE INDEX IF NOT EXISTS idx_market_data_timeframe ON market_data(timeframe);
CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timeframe_timestamp
    ON market_data(symbol, timeframe, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_created_at ON market_data(created_at DESC);

-- Partial index for frequently accessed recent data
CREATE INDEX IF NOT EXISTS idx_market_data_recent
    ON market_data(symbol, timeframe, timestamp DESC)
    WHERE timestamp > NOW() - INTERVAL '7 days';

-- Functions for common queries
CREATE OR REPLACE FUNCTION get_active_orders()
RETURNS TABLE(
    id UUID,
    symbol VARCHAR(20),
    side order_side,
    order_type order_type,
    status order_status,
    quantity DECIMAL(18, 8),
    limit_price DECIMAL(18, 8),
    stop_price DECIMAL(18, 8)
) AS $$
BEGIN
    RETURN QUERY
    SELECT o.id, o.symbol, o.side, o.order_type, o.status, o.quantity, o.limit_price, o.stop_price
    FROM orders o
    WHERE o.status IN ('pending', 'submitted', 'partially_filled')
    ORDER BY o.created_at DESC;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION get_position_pnl(p_id UUID, current_market_price DECIMAL(18, 8))
RETURNS TABLE(
    unrealized_pnl DECIMAL(18, 8),
    total_pnl DECIMAL(18, 8),
    pnl_percentage DECIMAL(8, 4)
) AS $$
DECLARE
    pos positions%ROWTYPE;
    entry_value DECIMAL(18, 8);
    current_value DECIMAL(18, 8);
    unrealized DECIMAL(18, 8);
    total DECIMAL(18, 8);
    percentage DECIMAL(8, 4);
BEGIN
    SELECT * INTO pos FROM positions WHERE id = p_id;

    IF NOT FOUND THEN
        RETURN;
    END IF;

    entry_value := ABS(pos.quantity) * pos.average_entry_price;
    current_value := ABS(pos.quantity) * current_market_price;

    -- Calculate unrealized P&L based on position direction
    IF pos.quantity > 0 THEN  -- Long position
        unrealized := current_value - entry_value;
    ELSE  -- Short position
        unrealized := entry_value - current_value;
    END IF;

    total := pos.realized_pnl + unrealized;

    -- Calculate percentage return
    IF entry_value > 0 THEN
        percentage := (total / entry_value) * 100;
    ELSE
        percentage := 0;
    END IF;

    RETURN QUERY SELECT unrealized, total, percentage;
END;
$$ LANGUAGE plpgsql;
