-- Migration script to create layer_transitions table for event sourcing
-- This table tracks all layer changes for audit and analytics purposes

-- Create the layer_transitions table
CREATE TABLE IF NOT EXISTS layer_transitions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    from_layer INTEGER,  -- NULL for initial qualification
    to_layer INTEGER NOT NULL CHECK (to_layer >= 0 AND to_layer <= 3),
    reason TEXT,
    metadata JSONB,
    transitioned_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    transitioned_by VARCHAR(100),  -- scanner name, user, or system component

    -- Add indexes for common queries
    CONSTRAINT valid_layer_transition CHECK (
        (from_layer IS NULL OR (from_layer >= 0 AND from_layer <= 3)) AND
        (from_layer IS NULL OR from_layer != to_layer)
    )
);

-- Create indexes for efficient querying
CREATE INDEX idx_layer_transitions_symbol ON layer_transitions(symbol);
CREATE INDEX idx_layer_transitions_symbol_time ON layer_transitions(symbol, transitioned_at DESC);
CREATE INDEX idx_layer_transitions_to_layer ON layer_transitions(to_layer);
CREATE INDEX idx_layer_transitions_from_layer ON layer_transitions(from_layer);
CREATE INDEX idx_layer_transitions_time ON layer_transitions(transitioned_at DESC);
CREATE INDEX idx_layer_transitions_metadata ON layer_transitions USING GIN (metadata);

-- Create a view for current layer assignments from transitions
CREATE OR REPLACE VIEW current_layer_from_transitions AS
WITH latest_transitions AS (
    SELECT DISTINCT ON (symbol)
        symbol,
        to_layer as layer,
        reason,
        transitioned_at,
        transitioned_by
    FROM layer_transitions
    ORDER BY symbol, transitioned_at DESC
)
SELECT * FROM latest_transitions;

-- Create a function to get layer history for a symbol
CREATE OR REPLACE FUNCTION get_layer_history(p_symbol VARCHAR)
RETURNS TABLE (
    from_layer INTEGER,
    to_layer INTEGER,
    reason TEXT,
    transitioned_at TIMESTAMP WITH TIME ZONE,
    transitioned_by VARCHAR,
    metadata JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        lt.from_layer,
        lt.to_layer,
        lt.reason,
        lt.transitioned_at,
        lt.transitioned_by,
        lt.metadata
    FROM layer_transitions lt
    WHERE lt.symbol = p_symbol
    ORDER BY lt.transitioned_at DESC;
END;
$$ LANGUAGE plpgsql;

-- Create a function to get layer movement statistics
CREATE OR REPLACE FUNCTION get_layer_movement_stats(
    p_start_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_DATE - INTERVAL '30 days',
    p_end_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
)
RETURNS TABLE (
    movement_type TEXT,
    from_layer INTEGER,
    to_layer INTEGER,
    count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        CASE
            WHEN lt.from_layer IS NULL THEN 'initial_qualification'
            WHEN lt.to_layer > COALESCE(lt.from_layer, -1) THEN 'promotion'
            WHEN lt.to_layer < lt.from_layer THEN 'demotion'
            ELSE 'unknown'
        END as movement_type,
        lt.from_layer,
        lt.to_layer,
        COUNT(*) as count
    FROM layer_transitions lt
    WHERE lt.transitioned_at BETWEEN p_start_date AND p_end_date
    GROUP BY movement_type, lt.from_layer, lt.to_layer
    ORDER BY movement_type, lt.from_layer, lt.to_layer;
END;
$$ LANGUAGE plpgsql;

-- Add comments for documentation
COMMENT ON TABLE layer_transitions IS 'Event sourcing table tracking all layer changes for symbols';
COMMENT ON COLUMN layer_transitions.symbol IS 'Trading symbol';
COMMENT ON COLUMN layer_transitions.from_layer IS 'Previous layer (NULL for initial qualification)';
COMMENT ON COLUMN layer_transitions.to_layer IS 'New layer (0=BASIC, 1=LIQUID, 2=CATALYST, 3=ACTIVE)';
COMMENT ON COLUMN layer_transitions.reason IS 'Human-readable reason for the transition';
COMMENT ON COLUMN layer_transitions.metadata IS 'Additional data like metrics, scores, or triggers';
COMMENT ON COLUMN layer_transitions.transitioned_by IS 'Component that triggered the transition';
